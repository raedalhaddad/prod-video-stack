from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Tuple
from urllib.request import urlopen

import numpy as np


class FrameStream(Protocol):
    def start(self) -> None: ...
    def read(self) -> Optional[tuple]: ...
    def close(self) -> None: ...


# Callable that maps whatever timestamp is in item -> pts_ms (epoch ms)
PtsMapper = Callable[[float], float]


@dataclass
class ReaderStats:
    frames_in: int = 0
    frames_out: int = 0
    drops: int = 0
    last_frame_age_ms: float = 0.0
    read_loop_us_mean: float = 0.0
    read_loop_us_p95: float = 0.0
    _loop_us_hist: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)

    def update_loop_us(self, dt_us: float) -> None:
        self._loop_us_hist.append(dt_us)
        if self._loop_us_hist:
            arr = np.fromiter(self._loop_us_hist, dtype=np.float64)
            self.read_loop_us_mean = float(arr.mean())
            self.read_loop_us_p95 = float(np.percentile(arr, 95))


def _default_pts_mapper(x: float) -> float:
    # Assume already epoch-ms and nonzero
    return float(x)


def _extract_timebase(data: dict):
    tb = data.get("timebase") or data
    be = float(tb["base_epoch_ms"])
    bp = float(tb.get("base_pts_ns", tb.get("base_pts")))
    units = str(tb.get("pts_units", "ns")).lower()
    return be, bp, units


def make_pts_mapper_from_health(health_url: str, timeout_s: float = 1.0):
    try:
        with urlopen(health_url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        base_epoch_ms, base_pts, units = _extract_timebase(data)
    except Exception:
        # identity fallback; age may be meaningless
        return lambda x: float(x)

    unit_scale_ms = {
        "ns": 1e-6,
        "us": 1e-3,
        "ms": 1.0,
        "epoch_ms": None,
        "epoch_s": "sec",
    }.get(units, 1e-6)

    if unit_scale_ms is None:

        def fn(x: float) -> float:
            return float(x)  # already epoch-ms

    elif unit_scale_ms == "sec":

        def fn(x: float) -> float:
            return float(x) * 1000.0  # epoch seconds → ms

    else:
        sc = float(unit_scale_ms)

        def fn(x: float, be: float = base_epoch_ms, bp: float = base_pts, sc: float = sc) -> float:
            return be + (float(x) - bp) * sc

    # mark authoritative
    try:
        fn._authoritative = True
        fn._desc = f"{units} base_epoch_ms={base_epoch_ms:.0f} base_pts={base_pts:.0f}"
    except Exception:
        pass
    return fn


class WithStats:
    """
    Wrap any FrameStream and expose stats().
    - Understands tuple shapes:
        (frame, pts, fid)
        (frame, pts, fid, last_age_ms)
      where 'pts' may be epoch-ms or ns; we normalize via pts_mapper.
    """

    def __init__(self, inner: FrameStream, pts_mapper: Optional[PtsMapper] = None) -> None:
        self._inner = inner
        self._stats = ReaderStats()
        self._pts_mapper = pts_mapper or _default_pts_mapper
        self._calibrated = False
        # default mapping until we calibrate
        self._chosen_fn = self._pts_mapper
        # debug breadcrumbs for demo
        self._last_pts_raw = None
        self._last_pts_ms = None
        self._candidates = None  # filled on first use
        # local timing buffer (µs) for mean/p95; do NOT rely on ReaderStats internals
        self._accum_us = deque(maxlen=512)
        # --- skew learning (ms) ---
        self._skew_samples = deque(maxlen=16)  # collect a few “future” readings
        self._offset_ms = 0.0  # learned correction
        self._skew_applied = False  # apply once when stable

    def _update_loop_stats(self) -> None:
        """Compute mean/p95 from our local buffer and publish onto ReaderStats."""
        n = len(self._accum_us)
        if n == 0:
            return
        mean_us = sum(self._accum_us) / n
        s = sorted(self._accum_us)
        p95_us = s[int(0.95 * (n - 1))]
        # publish (create attrs if they don't exist)
        try:
            self._stats.read_loop_us_mean = float(mean_us)
            self._stats.read_loop_us_p95 = float(p95_us)
        except Exception:
            pass

    def _build_candidates(self) -> list[tuple[str, PtsMapper]]:
        # Try common encodings, best-first.
        cands: list[tuple[str, PtsMapper]] = [
            ("epoch_ms", lambda x: float(x)),  # already epoch-ms
            ("epoch_s", lambda x: float(x) * 1_000.0),  # epoch seconds
        ]
        # If the caller provided a health-based mapper, also try ns/us/ms deltas
        if self._pts_mapper is not _default_pts_mapper:
            base_map = self._pts_mapper
            # If user gave a health ns-mapper, keep it first in health group
            cands.extend(
                [
                    ("health_ns", base_map),
                    ("health_us", lambda x: base_map(x * 1_000.0)),
                    ("health_ms", lambda x: base_map(x * 1_000_000.0)),
                ]
            )
        return cands

    def _calibrate(self, raw_pts: float) -> None:
        """
        If the provided pts_mapper is marked authoritative (from /health),
        trust it and skip heuristic guessing. Otherwise fall back to your
        existing candidate selection.
        """
        if getattr(self._pts_mapper, "_authoritative", False):
            self._chosen_fn = self._pts_mapper
            self._calibrated = True
            self._chosen_name = "health_authoritative"
            return
        # ---- keep your existing heuristic below, unchanged ----
        self._candidates = self._build_candidates()
        now_ms = time.time() * 1000.0
        best = None
        best_age = None
        for name, fn in self._candidates:
            try:
                age = now_ms - float(fn(raw_pts))
            except Exception:
                continue
            if 0 <= age <= 120_000:
                if best_age is None or age < best_age:
                    best = (name, fn)
                    best_age = age
            else:
                if best_age is None or abs(age) < abs(best_age):
                    best = (name, fn)
                    best_age = age
        self._chosen_name, self._chosen_fn = best if best else ("identity", lambda x: float(x))
        self._calibrated = True

    def start(self) -> None:
        self._inner.start()

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        t0 = time.perf_counter()
        tup = self._inner.read()
        if tup is None:
            # record loop timing even on idle
            self._accum_us.append((time.perf_counter() - t0) * 1e6)
            self._update_loop_stats()
            return None
        frame, raw_pts, fid = tup  # raw_pts is *raw* (ns/us/ms/epoch) from transport
        if not self._calibrated:
            self._calibrate(raw_pts)
        # map raw → epoch-ms using the chosen function
        pts_ms0 = float(self._chosen_fn(raw_pts))
        now_ms = time.time() * 1000.0
        age0 = now_ms - pts_ms0
        # Learn small positive skew (producer clock ahead) once, then fix permanently
        if not self._skew_applied:
            if -5000.0 < age0 < -50.0:  # frame appears “in the future”
                self._skew_samples.append(-age0)  # collect positive skew (ms)
            if len(self._skew_samples) >= 5:
                s = sorted(self._skew_samples)
                self._offset_ms = s[len(s) // 2]  # median is robust
                self._skew_applied = True
                with contextlib.suppress(Exception):
                    self._stats.skew_ms = float(self._offset_ms)
        # Apply correction (no-op until _skew_applied flips)
        pts_ms = pts_ms0 - self._offset_ms
        # expose for demo diagnostics
        self._last_pts_raw = float(raw_pts)
        self._last_pts_ms = pts_ms
        # update freshness every frame
        self._stats.last_frame_age_ms = max(0.0, now_ms - pts_ms)
        # counters + loop timing
        self._stats.frames_in += 1
        self._stats.frames_out += 1
        self._accum_us.append((time.perf_counter() - t0) * 1e6)
        self._update_loop_stats()
        return frame, pts_ms, fid

    def close(self) -> None:
        self._inner.close()

    def stats(self) -> ReaderStats:
        return self._stats


def wrap_with_stats(stream: FrameStream, pts_mapper: Optional[PtsMapper] = None) -> FrameStream:
    stats_fn = getattr(stream, "stats", None)
    return stream if callable(stats_fn) else WithStats(stream, pts_mapper=pts_mapper)
