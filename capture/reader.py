from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Literal, Optional, Protocol, Tuple

import numpy as np

DropPolicy = Literal["drop_new", "drop_old"]


class FrameStream(Protocol):
    def start(self) -> None: ...
    def read(self) -> Optional[Tuple[np.ndarray, float, int]]: ...  # (frame_bgr, pts_ms, frame_id)
    def close(self) -> None: ...


@dataclass
class ReaderStats:
    frames_in: int = 0
    frames_out: int = 0
    drops: int = 0
    last_frame_age_ms: float = 0.0
    read_loop_us_mean: float = 0.0
    read_loop_us_p95: float = 0.0
    _loop_us_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=1000), repr=False)

    def update_loop_us(self, dt_us: float) -> None:
        self._loop_us_hist.append(dt_us)
        if self._loop_us_hist:
            arr = np.fromiter(self._loop_us_hist, dtype=np.float64)
            self.read_loop_us_mean = float(arr.mean())
            self.read_loop_us_p95 = float(np.percentile(arr, 95))


class NullTransport:
    """A tiny source that synthesizes black frames. Useful for tests/dev."""

    def __init__(self, width: int = 640, height: int = 360, fps: float = 15.0):
        self.width, self.height, self.fps = width, height, fps
        self._running = False
        self._next_ts = 0.0
        self._frame_id = 0
        self._stats = ReaderStats()

    def start(self) -> None:
        self._running = True
        self._next_ts = time.time() * 1000.0

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        if not self._running:
            return None
        now_ms = time.time() * 1000.0
        if now_ms < self._next_ts:
            return None
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pts_ms = now_ms
        fid = self._frame_id
        self._frame_id += 1
        self._next_ts += 1000.0 / max(self.fps, 0.001)
        self._stats.frames_in += 1
        self._stats.frames_out += 1
        self._stats.last_frame_age_ms = 0.0
        return frame, pts_ms, fid

    def close(self) -> None:
        self._running = False

    def stats(self) -> ReaderStats:
        return self._stats


# --- SHM Transport scaffolding ------------------------------------------------


@dataclass
class ShmParams:
    map_name: str
    width: int
    height: int
    stride: int
    fmt: str  # 'BGR','BGRx'/'BGRA','RGB','RGBx'/'RGBA' supported
    ring_slots: int
    base_epoch_ms: float
    base_pts_ns: int


class ShmTransport:
    """
    SHM reader with bounded, non-blocking queue and single-copy BGRx→BGR conversion.

    Notes:
      * This class expects a ring-reader callback that returns (bgrx_bytes, pts_ns, frame_id, last_frame_age_ms)
        or None if no new frame. Wire it up using your native package during discovery.
    """

    def __init__(
        self,
        params: ShmParams,
        ring_reader: Callable[[], Optional[Tuple[memoryview, int, int, float]]],
        queue_max: int = 3,
        drop_policy: DropPolicy = "drop_new",
    ) -> None:
        self.params = params
        self._ring_reader = ring_reader
        self._queue: Deque[Tuple[np.ndarray, float, int]] = deque(maxlen=queue_max)
        self._drop_policy = drop_policy
        self._run_ev = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._stats = ReaderStats()

    # Bytes → BGR (drop alpha if present; reorder if RGB), single final copy to detach from SHM
    def _bytes_to_bgr(self, raw_bytes: memoryview) -> np.ndarray:
        p = self.params
        h, w, stride = p.height, p.width, p.stride
        fmt = (p.fmt or "BGR").upper()

        if fmt in ("BGRX", "BGRA"):
            bpp = 4
            take = w * bpp
            buf = np.frombuffer(raw_bytes, dtype=np.uint8, count=h * stride).reshape(h, stride)
            view = buf[:, :take].reshape(h, w, bpp)
            return view[:, :, :3].copy(order="C")

        if fmt == "BGR":
            bpp = 3
            take = w * bpp
            buf = np.frombuffer(raw_bytes, dtype=np.uint8, count=h * stride).reshape(h, stride)
            view = buf[:, :take].reshape(h, w, bpp)
            return view.copy(order="C")

        if fmt in ("RGBX", "RGBA"):
            bpp = 4
            take = w * bpp
            buf = np.frombuffer(raw_bytes, dtype=np.uint8, count=h * stride).reshape(h, stride)
            view = buf[:, :take].reshape(h, w, bpp)
            # convert RGBx/RGBA → BGR
            return view[:, :, :3][:, :, ::-1].copy(order="C")

        if fmt == "RGB":
            bpp = 3
            take = w * bpp
            buf = np.frombuffer(raw_bytes, dtype=np.uint8, count=h * stride).reshape(h, stride)
            view = buf[:, :take].reshape(h, w, bpp)
            return view[:, :, ::-1].copy(order="C")

        raise RuntimeError(
            f"Unsupported SHM pixel format: {p.fmt} (expected BGR/BGRx/BGRA/RGB/RGBx/RGBA)"
        )

    def _ingest_loop(self) -> None:
        self._run_ev.set()
        while self._run_ev.is_set():
            t0 = time.perf_counter()
            tup = self._ring_reader()
            if tup is None:
                # no frame; short spin without sleep to keep latency minimal
                # rely on bounded queue to prevent memory growth
                self._stats.update_loop_us((time.perf_counter() - t0) * 1e6)
                continue
            slot_bytes, pts_ns, fid, last_age_ms = tup
            try:
                frame = self._bytes_to_bgr(slot_bytes)
            except Exception:
                # If format mismatch or other error, skip this frame
                self._stats.drops += 1
                self._stats.update_loop_us((time.perf_counter() - t0) * 1e6)
                continue
            pts_ms = self.params.base_epoch_ms + (pts_ns - self.params.base_pts_ns) / 1e6
            item = (frame, float(pts_ms), int(fid))
            self._stats.frames_in += 1
            self._stats.last_frame_age_ms = float(last_age_ms)
            if len(self._queue) == self._queue.maxlen:
                # bounded queue: apply drop policy
                self._stats.drops += 1
                if self._drop_policy == "drop_new":
                    # drop the new frame (do nothing)
                    pass
                else:  # drop_old
                    self._queue.popleft()
                    self._queue.append(item)
            else:
                self._queue.append(item)
            self._stats.update_loop_us((time.perf_counter() - t0) * 1e6)

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._thr = threading.Thread(target=self._ingest_loop, name="shm-reader", daemon=True)
        self._thr.start()

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        if not self._thr or not self._thr.is_alive():
            return None
        if not self._queue:
            return None
        self._stats.frames_out += 1
        return self._queue.popleft()

    def close(self) -> None:
        if self._thr:
            self._run_ev.clear()
            self._thr.join(timeout=1.0)
            self._thr = None

    def stats(self) -> ReaderStats:
        return self._stats


# --- Discovery ---------------------------------------------------------------


@dataclass
class ReaderConfig:
    prefer: str = "shm"  # or "null"
    ctrl_pipe: str = r"\\.\pipe\rtva_cam0_ctrl"
    map_name: Optional[str] = None
    health_url: str = "http://127.0.0.1:8765/health"
    queue_max: int = 3
    drop_policy: DropPolicy = "drop_new"
    connect_timeout_ms: int = 1500


class ReaderFactory:
    @staticmethod
    def from_config(cfg: ReaderConfig) -> FrameStream:
        if cfg.prefer == "null":
            return NullTransport()
        # 1) Control-pipe handshake (canonical)
        meta = _try_ctrl_handshake(cfg.ctrl_pipe, timeout_ms=cfg.connect_timeout_ms)
        if meta is None:
            # 2) /health fallback
            meta = _try_fetch_health(cfg.health_url)
        if meta is None:
            # 3) Naming fallback
            meta = {
                "map_name": cfg.map_name or "Local\\rtva_cam0",
                "width": 1920,
                "height": 1080,
                "stride": 1920 * 4,
                "format": "BGRx",
                "ring_slots": 3,
                "base_epoch_ms": time.time() * 1000.0,
                "base_pts_ns": 0,
            }
        # pick a sane default stride if not provided: 4 bytes for *x/alpha formats, else 3
        fmt = str(meta.get("format", "BGRx"))
        fmt_u = fmt.upper()
        default_stride = int(meta.get("width", 1920)) * (
            4 if fmt_u in ("BGRX", "BGRA", "RGBX", "RGBA") else 3
        )
        params = ShmParams(
            map_name=meta["map_name"],
            width=int(meta.get("width", 1920)),
            height=int(meta.get("height", 1080)),
            stride=int(meta.get("stride", default_stride)),
            fmt=fmt,
            ring_slots=int(meta.get("ring_slots", 3)),
            base_epoch_ms=float(meta.get("base_epoch_ms", time.time() * 1000.0)),
            base_pts_ns=int(meta.get("base_pts_ns", 0)),
        )
        supported = {"BGR", "BGRX", "BGRA", "RGB", "RGBX", "RGBA"}
        if params.fmt.upper() not in supported:
            raise RuntimeError(
                f"Unsupported SHM pixel format: {params.fmt} (supported: {', '.join(sorted(supported))})"
            )
        # Prefer the native shm reader shipped with the gst_cap package
        try:
            from capture.shm_reader import ShmTransport as NativeShmTransport

            # Derive ring metadata for no-pipe mode
            header_size = 40  # per HEADER_FMT in shm_reader.py
            slot_bytes = header_size + params.stride * params.height
            slot_count = params.ring_slots

            # Treat "", "none", "null", "disabled", "off" as *no control-pipe handshake*. Native expects a string.
            _raw = (cfg.ctrl_pipe or "").strip()
            _ctrl_pipe = (
                "" if _raw.lower() in ("", "none", "null", "disabled", "off") else cfg.ctrl_pipe
            )
            return NativeShmTransport(
                map_name=params.map_name,
                ctrl_pipe=_ctrl_pipe,  # "" disables handshake in native code
                # ring metadata (lets native open SHM without the pipe)
                slot_count=slot_count,
                slot_bytes=slot_bytes,
                fmt=params.fmt,
                width=params.width,
                height=params.height,
                stride=params.stride,
            )
        except Exception as e:
            print(
                f"[reader.discovery] WARN: native shm_reader unavailable ({e}); using NullTransport."
            )
            return NullTransport()


# --- Helpers -----------------------------------------------------------------


def _try_ctrl_handshake(ctrl_pipe: str, timeout_ms: int = 1500) -> Optional[dict]:
    """
    Attempt a JSON handshake over the named control pipe. We keep this stdlib-only; if pywin32
    is available in your environment you can replace this with a robust implementation.
    Protocol (by convention): send a single line 'HELLO\n', read a single JSON line.
    """
    try:
        # Non-blocking open with timeout via polling
        t_deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < t_deadline:
            try:
                # Use a context manager per SIM115
                with open(ctrl_pipe, "r+b", buffering=0) as fh:
                    fh.write(b"HELLO\n")
                    fh.flush()
                    raw = fh.readline()  # read a single line of JSON
                    if not raw:
                        return None
                    try:
                        data = json.loads(raw.decode("utf-8"))
                    except Exception:
                        return None
                    shm = data.get("shm", {})
                    tb = data.get("timebase", {})
                    return {
                        "map_name": shm.get("map_name"),
                        "width": int(shm.get("width", 1920)),
                        "height": int(shm.get("height", 1080)),
                        "stride": int(shm.get("stride", shm.get("width", 1920) * 4)),
                        "format": str(shm.get("format", "BGRx")),
                        "ring_slots": int(shm.get("ring_slots", 3)),
                        "base_epoch_ms": float(tb.get("base_epoch_ms")),
                        "base_pts_ns": int(tb.get("base_pts_ns", tb.get("base_pts", 0))),
                        "pts_units": str(tb.get("pts_units", "ns")).lower(),
                    }
            except OSError:
                time.sleep(0.05)
        return None
    except Exception:
        return None


def _try_fetch_health(url: str) -> Optional[dict]:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        shm = data.get("shm", {})
        tb = data.get("timebase", {})
        meta = {
            "map_name": shm.get("map_name"),
            "width": int(shm.get("width", 1920)),
            "height": int(shm.get("height", 1080)),
            "stride": int(shm.get("stride", shm.get("width", 1920) * 4)),
            "format": str(shm.get("format", "BGRx")),
            "ring_slots": int(shm.get("ring_slots", 3)),
            "base_epoch_ms": float(tb.get("base_epoch_ms")),
            "base_pts_ns": int(tb.get("base_pts_ns", tb.get("base_pts", 0))),
            "pts_units": str(tb.get("pts_units", "ns")).lower(),
        }
        return meta if meta.get("map_name") else None
    except Exception:
        return None


# This factory returns a best-effort ring-reader callback. Replace with a binding
# to your native mmap ring (gst_cap_native) when available.


def _make_ring_reader(map_name: str, width: int, height: int, stride: int, ring_slots: int):
    """Return a callable that yields (bgrx_bytes, pts_ns, frame_id, last_frame_age_ms) or None.
    Placeholder implementation that raises until wired to the native ring.
    """
    _warned = False

    def _reader() -> Optional[Tuple[memoryview, int, int, float]]:
        nonlocal _warned
        if not _warned:
            _warned = True
            print(
                "[reader.shm] WARN: No native ring binding wired. Use NullTransport for now or wire gst_cap_native."
            )
        return None

    return _reader
