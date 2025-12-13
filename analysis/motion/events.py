from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from common.timebase import Timebase

from .model import MotionResult


@dataclass
class MotionEvent:
    """
    Coarse motion event over a time window.

    This is the main unit that downstream components (sidecar writer,
    clip manager) will consume.
    """

    # Primary timebase: producer PTS (units described by Timebase.pts_units).
    start_pts_ns: int
    stop_pts_ns: int

    # Derived epoch-ms timestamps aligned to the same Timebase.
    start_ms: float
    stop_ms: float

    # Peak motion score within the event window.
    max_score: float

    # Telemetry (best-effort)
    wind_like: bool = False
    real_override: bool = False
    foliage_motion_frac: float = 0.0
    wind_idx: float = 0.0
    wind_state: str = "Unknown"
    zones_active_frac: float = 0.0


@dataclass
class MotionEventConfig:
    """
    Configuration for the MotionEventBuilder.

    The defaults are conservative but can be tuned from config files.
    """

    # Thresholds on MotionResult.score
    score_on: float = 0.5
    score_off: float = 0.3

    # Minimum event duration (ms) before we materialise an event.
    min_event_ms: float = 500.0

    # How long score must stay below score_off (ms) before we end an event.
    min_gap_ms: float = 400.0

    # If two events are closer than this gap (ms), we merge them.
    merge_gap_ms: float = 2000.0


class MotionEventBuilder:
    """
    Turn a stream of MotionResult into MotionEvent windows.

    API:
        builder = MotionEventBuilder(timebase, MotionEventConfig())
        events = builder.consume(motion_result)   # list[MotionEvent]
        final_events = builder.flush()            # at end of stream
    """

    def __init__(
        self,
        timebase: Optional[Timebase] = None,
        config: Optional[MotionEventConfig] = None,
    ) -> None:
        self._tb = timebase
        self._cfg = config or MotionEventConfig()

        # Active event (not yet closed by score falling below off-threshold).
        self._active_start_ms: Optional[float] = None
        self._active_start_pts: Optional[float] = None
        self._active_max_score: float = 0.0
        self._active_wind_like: bool = False
        self._active_real_override: bool = False
        self._active_foliage_motion_frac: float = 0.0
        self._active_wind_idx: float = 0.0
        self._active_wind_state: str = "Unknown"
        self._active_zones_active_frac: float = 0.0
        self._last_above_ms: Optional[float] = None

        # Last completed event that is still eligible for merging.
        self._pending_event: Optional[MotionEvent] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _epoch_ms_to_pts(self, epoch_ms: float) -> int:
        """
        Convert epoch-ms to producer PTS using the Timebase, if available.

        When no Timebase is provided, we fall back to treating epoch-ms as the
        PTS timeline directly. This isn't ideal for prod but keeps the code
        usable in offline tests and tools.
        """
        if self._tb is None:
            return int(epoch_ms)
        return int(self._tb.epoch_ms_to_pts(epoch_ms))

    def _merge_or_buffer(self, ev: MotionEvent, out: List[MotionEvent]) -> None:
        """
        Merge `ev` into the pending event if the gap is small; otherwise emit
        the pending event and buffer `ev`.
        """
        if self._pending_event is None:
            self._pending_event = ev
            return

        gap_ms = ev.start_ms - self._pending_event.stop_ms
        if gap_ms <= self._cfg.merge_gap_ms:
            # Merge into a single window.
            merged = MotionEvent(
                start_pts_ns=self._pending_event.start_pts_ns,
                stop_pts_ns=ev.stop_pts_ns,
                start_ms=self._pending_event.start_ms,
                stop_ms=ev.stop_ms,
                max_score=max(self._pending_event.max_score, ev.max_score),
                wind_like=self._pending_event.wind_like or ev.wind_like,
                real_override=(self._pending_event.real_override or ev.real_override),
                foliage_motion_frac=max(
                    self._pending_event.foliage_motion_frac,
                    ev.foliage_motion_frac,
                ),
                wind_idx=max(self._pending_event.wind_idx, ev.wind_idx),
                wind_state=ev.wind_state or self._pending_event.wind_state,
                zones_active_frac=max(
                    self._pending_event.zones_active_frac,
                    ev.zones_active_frac,
                ),
            )
            self._pending_event = merged
        else:
            out.append(self._pending_event)
            self._pending_event = ev

    def _finalise_pending_if_far(self, now_ms: float, out: List[MotionEvent]) -> None:
        """
        Emit the pending event if it's safely in the past (beyond merge_gap_ms).
        """
        if self._pending_event is None:
            return

        gap_ms = now_ms - self._pending_event.stop_ms
        if gap_ms >= self._cfg.merge_gap_ms:
            out.append(self._pending_event)
            self._pending_event = None

    def _reset_active(self) -> None:
        self._active_start_ms = None
        self._active_start_pts = None
        self._active_max_score = 0.0
        self._active_wind_like = False
        self._active_real_override = False
        self._active_foliage_motion_frac = 0.0
        self._active_wind_idx = 0.0
        self._active_wind_state = "Unknown"
        self._active_zones_active_frac = 0.0
        self._last_above_ms = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def consume(self, res: MotionResult) -> List[MotionEvent]:
        """
        Consume a single MotionResult and return any completed MotionEvents.

        Most calls will return [], and occasionally [event].
        """
        out: List[MotionEvent] = []

        t_ms = float(res.pts_ms)
        score = float(res.score)

        # If we have an active event, update its state first.
        if self._active_start_ms is not None:
            # Track stats within the active window.
            self._active_max_score = max(self._active_max_score, score)
            self._active_wind_like = self._active_wind_like or res.wind_like
            self._active_real_override = self._active_real_override or res.real_override
            self._active_foliage_motion_frac = max(
                self._active_foliage_motion_frac,
                res.foliage_motion_frac,
            )
            self._active_wind_idx = max(self._active_wind_idx, res.wind_idx)
            self._active_wind_state = res.wind_state or self._active_wind_state
            self._active_zones_active_frac = max(
                self._active_zones_active_frac,
                res.zones_active_frac,
            )

            # Track the last time we were clearly above the "on" threshold.
            if score >= self._cfg.score_on:
                self._last_above_ms = t_ms

            # Check for event end: score has stayed below score_off long enough.
            if score <= self._cfg.score_off and self._last_above_ms is not None:
                gap_ms = t_ms - self._last_above_ms
                if gap_ms >= self._cfg.min_gap_ms:
                    stop_ms = self._last_above_ms
                    duration_ms = stop_ms - self._active_start_ms
                    if duration_ms >= self._cfg.min_event_ms:
                        start_ms = self._active_start_ms
                        start_pts = (
                            self._active_start_pts
                            if self._active_start_pts is not None
                            else self._epoch_ms_to_pts(start_ms)
                        )
                        stop_pts = self._epoch_ms_to_pts(stop_ms)
                        ev = MotionEvent(
                            start_pts_ns=int(start_pts),
                            stop_pts_ns=int(stop_pts),
                            start_ms=start_ms,
                            stop_ms=stop_ms,
                            max_score=self._active_max_score,
                            wind_like=self._active_wind_like,
                            real_override=self._active_real_override,
                            foliage_motion_frac=self._active_foliage_motion_frac,
                            wind_idx=self._active_wind_idx,
                            wind_state=self._active_wind_state,
                            zones_active_frac=self._active_zones_active_frac,
                        )
                        self._merge_or_buffer(ev, out)

                    # Whether or not we emitted an event, the active window closes.
                    self._reset_active()

        # If we don't currently have an active event, check for a new one.
        if self._active_start_ms is None and score >= self._cfg.score_on:
            self._active_start_ms = t_ms
            self._active_start_pts = self._epoch_ms_to_pts(t_ms)
            self._active_max_score = score
            self._active_wind_like = res.wind_like
            self._active_real_override = res.real_override
            self._active_foliage_motion_frac = res.foliage_motion_frac
            self._active_wind_idx = res.wind_idx
            self._active_wind_state = res.wind_state
            self._active_zones_active_frac = res.zones_active_frac
            self._last_above_ms = t_ms

        # Decide whether the pending event is far enough in the past that we can
        # safely emit it (it can no longer merge with future events).
        self._finalise_pending_if_far(t_ms, out)
        return out

    def flush(self) -> List[MotionEvent]:
        """
        Finalise any in-flight event and return all remaining events.

        Call this once at end-of-stream to avoid dropping a trailing event.
        """
        out: List[MotionEvent] = []

        # If an event is active, close it at the last-above timestamp (or
        # immediately if we never crossed score_on after starting).
        if self._active_start_ms is not None:
            stop_ms = self._last_above_ms or self._active_start_ms
            duration_ms = stop_ms - self._active_start_ms
            if duration_ms >= self._cfg.min_event_ms:
                start_ms = self._active_start_ms
                start_pts = (
                    self._active_start_pts
                    if self._active_start_pts is not None
                    else self._epoch_ms_to_pts(start_ms)
                )
                stop_pts = self._epoch_ms_to_pts(stop_ms)
                ev = MotionEvent(
                    start_pts_ns=int(start_pts),
                    stop_pts_ns=int(stop_pts),
                    start_ms=start_ms,
                    stop_ms=stop_ms,
                    max_score=self._active_max_score,
                    wind_like=self._active_wind_like,
                    real_override=self._active_real_override,
                    foliage_motion_frac=self._active_foliage_motion_frac,
                    wind_idx=self._active_wind_idx,
                    wind_state=self._active_wind_state,
                    zones_active_frac=self._active_zones_active_frac,
                )
                self._merge_or_buffer(ev, out)

        self._reset_active()

        # Emit any remaining pending event.
        if self._pending_event is not None:
            out.append(self._pending_event)
            self._pending_event = None

        return out
