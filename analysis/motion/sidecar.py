from __future__ import annotations

from pathlib import Path
from typing import Any

from sidecar.writer import SidecarWriter

from .events import MotionEvent


class MotionSidecarWriter:
    """
    Thin wrapper around SidecarWriter for motion events.

    Writes one JSON object per line, with the schema sketched in the
    Phase 3 brief (type="motion_event", start/stop PTS + epoch-ms, etc.).
    """

    def __init__(self, path: str | Path):
        self._writer = SidecarWriter(path)

    def __enter__(self) -> MotionSidecarWriter:
        self._writer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._writer.__exit__(exc_type, exc, tb)

    def write_event(self, ev: MotionEvent) -> None:
        payload: dict[str, Any] = {
            "type": "motion_event",
            "start_pts_ns": int(ev.start_pts_ns),
            "stop_pts_ns": int(ev.stop_pts_ns),
            "start_ms": float(ev.start_ms),
            "stop_ms": float(ev.stop_ms),
            "max_score": float(ev.max_score),
            "wind_like": bool(ev.wind_like),
            "real_override": bool(ev.real_override),
            "foliage_motion_frac": float(ev.foliage_motion_frac),
            "wind_idx": float(ev.wind_idx),
            "wind_state": ev.wind_state,
            "zones_active_frac": float(ev.zones_active_frac),
        }
        # Use the generic "raw" append API provided by SidecarWriter.
        self._writer.append_raw(payload)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
