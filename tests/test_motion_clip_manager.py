from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from analysis.motion.clip_manager import MotionClipConfig, MotionClipManager
from analysis.motion.events import MotionEvent
from common.timebase import Timebase
from record.recorder import ClipRequest


@dataclass
class _FakeRecorder:
    timebase: Optional[Timebase]
    requests: List[ClipRequest]

    def __init__(self, timebase: Optional[Timebase]) -> None:
        self.timebase = timebase
        self.requests = []

    def request_clip(self, req: ClipRequest):
        self.requests.append(req)
        return object()


def _mk_event(start_ms: float, stop_ms: float) -> MotionEvent:
    return MotionEvent(
        start_pts_ns=int(start_ms),
        stop_pts_ns=int(stop_ms),
        start_ms=start_ms,
        stop_ms=stop_ms,
        max_score=0.8,
    )


def test_clip_manager_single_event_pts_mode():
    # pts_units="epoch_ms" ⇒ pts == epoch_ms for testing simplicity
    tb = Timebase(base_epoch_ms=0.0, base_pts_ns=0.0, pts_units="epoch_ms")
    rec = _FakeRecorder(timebase=tb)

    cfg = MotionClipConfig(
        preroll_ms=1000,
        postroll_ms=2000,
        min_clip_ms=0,
        max_clip_ms=60_000,
        merge_if_gap_ms=0,
        cooldown_ms=0,
        label="motion",
        out_dir=Path("/clips"),
        max_wait_ms=None,
    )

    mgr = MotionClipManager(recorder=rec, config=cfg)

    ev = _mk_event(10_000.0, 12_000.0)
    mgr.handle_event(ev)
    mgr.flush()

    assert len(rec.requests) == 1
    req = rec.requests[0]

    # With epoch_ms timebase, epoch_ms_to_pts is identity, then cast to int.
    assert req.start_pts_ns == 9000
    assert req.stop_pts_ns == 14_000
    assert req.start_ms is None
    assert req.stop_ms is None
    assert req.label == "motion"
    assert req.out_dir == Path("/clips")


def test_clip_manager_merges_close_events():
    """
    With the new eager behavior, MotionClipManager issues one clip per
    MotionEvent. Any merging of close motion bursts is handled upstream
    by MotionEventBuilder, not here.

    So two separate MotionEvents → two separate clip requests.
    """
    tb = Timebase(base_epoch_ms=0.0, base_pts_ns=0.0, pts_units="epoch_ms")
    rec = _FakeRecorder(timebase=tb)
    cfg = MotionClipConfig(
        preroll_ms=0,
        postroll_ms=0,
        min_clip_ms=0,
        max_clip_ms=60_000,
        merge_if_gap_ms=500,
        cooldown_ms=0,
        label="motion",
        out_dir=None,
        max_wait_ms=None,
    )
    mgr = MotionClipManager(recorder=rec, config=cfg)

    ev1 = _mk_event(1000.0, 2000.0)
    ev2 = _mk_event(2300.0, 2600.0)  # gap 300ms < merge_if_gap_ms

    mgr.handle_event(ev1)
    mgr.handle_event(ev2)
    mgr.flush()

    # New behavior: two distinct clip requests, matching each event.
    assert len(rec.requests) == 2

    req1 = rec.requests[0]
    req2 = rec.requests[1]

    assert req1.start_pts_ns == 1000
    assert req1.stop_pts_ns == 2000

    assert req2.start_pts_ns == 2300
    assert req2.stop_pts_ns == 2600


def test_clip_manager_epoch_fallback_mode():
    rec = _FakeRecorder(timebase=None)
    cfg = MotionClipConfig(
        preroll_ms=0,
        postroll_ms=0,
        min_clip_ms=0,
        max_clip_ms=60_000,
        merge_if_gap_ms=0,
        cooldown_ms=0,
        label="motion",
        out_dir=None,
        max_wait_ms=None,
    )
    mgr = MotionClipManager(recorder=rec, config=cfg)

    ev = _mk_event(5000.0, 7000.0)
    mgr.handle_event(ev)
    mgr.flush()

    assert len(rec.requests) == 1
    req = rec.requests[0]
    assert req.start_pts_ns is None
    assert req.stop_pts_ns is None
    assert req.start_ms == 5000.0
    assert req.stop_ms == 7000.0
