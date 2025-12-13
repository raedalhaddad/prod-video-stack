from __future__ import annotations

import math

from analysis.motion import MotionEventBuilder, MotionEventConfig, MotionResult
from common.frame import Frame
from common.timebase import Timebase


def _mk_result(t_ms: float, score: float) -> MotionResult:
    # For testing we don't care about image content; use a dummy frame and
    # just propagate pts_ms / frame_id into MotionResult.
    f = Frame(img=None, pts_ms=t_ms, frame_id=int(t_ms))  # type: ignore[arg-type]
    return MotionResult(
        is_motion=score > 0.0,
        score=score,
        pts_ms=f.pts_ms,
        frame_id=f.frame_id,
        area_frac=score,
    )


def test_single_event_basic():
    cfg = MotionEventConfig(
        score_on=0.5,
        score_off=0.3,
        min_event_ms=0.0,
        min_gap_ms=0.0,
        merge_gap_ms=0.0,
    )
    tb = Timebase(base_epoch_ms=0.0, base_pts_ns=0.0, pts_units="ns")

    builder = MotionEventBuilder(timebase=tb, config=cfg)

    times = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    scores = [0.0, 0.1, 0.6, 0.7, 0.4, 0.2]

    events = []
    for t_ms, sc in zip(times, scores):
        events.extend(builder.consume(_mk_result(t_ms, sc)))
    events.extend(builder.flush())

    assert len(events) == 1
    ev = events[0]

    # Event should start when score first crosses score_on and end at the last
    # time it was at or above score_on. In this sequence the score is:
    # 0.0, 0.1, 0.6, 0.7, 0.4, 0.2
    # so the event spans 200ms → 300ms.
    assert math.isclose(ev.start_ms, 200.0)
    assert math.isclose(ev.stop_ms, 300.0)
    assert ev.start_pts_ns < ev.stop_pts_ns
    assert math.isclose(ev.max_score, 0.7, rel_tol=1e-6)


def test_merging_of_close_events():
    cfg = MotionEventConfig(
        score_on=0.5,
        score_off=0.3,
        min_event_ms=0.0,
        min_gap_ms=0.0,
        merge_gap_ms=500.0,  # ms
    )
    tb = Timebase(base_epoch_ms=0.0, base_pts_ns=0.0, pts_units="ns")
    builder = MotionEventBuilder(timebase=tb, config=cfg)

    # Two motion bursts separated by 250 ms (< merge_gap_ms), so they should
    # collapse into a single MotionEvent.
    seq = [
        (0.0, 0.0),
        (100.0, 0.6),
        (200.0, 0.6),
        (300.0, 0.2),
        (450.0, 0.6),
        (550.0, 0.6),
        (650.0, 0.2),
        (1000.0, 0.0),
    ]

    events = []
    for t_ms, sc in seq:
        events.extend(builder.consume(_mk_result(t_ms, sc)))
    events.extend(builder.flush())

    assert len(events) == 1
    ev = events[0]

    # The merged event should span from the first burst start to the last
    # burst stop.
    assert math.isclose(ev.start_ms, 100.0)
    assert math.isclose(ev.stop_ms, 550.0)
    assert ev.max_score > 0.5
