from __future__ import annotations

import numpy as np

from analysis.motion import MotionConfig, MotionEngine, MotionResult
from common.frame import Frame


def test_motion_smoke_run() -> None:
    # Construct an engine with default configuration and run a single
    # frame through it to confirm we get a MotionResult back.
    eng = MotionEngine(MotionConfig())
    f = Frame(
        img=np.zeros((64, 64, 3), dtype=np.uint8),
        pts_ms=1_700_000_000_000.0,
        frame_id=0,
    )
    out = eng.step(f)

    assert isinstance(out, MotionResult)
    assert out.pts_ms == f.pts_ms
    assert out.frame_id == f.frame_id
    # Score should be in the expected range.
    assert 0.0 <= out.score <= 1.0
