from __future__ import annotations

import numpy as np

from analysis.motion.engine import MotionEngine
from common.frame import Frame


def test_motion_smoke_run():
    eng = MotionEngine()
    f = Frame(img=np.zeros((64, 64, 3), dtype=np.uint8), pts_ms=1_700_000_000_000.0, frame_id=0)
    out = eng.step(f)
    assert isinstance(out, dict)
    assert "boxes" in out and "ts_ms" in out and "frame" in out
