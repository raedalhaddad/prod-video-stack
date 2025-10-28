from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    img: np.ndarray  # BGR (H,W,3), uint8
    pts_ms: float  # epoch ms (float)
    frame_id: int
