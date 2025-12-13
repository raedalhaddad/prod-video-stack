from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionResult:
    """
    Lightweight, per-frame output of the motion engine.

    This is intentionally compact and stable so downstream components
    (event builder, clip manager, sidecar writer) can depend on it
    without pulling in the legacy dev motion code.
    """

    # Core signal
    is_motion: bool
    score: float  # scalar motion "strength" in [0, 1] (approx.)
    pts_ms: float  # producer-aligned epoch ms for the frame
    frame_id: int  # monotonic frame counter from the reader

    # Telemetry (best-effort; safe defaults so callers can rely on presence)
    area_frac: float = 0.0  # fraction of frame marked as motion
    wind_like: bool = False  # True if motion looks like wind/foliage
    real_override: bool = False  # True when "definitely real" motion
    foliage_motion_frac: float = 0.0
    wind_idx: float = 0.0
    wind_state: str = "Unknown"
    zones_active_frac: float = 0.0


@dataclass
class MotionConfig:
    """
    Configuration knobs for the new motion engine.

    Values are chosen to be conservative and can be tuned later from a
    config file without breaking the API.
    """

    # Background subtractor / mask generation
    fg_history: int = 500
    fg_var_threshold: float = 24.0
    fg_detect_shadows: bool = True

    # Binary mask clean-up
    fg_threshold: int = 32
    morph_kernel: int = 3  # radius for a simple opening op; 0/1 disables

    # Motion score & gating
    min_motion_area_frac: float = 0.0005  # minimum area to consider motion
    score_scale: float = 1.0  # simple linear scaling for area->score
    score_cap: float = 1.0  # clamp score to this max

    # Optional stride (process every Nth frame)
    stride: int = 1
