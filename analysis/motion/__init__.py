"""Public exports for the motion analysis package."""

from __future__ import annotations

from .clip_manager import MotionClipConfig, MotionClipManager
from .engine import MotionEngine
from .events import MotionEvent, MotionEventBuilder, MotionEventConfig
from .model import MotionConfig, MotionResult
from .sidecar import MotionSidecarWriter

__all__ = [
    "MotionEngine",
    "MotionResult",
    "MotionConfig",
    "MotionEvent",
    "MotionEventBuilder",
    "MotionEventConfig",
    "MotionClipConfig",
    "MotionClipManager",
    "MotionSidecarWriter",
]
