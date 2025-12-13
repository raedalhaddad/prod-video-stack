"""New motion engine implementation for prod-video-stack.

This intentionally **does not** depend on the legacy dev `motion.py`
pipeline. Instead, it provides a compact, self-contained engine that:

- Consumes `common.frame.Frame` objects (BGR image, pts_ms, frame_id).
- Produces `MotionResult` instances from `analysis.motion.model`.
- Uses OpenCV background subtraction when available, but degrades
  gracefully when `cv2` is not installed.

The goal is to provide a stable API and sane behaviour so that higher
level components (motion events, sidecars, clip manager) can be built
on top without needing to know how the underlying mask is computed.
"""

from __future__ import annotations

from typing import Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - environment without OpenCV
    cv2 = None  # type: ignore

import numpy as np

from common.frame import Frame
from common.timebase import Timebase

from .model import MotionConfig, MotionResult


class MotionEngine:
    """Stateful, background-subtraction based motion engine.

    The implementation here is deliberately modest compared to the dev
    pipeline: it computes a foreground mask, cleans it up with a couple
    of morphological operations, and turns the resulting motion area
    into a scalar `score` in [0, 1].

    All higher-level semantics (events, sidecars, clip-gated recording)
    are downstream consumers of `MotionResult` and do not need to know
    about the specific mask details here.
    """

    def __init__(self, timebase: Timebase, config: Optional[MotionConfig] = None) -> None:
        self._timebase = timebase
        self._cfg = config or MotionConfig()

        # Background subtractor is best-effort: if OpenCV is not present
        # we gracefully fall back to "no motion" outputs.
        self._bg = self._make_bg_subtractor()
        self._frame_idx = 0
        self._last_result: Optional[MotionResult] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _make_bg_subtractor(self):
        if cv2 is None:
            return None
        try:
            return cv2.createBackgroundSubtractorMOG2(
                history=int(self._cfg.fg_history),
                varThreshold=float(self._cfg.fg_var_threshold),
                detectShadows=bool(self._cfg.fg_detect_shadows),
            )
        except Exception:
            # If anything goes wrong here, fall back to a simple MOG2
            # with defaults – motion detection should never crash.
            try:  # pragma: no cover - extremely defensive
                return cv2.createBackgroundSubtractorMOG2()
            except Exception:
                return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def step(self, frame: Frame, recording_active: bool = False) -> MotionResult:
        """Process a single frame and return a `MotionResult`.

        Parameters
        ----------
        frame:
            The current video frame from the reader, carrying `img`,
            `pts_ms`, and `frame_id`.
        recording_active:
            Reserved for future use (e.g. to tweak thresholds while
            a clip is actively being recorded). Ignored for now.
        """
        self._frame_idx += 1

        # --- NEW: derive epoch_ms from producer PTS when available ---
        pts_ms: float
        pts_ns = getattr(frame, "pts_ns", None)

        if pts_ns is not None:
            # Producer-aligned epoch_ms from PTS
            pts_ms = float(self._timebase.pts_to_epoch_ms(pts_ns))
        else:
            # Fallback for legacy/other readers that already provide pts_ms.
            # This keeps compatibility but loses strict alignment guarantees.
            raw_pts_ms = getattr(frame, "pts_ms", None)
            if raw_pts_ms is None:
                raise RuntimeError(
                    "Frame is missing both pts_ns and pts_ms; cannot timestamp motion"
                )
            pts_ms = float(raw_pts_ms)

        frame_id = int(getattr(frame, "frame_id", 0))

        # If we're configured to stride, we only recompute the mask on
        # every Nth frame and reuse the last result in between.
        if (
            self._cfg.stride > 1
            and (self._frame_idx - 1) % self._cfg.stride != 0
            and self._last_result is not None
        ):
            # Update timing fields to match the new frame but keep
            # the previous motion score.
            reused = MotionResult(
                is_motion=self._last_result.is_motion,
                score=self._last_result.score,
                pts_ms=pts_ms,
                frame_id=frame_id,
                area_frac=self._last_result.area_frac,
                wind_like=self._last_result.wind_like,
                real_override=self._last_result.real_override,
                foliage_motion_frac=self._last_result.foliage_motion_frac,
                wind_idx=self._last_result.wind_idx,
                wind_state=self._last_result.wind_state,
                zones_active_frac=self._last_result.zones_active_frac,
            )
            self._last_result = reused
            return reused

        # Best-effort motion estimation; any failure becomes "no motion"
        # for this frame, but we still emit a MotionResult.
        img = getattr(frame, "img", None)
        result = MotionResult(
            is_motion=False,
            score=0.0,
            pts_ms=pts_ms,
            frame_id=frame_id,
        )

        if (
            cv2 is None
            or self._bg is None
            or img is None
            or not hasattr(img, "ndim")
            or img.ndim != 3
        ):
            # No OpenCV or image – we can't compute a proper mask, but we
            # still return a valid result so callers can rely on the API.
            self._last_result = result
            return result

        # Ensure we have a numpy array with a sane dtype
        try:
            arr = np.asarray(img)
        except Exception:
            self._last_result = result
            return result

        if arr.ndim != 3:
            self._last_result = result
            return result

        try:
            fg = self._bg.apply(arr)  # type: ignore[union-attr]
        except Exception:
            self._last_result = result
            return result

        if fg is None:
            self._last_result = result
            return result

        try:
            # Threshold the foreground mask to a clean binary mask
            _, mask = cv2.threshold(  # type: ignore[arg-type]
                fg,
                int(self._cfg.fg_threshold),
                255,
                cv2.THRESH_BINARY,
            )

            if self._cfg.morph_kernel and self._cfg.morph_kernel > 1:
                ksize = int(self._cfg.morph_kernel)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (ksize, ksize),
                )
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            motion_px = int((mask > 0).sum())
            total_px = int(mask.size) if mask is not None else 0

            area_frac = float(motion_px) / float(total_px) if total_px > 0 else 0.0

            # Simple scalar score based on area fraction
            raw_score = area_frac * float(self._cfg.score_scale)
            score = max(0.0, min(float(self._cfg.score_cap), raw_score))

            is_motion = area_frac >= float(self._cfg.min_motion_area_frac) and score > 0.0

            result = MotionResult(
                is_motion=is_motion,
                score=score,
                pts_ms=pts_ms,
                frame_id=frame_id,
                area_frac=area_frac,
                # For now we treat all motion as "not explicitly wind-like";
                # downstream components can add more nuance later without
                # changing this core API.
                wind_like=False,
                real_override=is_motion,
            )
        except Exception:
            # Anything unexpected falls back to "no motion" for safety.
            result = MotionResult(
                is_motion=False,
                score=0.0,
                pts_ms=pts_ms,
                frame_id=frame_id,
            )

        self._last_result = result
        return result
