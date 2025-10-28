from __future__ import annotations

from typing import Any

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore
import os

from common.frame import Frame


class MotionEngine:
    def __init__(self) -> None:
        self._impl = None
        self._use_legacy = False
        # Only use the legacy engine when explicitly enabled.
        if os.getenv("RTVA_USE_LEGACY", "0") == "1":
            try:
                from ._legacy_engine import MotionEngine as _Legacy  # noqa: F401

                self._impl = _Legacy()
                self._use_legacy = True
            except Exception:
                self._use_legacy = False
        # Initialize smoke path if not using legacy
        if not self._use_legacy:
            self._bg = None
            if cv2 is not None:
                try:
                    self._bg = cv2.createBackgroundSubtractorMOG2(
                        history=200, varThreshold=16, detectShadows=True
                    )
                except Exception:
                    self._bg = None

    def step(self, frame: Frame, recording_active: bool = False) -> dict[str, Any]:
        if getattr(self, "_use_legacy", False) and self._impl is not None:
            ts_s = float(frame.pts_ms) / 1000.0
            res = self._impl.step(
                ts=ts_s,
                frame_bgr=frame.img,
                recording_active=bool(recording_active),
                dt=0.0,
                g=globals(),
            )
            return {
                "ts_ms": float(frame.pts_ms),
                "frame": int(frame.frame_id),
                "boxes": list(getattr(res, "motion_list", []) or []),
                "area_frac": float(getattr(res, "area_frac_weighted", 0.0)),
                "wind_like": bool(getattr(res, "wind_like", False)),
            }
        boxes = []
        if cv2 is not None and getattr(frame, "img", None) is not None and frame.img.ndim == 3:
            try:
                fg = self._bg.apply(frame.img) if self._bg is not None else None  # type: ignore
                if fg is not None:
                    _, th = cv2.threshold(fg, 32, 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts or []:
                        x, y, w, h = cv2.boundingRect(c)
                        if w * h < 64:
                            continue
                        boxes.append({"xyxy": [x, y, x + w, y + h], "area": int(w * h)})
            except Exception:
                boxes = []
        return {"ts_ms": float(frame.pts_ms), "frame": int(frame.frame_id), "boxes": boxes}
