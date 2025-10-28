from __future__ import annotations

# ruff: noqa: UP007  # keep Optional/Union (Py3.9-safe); don't force X | Y
from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        agnostic: bool = False,
        half: bool = True,
        warmup: bool = False,
        allowed_classes: Optional[Iterable[Union[str, int]]] = None,
    ):
        self._impl = None
        try:
            from ultralytics import YOLO  # noqa

            try:
                from ._yolo_impl import YoloDetector

                detector_cls = YoloDetector
            except Exception:
                import torch  # type: ignore
                from ultralytics import YOLO as _YOLO  # type: ignore

                class _InlineYolo:
                    def __init__(
                        self,
                        model_path,
                        device,
                        imgsz,
                        conf,
                        iou,
                        agnostic,
                        half,
                        warmup,
                        allowed_classes,
                    ):
                        self.device = device or (
                            "cuda:0"
                            if getattr(torch, "cuda", None) and torch.cuda.is_available()
                            else "cpu"
                        )
                        self.model = _YOLO(model_path)
                        self.model.to(self.device)
                        self.imgsz = imgsz
                        self.conf = conf
                        self.iou = float(iou)
                        self.agnostic = bool(agnostic)
                        if warmup:
                            try:
                                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                                _ = self.model.predict(
                                    dummy,
                                    imgsz=self.imgsz,
                                    conf=self.conf,
                                    iou=self.iou,
                                    device=self.device,
                                    verbose=False,
                                )
                            except Exception:
                                pass

                    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> list[dict]:
                        size = int(imgsz) if imgsz else self.imgsz
                        res = self.model.predict(
                            frame,
                            imgsz=size,
                            conf=self.conf,
                            iou=self.iou,
                            agnostic_nms=self.agnostic,
                            device=self.device,
                            verbose=False,
                        )[0]
                        out: list[dict] = []
                        boxes = getattr(res, "boxes", None)
                        if boxes is None or len(boxes) == 0:
                            return out
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        confs = boxes.conf.detach().cpu().numpy()
                        clss = boxes.cls.detach().cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
                            out.append(
                                {
                                    "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                                    "conf": float(cf),
                                    "cls_id": int(ci),
                                    "cls_name": str(int(ci)),
                                }
                            )
                        return out

                detector_cls = _InlineYolo
            self._impl = detector_cls(
                model_path=model_path,
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                agnostic=agnostic,
                half=half,
                warmup=warmup,
                allowed_classes=allowed_classes,
            )
        except Exception:
            self._impl = None

    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> list[dict[str, Any]]:
        if self._impl is None:
            raise RuntimeError("YOLO/torch not available; install deps or mock Detector in tests.")
        return self._impl.detect(frame, imgsz=imgsz)
