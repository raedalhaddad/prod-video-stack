from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .reader import FrameStream


class VideoSource:
    """Drop-in shim to replace cv2.VideoCapture usage in analysis code."""

    def __init__(self, stream: FrameStream):
        self._stream = stream

    def start(self) -> None:
        self._stream.start()

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        return self._stream.read()

    def close(self) -> None:
        self._stream.close()

    def stats(self):
        # Optional convenience passthrough
        stats = getattr(self._stream, "stats", None)
        return stats() if callable(stats) else None
