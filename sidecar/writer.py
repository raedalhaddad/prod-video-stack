# ruff: noqa: UP007  # keep Optional[...] for Py3.9; don't force X | Y
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO


class SidecarWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: Optional[TextIO] = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="")

    def append_meta(self, meta: dict) -> None:
        if not self._fh:
            raise RuntimeError("SidecarWriter is not open")
        self._fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def append_frame(self, rec: dict) -> None:
        if not self._fh:
            raise RuntimeError("SidecarWriter is not open")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_raw(self, rec: dict) -> None:
        if not self._fh:
            raise RuntimeError("SidecarWriter is not open")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
