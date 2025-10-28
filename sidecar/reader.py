from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class SidecarReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
