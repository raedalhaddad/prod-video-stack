from __future__ import annotations

from pathlib import Path

from sidecar.reader import SidecarReader
from sidecar.writer import SidecarWriter


def test_sidecar_roundtrip(tmp_path: Path):
    path = tmp_path / "sample_motion.jsonl"
    w = SidecarWriter(path)
    w.open()
    w.append_meta({"type": "meta", "schema": "rtva.motion.v2.meta", "cam": "cam1"})
    w.append_frame(
        {"ts_ms": 1000.0, "frame": 1, "boxes": [{"xyxy": [0.1, 0.1, 0.2, 0.2], "area": 0.01}]}
    )
    w.append_frame({"ts_ms": 1033.0, "frame": 2, "boxes": []})
    w.close()
    rows = list(SidecarReader(path))
    assert rows[0]["type"] == "meta"
    assert rows[1]["frame"] == 1 and isinstance(rows[1]["boxes"], list)
    assert rows[2]["frame"] == 2 and rows[2]["boxes"] == []


def test_sidecar_writer_context_manager(tmp_path):
    p = tmp_path / "cm.jsonl"
    with SidecarWriter(p) as w:
        w.append_meta({"schema": "rtva.motion.v2.meta", "cam": "cam1"})
        w.append_frame({"ts_ms": 123.0, "frame": 0, "boxes": []})
    # file should exist & be readable immediately
    lines = list(SidecarReader(p))
    assert lines[0]["type"] == "meta"
    assert lines[1]["frame"] == 0
