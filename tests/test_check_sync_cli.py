import subprocess
import sys
from pathlib import Path

def test_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.check_sync", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()

def test_sample_outputs_markdown(tmp_path: Path):
    out = tmp_path / "report.md"
    proc = subprocess.run([sys.executable, "-m", "tools.check_sync", "sample", "-o", str(out)])
    assert proc.returncode == 0
    assert out.exists()
    assert "Sync Report" in out.read_text(encoding="utf-8")
