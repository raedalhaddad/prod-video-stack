[![CI](https://github.com/raedalhaddad/prod-video-stack/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/raedalhaddad/prod-video-stack/actions/workflows/ci.yml?query=branch%3Amain)

# prod-video-stack

Production-grade video analysis stack **skeleton**. This greenfield repo hosts the modernized capture/record/sidecar toolchain while the legacy app continues to run for stability.

## Architecture (high-level)

```mermaid
flowchart LR
  subgraph COMMON[common/]
    F[frame.py → Frame(img, pts_ms, frame_id)]
    T[time.py]
  end

  CAP[capture/] -->|Frame (BGR, pts_ms, id)| MOTION[analysis/motion/engine.py]
  CAP -->|Frame| DETECT[analysis/detect/detector.py]
  CAP -->|segments| REC[record/]

  subgraph MOTION_STACK[analysis/motion/]
    MOTION -->|env-gated| LEG[_legacy_engine.py]:::legacy
    LEG --- U[utils/motion_utils.py]:::legacy
    LEG --- CFG[config.py (shim → RTVA_CONFIG_MODULE)]:::shim
  end

  DETECT -->|boxes/classes/scores| SIDEW[sidecar/writer.py]
  MOTION -->|motion boxes/metrics| SIDEW
  SIDER[sidecar/reader.py] --> REC
  REC -->|finalize/concat/trim| OUT[(artifacts)]

  TOOLS[tools/] -->|ops checks| CAP & REC & SIDEW
  NATIVE[native/] -->|future: C++/Rust svc| CAP
  INFRA[infra/] -->|svc files, packaging| CAP & REC & SIDEW

  classDef legacy fill:#fff3e6,stroke:#c77,stroke-width:1px;
  classDef shim fill:#eef5ff,stroke:#77c,stroke-width:1px;
```

## Getting started (Windows PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev]

pre-commit install
pre-commit run --all-files
pre-commit run --all-files

pytest

python -m tools.check_sync --help
python -m tools.check_sync sample -o tools\samples\sync_report.md
```

> **Note:** Do **not** run `pre-commit autoupdate` while on Python 3.9 for the isort repo. Hooks are pinned to Py3.9-safe versions.
