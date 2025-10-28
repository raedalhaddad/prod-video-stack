# analysis/motion/config.py
import os
from importlib import import_module

for name in filter(
    None,
    [
        os.environ.get("RTVA_CONFIG_MODULE"),
        "config",
        "common.config",
        "real_time_video_analysis.config",
    ],
):
    try:
        _cfg = import_module(name)
        break
    except Exception:
        _cfg = None

if _cfg is None:
    raise ImportError(
        "Could not locate runtime config module. "
        "Set RTVA_CONFIG_MODULE to your config module (e.g., 'config')."
    )

globals().update({k: getattr(_cfg, k) for k in dir(_cfg) if not k.startswith("_")})
__all__ = [k for k in globals() if not k.startswith("_")]
