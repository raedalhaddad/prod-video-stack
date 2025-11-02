# capture/__init__.py
"""Capture package: SHM reader, nonblocking adapter, and stats/PTS mapping."""

from .nonblocking_adapter import wrap_nonblocking
from .reader import ReaderConfig, ReaderFactory
from .stats_adapter import make_pts_mapper_from_health, wrap_with_stats

__all__ = [
    "ReaderFactory",
    "ReaderConfig",
    "wrap_nonblocking",
    "wrap_with_stats",
    "make_pts_mapper_from_health",
]

__version__ = "0.1.0"
