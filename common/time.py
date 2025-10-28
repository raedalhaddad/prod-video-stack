from __future__ import annotations

from datetime import datetime, timezone


def now_ms() -> float:
    return datetime.now(tz=timezone.utc).timestamp() * 1000.0


def to_iso_utc(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
