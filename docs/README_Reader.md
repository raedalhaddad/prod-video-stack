# Reader & Discovery (Task G / Phase 1)

This module provides a non-blocking Reader that consumes frames from the native
producer's shared-memory ring (BGR/BGRx) and exposes `(frame_bgr, pts_ms,
frame_id)` tuples to downstream analysis.

Phase 1 ("Reader Hardening") keeps the core design from Task G, but makes the
contracts explicit and adds tests around the `/health.timebase` mapping.

## Key properties

- **Non-blocking** `read()`; returns `None` when no new frame is available.
- **Bounded queue** with `drop_new` or `drop_old` policy — never backpressure
  the producer.
- **Single-copy** BGRx→BGR conversion for SHM frames.
- **Reconnect-safe** and discovery-first design:
  - control-pipe (when available) → `/health` → naming fallback (dev-only).

## Canonical usage (Python)

The canonical way to build a reader for analysis code is:

```python
from capture.reader import ReaderConfig, ReaderFactory
from capture.nonblocking_adapter import wrap_nonblocking
from capture.stats_adapter import make_pts_mapper_from_health, wrap_with_stats

cfg = ReaderConfig(
    prefer="shm",
    # Use "" / "none" / "off" to disable control-pipe discovery when not available
    ctrl_pipe="",
    health_url="http://127.0.0.1:8765/health",
)

stream = ReaderFactory.from_config(cfg)
stream = wrap_nonblocking(stream, queue_max=cfg.queue_max, drop_policy=cfg.drop_policy)

pts_map = make_pts_mapper_from_health(cfg.health_url)
reader = wrap_with_stats(stream, pts_mapper=pts_map)

frame_bgr, pts_ms, frame_id = reader.read()
```

Where:

- `frame_bgr` is a `H×W×3` `np.ndarray` (BGR).
- `pts_ms` is a float timestamp in **epoch milliseconds**, derived from the
  producer's PTS and `/health.timebase`.
- `frame_id` is the monotonically increasing frame id from the producer.

## Producer contracts

The reader relies on the producer's `/health` endpoint to discover both the SHM
geometry and the timebase:

- `shm.map_name`, `shm.width`, `shm.height`, `shm.stride`, `shm.format`,
  `shm.ring_slots`
- `timebase.base_epoch_ms`, `timebase.base_pts_ns`, `timebase.pts_units`

The schema for `/health` is `rtva.producer.health.v1`. Only a small subset is
required by the reader; the rest is for telemetry and monitoring.

If a control pipe is available, it can provide the same metadata via a local
named pipe handshake. In environments where the pipe is not present, the reader
should be configured to skip it and rely solely on `/health` (see `ctrl_pipe`
below).

## Timebase mapping

`capture.stats_adapter.make_pts_mapper_from_health()` builds a pts→epoch-ms
mapping from `/health.timebase`. Internally, it applies:

```text
pts_ms = base_epoch_ms + (raw_pts - base_pts) * scale
```

where `scale` depends on `pts_units`:

- `"ns"`      → `scale = 1e-6`
- `"us"`      → `scale = 1e-3`
- `"ms"`      → `scale = 1.0`
- `"epoch_ms"`→ raw pts already represent epoch-ms (identity mapping)
- `"epoch_s"` → raw pts represent epoch seconds (×1000 for ms)

The `WithStats` wrapper also learns a small skew correction when frames appear
slightly "in the future" relative to the consumer clock; this is exposed as
`ReaderStats.skew_ms`. Negative ages are clamped to zero in `last_frame_age_ms`.

## Config (TOML/env)

```toml
[reader]
prefer = "shm"   # or "null"

[reader.shm]
ctrl_pipe = ""  # "" / "none" / "off" to disable control pipe and rely on /health
map_name = "Local\\rtva_cam0"  # optional when discovery works
health_url = "http://127.0.0.1:8765/health"
queue_max = 3
drop_policy = "drop_new"
connect_timeout_ms = 1500
