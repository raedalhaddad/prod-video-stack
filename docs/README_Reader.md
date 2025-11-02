# Reader & Discovery (Task G)

This module provides a non-blocking Reader that consumes frames from the native producer via MMAP shared memory (BGRx) and exposes `(frame_bgr, pts_ms, frame_id)` tuples.

## Key properties
- **Non-blocking** `read()`; use `None` when no new frame.
- **Bounded queue** with `drop_new` or `drop_old` policy — never backpressure the producer.
- **Single-copy** BGRx→BGR conversion.
- **Reconnect-safe** and discovery-first design: control-pipe → `/health` → naming fallback.

## Config (TOML/env)
```toml
[reader]
prefer = "shm"   # or "null"

[reader.shm]
ctrl_pipe = "\\\\.\\pipe\\rtva_cam0_ctrl"
map_name = "Local\\rtva_cam0"  # optional when discovery works
health_url = "http://127.0.0.1:8765/health"
queue_max = 3
drop_policy = "drop_new"
connect_timeout_ms = 1500
