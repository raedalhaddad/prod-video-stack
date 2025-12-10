# Segment-based Recorder (Phase 2)

This module provides a client-side **Recorder** that drives the native
producer's segment-based `/clip` API to build final MP4 clips from existing
segments.

The recorder is responsible for turning *analysis events* (motion windows,
retrieval hits, etc.) into trimmed clips that are aligned with the producer's
segment timeline and sidecars.

---

## Scope

- Lives in `record/recorder.py`.
- Talks to the native producer via:
  - `POST /clip` — enqueue a clip job over existing segments.
  - `GET /clip/{clip_id}` — poll job status and retrieve final clip metadata.
- Optionally works alongside `POST /force_rotate` (called by analysis code,
  not by `Recorder` itself) to force-close the current open segment when needed.

Phase 2 ("Recorder via /clip") assumes the native producer already writes a
rolling set of MP4 segments and exposes a `/clip` endpoint that can assemble
clips from those segments.

---

## Key types

### `RecorderConfig`

A small configuration dataclass that tells the recorder how to reach the
producer and where to put clips.

Conceptually it includes:

- `producer_base_url`: Base URL for the producer
  (e.g. `http://127.0.0.1:8765`).
- `default_out_dir`: Optional default directory where completed clips should
  land (can be overridden per-request).
- `default_label`: Optional default label applied to all clip requests.
- `max_wait_ms`: Default maximum time to wait for a clip job to finish.
- `poll_interval_ms`: How often to poll `GET /clip/{id}` while a job is
  running.

`recorder_config_from_cfg(cfg_module)` provides a thin shim from an analysis
config module (e.g. `analysis.motion.config`) to `RecorderConfig`. Users are
expected to wire this to their own `.env`/TOML/environment preferences.

---

### `ClipRequest`

Represents a single clip request from the analysis side:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ClipRequest:
    # Window in producer PTS units (recommended for production)
    start_pts_ns: Optional[int] = None
    stop_pts_ns: Optional[int] = None

    # Window in epoch-ms (used when PTS is not available)
    start_ms: Optional[float] = None
    stop_ms: Optional[float] = None

    # Optional clip metadata / overrides
    label: Optional[str] = None
    out_dir: Optional[Path] = None
    filename: Optional[str] = None

    # Optional per-request overrides (fall back to RecorderConfig defaults)
    preroll_ms: Optional[int] = None
    postroll_ms: Optional[int] = None
    max_wait_ms: Optional[int] = None
```

Rules:

- You must provide **either** a PTS window or an epoch-ms window.
- If both PTS and epoch-ms are present, **PTS takes precedence**.
- The recorder will normalize the window and reject invalid ranges with a
  `ClipRequestError`, for example:
  - `stop <= start`
  - epoch-ms window mapping to a **negative PTS** (via `Timebase`)

---

### `ClipResult`

Represents the final outcome of a clip job:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any, Optional

@dataclass
class ClipResult:
    job_id: str
    status: str                # "done" or "error"
    mode: str                  # "pts" or "epoch"
    path: Optional[Path]       # final clip path if status == "done"

    # Normalized window as seen by the recorder
    start_pts_ns: Optional[int] = None
    stop_pts_ns: Optional[int] = None
    start_ms: Optional[float] = None
    stop_ms: Optional[float] = None

    # Raw JSON payload from the last GET /clip/{id} response
    raw: Optional[Mapping[str, Any]] = None
```

If the job failed or the producer returned a non-terminal state, the recorder
raises a `ClipError` (or subtype) instead of returning a `ClipResult`.

---

### `Recorder`

The main orchestration class:

```python
from common.timebase import Timebase
from record.recorder import Recorder, RecorderConfig, ClipRequest, ClipResult

rec = Recorder(cfg: RecorderConfig, timebase: Optional[Timebase] = None)
clip = rec.request_clip(req: ClipRequest)  # -> ClipResult
```

Responsibilities:

- Validate and normalize the requested time window.
- Construct a `POST /clip` body in either PTS or epoch-ms mode.
- Issue the HTTP request and parse the initial response.
- Poll `GET /clip/{clip_id}` until the job reaches a terminal status or
  times out.
- Return a `ClipResult` or raise a `ClipError` / `ClipTimeoutError` on
  failure.

---

## Timebase & window modes

The recorder supports two window modes.

### 1. PTS mode (recommended for production)

In PTS mode you pass `start_pts_ns` and `stop_pts_ns` on the `ClipRequest`.
These values live in the same PTS units the producer uses for segments
(`pts_units` from `/health.timebase`).

- If `timebase` is provided when constructing `Recorder`, it is **not** used
  to convert the clip window when PTS is provided; PTS is treated as the
  canonical clock.
- This is the most precise way to trim clips because it avoids any
  epoch/PTS round-tripping.

Typical flow for motion events:

1. The reader exposes frames as `(frame_bgr, ts_ms, pts_ns, frame_id)`.
2. The motion engine collects `event_start_pts_ns` / `event_stop_pts_ns`.
3. Preroll/postroll are applied directly in PTS units.
4. The recorder is called with a PTS-only `ClipRequest`.

The epoch timestamps (`ts_ms`) are still useful for sidecars and UI, but the
cut lines are driven entirely by PTS.

---

### 2. Epoch-ms mode (used when PTS is not available)

In epoch-ms mode you pass `start_ms` and `stop_ms` on the `ClipRequest`:

- If the recorder was constructed with a `Timebase` (see
  `common.timebase.Timebase`), it will map the epoch window into PTS using
  `epoch_ms_to_pts()` and then behave like PTS mode under the hood.
- If no `Timebase` is provided, the recorder sends the epoch window directly
  as `start_epoch_ms` / `stop_epoch_ms` to the producer.

Epoch mode is primarily intended for:

- Quick smoke tests.
- Scenarios where the analysis side only has epoch timestamps and cannot
  access PTS.

For production paths, prefer PTS mode once PTS is available in the analysis
pipeline.

---

## Producer `/clip` contract (summary)

The recorder assumes a `/clip` API roughly of the form:

- `POST /clip` with a JSON body containing either:
  - `start_pts_ns` / `stop_pts_ns`, or
  - `start_epoch_ms` / `stop_epoch_ms`,
  plus optional `label`, `out_dir`, `filename`, and `max_wait_ms`.
- The initial response includes a job identifier:
  - Some implementations return `{"clip_id": "..."}`.
  - Others may use `{"id": "..."}`.
  - The recorder supports **both** shapes.
- `GET /clip/{clip_id}` returns the job status and (on success) the final
  clip metadata. The recorder is tolerant of both:

  ```json
  {
    "ok": true,
    "status": "done",
    "path": "D:/.../clip_foo.mp4",
    "message": null
  }
  ```

  and:

  ```json
  {
    "ok": true,
    "clip": {
      "clip_id": "clip-0000000000000003",
      "status": "done",
      "path": "D:/.../clip_foo.mp4"
    },
    "message": null
  }
  ```

On error, the producer may return:

- HTTP 409 with `"message": "no segments overlap requested window (possibly purged)"`.
- HTTP 404 for unknown `clip_id`.
- Other non-2xx statuses.

These are surfaced as `ClipHttpError` exceptions.

---

## Error handling

The recorder raises:

- `ClipRequestError` for invalid client-side windows (e.g. mixed/invalid
  parameters, negative mapped PTS, `stop <= start`).
- `ClipHttpError` for HTTP-level or JSON-level problems talking to the
  producer.
- `ClipTimeoutError` if a clip job does not reach a terminal state within
  `max_wait_ms`.

Callers should treat any exception as a failure to create the clip and
decide whether to retry, drop, or log the event.

---

## Example usage

### 1. Epoch-ms smoke test (development)

This is a simplified example for local testing; it assumes you have a running
producer exposing `/health`, `/force_rotate`, and `/clip`:

```python
import requests
from analysis.motion import config as cfg
from common.timebase import timebase_from_health_payload
from record.recorder import Recorder, ClipRequest, recorder_config_from_cfg

# Snapshot "now" in epoch-ms
health = requests.get(cfg.HEALTH_URL, timeout=1.0).json()
tb = timebase_from_health_payload(health)
now_ms = float(health.get("ts_ms", tb.base_epoch_ms))

# Define a window in the (recent) past
start_ms = now_ms - 5000
stop_ms = now_ms - 2000

# Optionally force-rotate to seal the current segment
requests.post(cfg.FORCE_ROTATE_URL, timeout=1.0)

# Build recorder in epoch mode (no Timebase)
rec_cfg = recorder_config_from_cfg(cfg)
rec = Recorder(cfg=rec_cfg, timebase=None)

req = ClipRequest(
    start_ms=start_ms,
    stop_ms=stop_ms,
    label="smoke_test",
)
result = rec.request_clip(req)
print("status:", result.status)
print("path:", result.path)
```

---

### 2. PTS-based event clipping (production pattern)

In the production motion pipeline, you would typically:

1. Track events in PTS space:

   ```python
   event_start_pts_ns = first_frame.pts_ns
   event_stop_pts_ns = last_frame.pts_ns
   ```

2. Apply preroll/postroll in PTS units using the `Timebase` scale.
3. Optionally call `/force_rotate` once the event is over to seal the current
   segment.
4. Call the recorder in PTS mode:

   ```python
   import requests
   from analysis.motion import config as cfg
   from common.timebase import timebase_from_health_payload
   from record.recorder import Recorder, ClipRequest, recorder_config_from_cfg

   health = requests.get(cfg.HEALTH_URL, timeout=1.0).json()
   tb = timebase_from_health_payload(health)

   rec_cfg = recorder_config_from_cfg(cfg)
   rec = Recorder(cfg=rec_cfg, timebase=tb)

   req = ClipRequest(
       start_pts_ns=clip_start_pts_ns,
       stop_pts_ns=clip_stop_pts_ns,
       label="motion_event",
   )
   clip = rec.request_clip(req)
   print("status:", clip.status, "path:", clip.path)
   ```

This design keeps the cut lines in the producer's native PTS clock, while
still allowing the rest of the analysis stack to reason in epoch-ms for
sidecars and UI.
