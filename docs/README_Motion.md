# Motion Detection & Motion-Gated Recorder

This document describes the **motion pipeline** in `prod-video-stack` — how we
detect motion, turn it into events, and drive the native producer's
segment-based `/clip` endpoint to create motion-gated recordings.

It is meant to sit alongside the reader and recorder docs and focuses on the
analysis-side orchestration that bridges the two.

---

## 1. Scope & directory layout

The motion stack is a small, self-contained layer that sits between:

- The **reader** (frames from the native producer via SHM + `/health`), and
- The **recorder** (segment-based `/clip` endpoint).

It lives primarily under:

- **`analysis/motion/`**
  - `engine.py` – per-frame motion engine.
  - `events.py` – motion event builder (score → events).
  - `clip_manager.py` – motion-gated clip manager (events → `/clip`).
  - `sidecar.py` – motion event sidecar writer.
  - `config.py` – motion-related config shims / dataclasses.
  - `model.py` – core motion dataclasses (`MotionResult`, `MotionEvent`,
    configs, etc.).

Other dependencies:

- **Reader**:
  - `capture.reader`, `capture.nonblocking_adapter`, `capture.stats_adapter`
  - Responsible for delivering `(img, pts_ns, frame_id)` from the native
    producer, plus basic perf stats.
- **Recorder**:
  - `record.recorder.Recorder`
  - Wraps `/force_rotate` and `/clip` into a clean Python API.
- **Core utilities**:
  - `common.timebase.Timebase`
  - `common.frame.Frame`

---

## 2. Timebase & clocks

The entire motion pipeline is designed around a **single, shared timebase**
exposed by the native producer’s `/health` endpoint, something like:

```json
{
  "timebase": {
    "base_epoch_ms": 1765474598768.0,
    "base_pts_ns": 0,
    "pts_units": "ns"
  },
  "ts_ms": 1765474599000.0
}
```

On the Python side:

- `common.timebase.Timebase` encapsulates this mapping.
- The reader exposes **raw PTS** from SHM:
  - Example: `frame.pts_ns` coming directly from the native stream.
- `Timebase.pts_to_epoch_ms(pts_ns)` maps PTS → epoch-ms for logging and UI.
- `Timebase.epoch_ms_to_pts(epoch_ms)` maps epoch-ms → PTS (for `/clip`).

**Key rule for motion:**

> Motion results and events are timestamped using producer PTS, mapped to
> epoch-ms via `Timebase`. The host wall-clock is not used for motion timing.

This ensures:

- Motion windows and segment horizons live on the **same clock**.
- PTS-mode `/clip` requests line up mathematically with segment PTS ranges.

---

## 3. Motion engine (`analysis.motion.engine`)

### 3.1 Purpose

`MotionEngine` turns raw frames into scalar motion scores and lightweight
per-frame metadata. It is deliberately simpler than any dev-only pipeline and
is meant to:

- Be **stable** and predictable for production.
- Never crash the pipeline, even if OpenCV is absent or misbehaves.
- Provide a consistent `MotionResult` API for downstream consumers
  (events, sidecars, clips).

### 3.2 Input type

The engine consumes `common.frame.Frame`:

```python
@dataclass
class Frame:
    img: np.ndarray      # H×W×3 BGR image
    pts_ms: float        # epoch-ms derived from PTS
    frame_id: int        # monotonically increasing frame index
```

### 3.3 Output type (`MotionResult`)

Defined in `analysis.motion.model`:

```python
@dataclass
class MotionResult:
    is_motion: bool
    score: float          # scalar in [0, 1]
    pts_ms: float         # epoch-ms
    frame_id: int
    area_frac: float = 0.0

    # Forward-looking fields for richer semantics:
    wind_like: bool = False
    real_override: bool = False
    foliage_motion_frac: float = 0.0
    wind_idx: float = 0.0
    wind_state: str = "Unknown"
    zones_active_frac: float = 0.0
```

### 3.4 Configuration (`MotionConfig`)

`MotionConfig` (in `config.py` / `model.py`) provides tuning knobs, typically:

- Background subtraction:
  - `fg_history`
  - `fg_var_threshold`
  - `fg_detect_shadows`
- Mask cleanup:
  - `fg_threshold`
  - `morph_kernel` (kernel size for morphological open)
- Scoring:
  - `score_scale`
  - `score_cap`
  - `min_motion_area_frac`
- Performance:
  - `stride` (compute a fresh mask every N frames, reuse in between).

### 3.5 Behavior

High-level steps per frame:

1. Convert `Frame.img` to a `np.ndarray`.
2. Apply MOG2 background subtractor to get `fg_mask`.
3. Threshold `fg_mask` to a binary mask; optional morphological open.
4. Compute:
   - `motion_px = (mask > 0).sum()`
   - `total_px = mask.size`
   - `area_frac = motion_px / total_px`
5. Compute scalar score:
   - `raw_score = area_frac * score_scale`
   - `score = clamp(raw_score, 0.0, score_cap)`
6. Decide motion:
   - `is_motion = (area_frac >= min_motion_area_frac) and (score > 0.0)`
7. Fill `MotionResult` with:
   - `pts_ms` and `frame_id` from the frame
   - derived `score`, `area_frac`, and flags.

**Stride behavior:**

- If `stride > 1`, the engine recomputes the mask every Nth frame.
- For skipped frames, it:
  - Reuses the previous `MotionResult`’s score and area,
  - Updates `pts_ms` and `frame_id`,
  - Returns a new `MotionResult` reflecting the current timestamp.

**Failure modes:**

- If OpenCV is missing, the subtractor fails, or the frame is malformed,
  the engine returns:

```python
MotionResult(
    is_motion=False,
    score=0.0,
    pts_ms=frame.pts_ms,
    frame_id=frame.frame_id,
)
```

No exceptions are propagated to callers.

---

## 4. Motion events (`analysis.motion.events`)

Per-frame scores are noisy. The `MotionEventBuilder` aggregates `MotionResult`
objects into **motion events**, each describing a contiguous window of motion
activity.

### 4.1 MotionEvent dataclass

```python
@dataclass
class MotionEvent:
    start_pts_ns: int
    stop_pts_ns: int
    start_ms: float
    stop_ms: float
    max_score: float
    wind_like: bool
    real_override: bool
    foliage_motion_frac: float
    wind_idx: float
    wind_state: str
    zones_active_frac: float
```

- `start_pts_ns` / `stop_pts_ns` are derived using the shared `Timebase`.
- `start_ms` / `stop_ms` are epoch-ms equivalents for logging, UI, and
  sidecars.
- `max_score` tracks the peak motion score during the event.

### 4.2 Configuration (`MotionEventConfig`)

Key parameters:

- `score_on`: threshold above which an event turns **on**.
- `score_off`: threshold below which an active event can turn **off**.
- `min_event_ms`: minimum duration to consider an event valid.
- `min_gap_ms`: minimum gap between events to treat them as separate.
- `merge_gap_ms`: if two events are closer than this, they are merged.

Standard behavior:

- **Start** an event when `score` first crosses `score_on`.
- Keep the event active until the score drops below `score_off` and
  duration/gap constraints are satisfied.
- Emits `MotionEvent` objects as the stream progresses; events are not
  tied to pipeline-wide flushes.

### 4.3 Usage pattern

Typical pattern in an analysis loop:

```python
builder = MotionEventBuilder(timebase=tb, config=event_cfg)

for result in motion_results_stream:
    for ev in builder.consume(result):
        handle_motion_event(ev)

# At shutdown:
for ev in builder.flush():
    handle_motion_event(ev)
```

Where `handle_motion_event()` might:

- Write to the motion sidecar.
- Forward the event to `MotionClipManager` for clip creation.

---

## 5. Motion-gated clip manager (`analysis.motion.clip_manager`)

`MotionClipManager` bridges motion events to the recorder. It:

- Accepts `MotionEvent` objects.
- Merges / extends them according to clip config.
- Issues `/force_rotate` + `/clip` requests via `record.recorder.Recorder`.

### 5.1 Clip config (`MotionClipConfig`)

Key fields:

- `preroll_ms`: extra time before the event window.
- `postroll_ms`: extra time after the event window.
- `min_clip_ms`: minimum clip duration; shorter events are padded or
  dropped based on config.
- `max_clip_ms`: optional maximum clip duration; longer events may be split
  or truncated in future revisions.
- `merge_if_gap_ms`: merge successive events whose gap is less than this.
- `cooldown_ms`: optional “quiet period” after a clip before triggering
  another.
- `label`: label to attach to clips (e.g. `"motion"`).
- `out_dir`: optional override output directory for clips.
- `max_wait_ms`: how long the clip manager should wait for `/clip` jobs
  to finish, if it is responsible for polling.

### 5.2 Real-time semantics

The core design is **real-time**:

- As soon as a motion event is finalized (after merge/gap/cooldown logic),
  the manager **immediately**:

  1. Calls `POST /force_rotate` to seal the current segment.
  2. Waits for HTTP 200 from `/force_rotate`.
  3. Issues a `/clip` request covering the motion window plus pre/post-roll.

- `flush()` is only used at shutdown to handle a trailing in-progress event.

This gives users a strong guarantee:

> Every completed motion event triggers a clip request right away, not just
> at pipeline shutdown.

### 5.3 PTS-mode clip requests

Given a `MotionEvent` with `start_ms` and `stop_ms`, the manager:

1. Applies `preroll_ms` and `postroll_ms` to build the **clip window**.
2. Uses the shared `Timebase` to convert epoch-ms → PTS:
   - `start_pts_ns = tb.epoch_ms_to_pts(start_ms)`
   - `stop_pts_ns  = tb.epoch_ms_to_pts(stop_ms)`
3. Constructs a `ClipRequest` in **PTS mode**:
   - PTS values set,
   - Epoch-ms fields set only if useful for inspection/logging.
4. Passes the request to `Recorder.request_clip()`.

PTS mode is preferred for precise trimming because it bypasses any host
clock vs producer clock drift.

### 5.4 Interaction with `/force_rotate`

To avoid race conditions where `/clip` is asked for a time range slightly
beyond the current finalized segment:

- `/force_rotate` on the native side was updated so that it returns **only
  after**:
  - The current segment is finalized and written.
  - The new “current” segment is indexed and visible to `/clip`.

The clip manager:

- Never calls `/clip` until it has received an HTTP 200 from `/force_rotate`
  for the current event.
- This, combined with the shared `Timebase`, eliminates spurious “future
  window” 409 errors for recent motion.

### 5.5 Error handling

When `/clip` returns an error (e.g. HTTP 409):

- The recorder raises a `ClipHttpError` containing:
  - `status_code`,
  - Raw response body,
  - Parsed JSON payload.
- The motion clip manager catches this and logs a detailed warning including:
  - The requested window,
  - The producer’s error `message`,
  - Any enriched fields such as `earliest_ms` / `latest_ms`.
- The error is treated as **non-fatal**. The manager continues to process
  subsequent motion events.

---

## 6. Motion event sidecar (`analysis.motion.sidecar`)

The motion pipeline logs event metadata in a **JSONL sidecar**, one line per
event. This allows:

- Auditing motion thresholds and behavior.
- Cross-checking events vs clips.
- Offline analytics and visualization.

### 6.1 Format

Each line is roughly:

```json
{
  "type": "motion_event",
  "start_pts_ns": 21368513175280,
  "stop_pts_ns": 21380836392321,
  "start_ms": 1765397201708.882,
  "stop_ms": 1765397214032.099,
  "max_score": 0.7142,
  "wind_like": false,
  "real_override": true,
  "foliage_motion_frac": 0.0,
  "wind_idx": 0.0,
  "wind_state": "Unknown",
  "zones_active_frac": 0.0
}
```

Conventions:

- All time fields are derived from the **shared `Timebase`**.
- `start_ms` / `stop_ms` are epoch-ms values consistent with:
  - `/health.ts_ms`,
  - Segment indices,
  - `/clip` request/response payloads.

**Phase 3.1 scope:**

- Only **event-level** motion sidecar is emitted.
- No per-frame motion sidecar yet.
- No motion bounding boxes yet.

These are planned for a later phase focused on full frame-accurate motion
auditing.

---

## 7. Integrating the motion stack

There is intentionally **no** hard-coded “motion module” wrapper file; the
stack is designed so that any analysis runner can wire it together using a
few simple steps.

A typical integration flow looks like:

1. **Discover timebase**
   - Call `/health` on the producer.
   - Build a `Timebase` from `base_epoch_ms`, `base_pts_ns`, and `pts_units`.

2. **Construct your reader**
   - Build the SHM reader (and any non-blocking / stats adapters).
   - Ensure each frame you pull has:
     - `img` (BGR `np.ndarray`),
     - `pts_ns` or equivalent,
     - `frame_id`.

3. **Wrap frames into `Frame`**
   - Convert PTS → epoch-ms via `Timebase.pts_to_epoch_ms`.
   - Create `Frame(img=..., pts_ms=..., frame_id=...)`.

4. **Build the motion engine and configs**
   - Create `MotionConfig`, `MotionEventConfig`, and `MotionClipConfig` from
     your config system (TOML, YAML, env, etc.).
   - Construct:
     - `MotionEngine(timebase=tb, config=motion_cfg)`
     - `MotionEventBuilder(timebase=tb, config=event_cfg)`
     - `Recorder(...)`
     - `MotionClipManager(recorder=recorder, config=clip_cfg, timebase=tb)`

5. **Wire the main loop**
   - For each `Frame`:
     1. `result = engine.step(frame)`
     2. For each `event` emitted by `builder.consume(result)`:
        - Write the event to the motion sidecar.
        - Call `clip_manager.handle_event(event)` to request a clip.

6. **Shutdown**
   - On pipeline shutdown, call:
     - `for ev in builder.flush(): ...`
     - `clip_manager.flush()`
   - Close the motion sidecar file.

This keeps the motion stack flexible: you can host it in a standalone
process, embed it inside a broader analysis service, or wire it into any
orchestration layer without depending on a specific “motion module” wrapper.

---

## 8. 409 errors & segment horizon diagnostics

The recorder README describes the `/clip` API in detail; the motion module
adds a couple of pieces specifically for motion-driven clips.

### 8.1 Enriched 409 responses

When `/clip` returns HTTP 409 because no segments overlap the requested
window, the producer now includes extra fields:

```json
{
  "ok": false,
  "clip_id": null,
  "status": "error",
  "dropped_tail_ms": 0,
  "message": "no segments overlap requested window (possibly purged)",
  "earliest_ms": 1765493315733.0,
  "latest_ms": 1765493328806.0
}
```

Where:

- `earliest_ms`: earliest segment start epoch-ms in the current index.
- `latest_ms`: latest segment end epoch-ms in the current index.

All in the same timebase as motion events and `/health.ts_ms`.

The motion clip manager uses these for logging and can classify the error as:

- `window_before_earliest`
- `window_after_latest`
- `window_within_horizon_no_overlap`
- Or `unknown` if enrichment is unavailable.

### 8.2 Segment horizon debug tool

For deeper debugging (primarily for developers), there is a small helper:

- `tools/debug_segments_horizon.py`

It parses producer logs containing lines like:

```text
[clip] indexed segment: start=1765493315733 end=1765493328806 ms location=D:/media/security_camera_app/segments/seg_08494.mp4
```

Given a log file and an optional motion window, it prints:

- The last N segments (index, start_ms, end_ms, duration, path).
- The overall segment horizon.
- A classification of the given event window vs that horizon.

This is useful if you see repeated 409s and want to verify whether:

- The motion window truly lies outside the retained segments, or
- There is a mismatch between on-disk segments and the `/clip` segment
  index.

---

## 9. Summary

The motion module in `prod-video-stack` provides:

- A **timebase-aligned** motion engine using producer PTS.
- A configurable **event builder** that turns noisy per-frame scores into
  stable motion events.
- A **motion-gated clip manager** that:
  - Seals segments via `/force_rotate`,
  - Issues `/clip` requests in PTS mode,
  - Handles enrichment and error cases gracefully.
- A JSONL **motion event sidecar** for auditing and downstream use.

Together with the reader and recorder, this forms a production-ready path:

> Native producer (SHM + timebase) → Motion engine → Motion events →
> Motion-gated `/clip` → Segment-based MP4 clips.

Future phases will extend this with:

- Per-frame motion sidecars for frame-accurate auditing,
- Motion bounding boxes and zone semantics,
- Strong invariants that guarantee clips cover all motion frames with
  well-bounded pre/post roll.
