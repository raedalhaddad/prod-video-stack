from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional

from analysis.motion.clip_manager import MotionClipConfig, MotionClipManager
from analysis.motion.engine import MotionEngine
from analysis.motion.events import MotionEventBuilder, MotionEventConfig
from analysis.motion.model import MotionConfig
from analysis.motion.sidecar import MotionSidecarWriter
from capture.nonblocking_adapter import wrap_nonblocking
from capture.reader import ReaderConfig, ReaderFactory
from common.frame import Frame
from common.time import now_ms
from common.timebase import Timebase, timebase_from_health_payload
from record.recorder import Recorder, RecorderConfig

_LOG = logging.getLogger(__name__)


def _fetch_timebase(health_url: str, timeout_s: float = 1.0) -> Optional[Timebase]:
    """Best-effort fetch of the producer timebase from /health."""
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        tb = timebase_from_health_payload(payload)
        _LOG.info(
            "Discovered timebase from %s: base_epoch_ms=%.3f base_pts_ns=%d units=%s",
            health_url,
            tb.base_epoch_ms,
            tb.base_pts_ns,
            tb.pts_units,
        )
        return tb
    except Exception as exc:  # best-effort
        _LOG.warning(
            "Failed to fetch timebase from %s (falling back to epoch-ms mode): %s",
            health_url,
            exc,
        )
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run motion detection + motion-gated recording pipeline.",
    )
    ap.add_argument(
        "--health",
        type=str,
        default="http://127.0.0.1:8765/health",
        help="Producer /health endpoint used for timebase and diagnostics.",
    )
    ap.add_argument(
        "--producer",
        type=str,
        default="http://127.0.0.1:8765",
        help="Base URL for the producer HTTP API (e.g. http://127.0.0.1:8765).",
    )
    ap.add_argument(
        "--ctrl-pipe",
        type=str,
        default=r"\\.\pipe\rtva_cam0_ctrl",
        help="Named pipe for SHM discovery handshake.",
    )
    ap.add_argument(
        "--prefer",
        type=str,
        choices=["shm", "null"],
        default="shm",
        help='Reader backend to use ("shm" for prod, "null" for synthetic frames).',
    )
    ap.add_argument(
        "--motion-sidecar",
        type=str,
        default="motion_events.jsonl",
        help="Path to write motion-event JSONL sidecar.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional directory where clips should be written.",
    )
    ap.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="If > 0, stop after this many seconds; otherwise run until Ctrl+C.",
    )

    # Basic motion-event tuning (hysteresis etc.)
    ap.add_argument(
        "--motion-score-on",
        type=float,
        default=0.5,
        help="Motion score threshold to turn events on.",
    )
    ap.add_argument(
        "--motion-score-off",
        type=float,
        default=0.3,
        help="Motion score threshold to turn events off.",
    )
    ap.add_argument(
        "--motion-min-event-ms",
        type=float,
        default=500.0,
        help="Minimum event duration in ms before emitting.",
    )
    ap.add_argument(
        "--motion-min-gap-ms",
        type=float,
        default=400.0,
        help="Minimum gap below score_off in ms to end an event.",
    )

    # Clip pre/post-roll + merge
    ap.add_argument(
        "--clip-preroll-ms",
        type=int,
        default=1500,
        help="Clip preroll in ms before each motion event window.",
    )
    ap.add_argument(
        "--clip-postroll-ms",
        type=int,
        default=500,
        help="Clip postroll in ms after each motion event window.",
    )
    ap.add_argument(
        "--clip-merge-gap-ms",
        type=int,
        default=2000,
        help="Merge clip windows when the gap between them is <= this value (ms).",
    )
    ap.add_argument(
        "--clip-min-ms",
        type=int,
        default=3000,
        help="Minimum clip duration in ms after preroll/postroll adjustments.",
    )
    ap.add_argument(
        "--clip-max-ms",
        type=int,
        default=120_000,
        help="Maximum clip duration in ms (0 disables).",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    ap.add_argument(
        "--enable-force-rotate",
        action="store_true",
        help="Call /force_rotate on the producer before each motion clip request.",
    )
    ap.add_argument(
        "--force-rotate-timeout-ms",
        type=int,
        default=1000,
        help="Timeout in ms for /force_rotate when enabled.",
    )
    ap.add_argument(
        "--force-rotate-require-ok",
        action="store_true",
        help=(
            "If set, skip clip requests when /force_rotate fails or returns "
            "non-2xx. If not set, log a warning but still try /clip."
        ),
    )

    return ap


def main(argv: Optional[list[str]] = None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------ reader

    reader_cfg = ReaderConfig(
        prefer=args.prefer,
        ctrl_pipe=args.ctrl_pipe,
        health_url=args.health,
    )
    reader = ReaderFactory.from_config(reader_cfg)
    reader = wrap_nonblocking(
        reader,
        queue_max=3,
        drop_policy="drop_old",
        start_timeout_s=6.0,
        close_timeout_s=0.75,
    )

    # ------------------------------------------------------------------ timebase + recorder

    timebase = _fetch_timebase(args.health)
    if timebase is None:
        raise RuntimeError(
            "Motion pipeline requires producer timebase; " "failed to fetch from /health."
        )
    out_dir_path = Path(args.out_dir) if args.out_dir else None

    rec_cfg = RecorderConfig(
        producer_base_url=args.producer,
        default_out_dir=out_dir_path,
        default_label="motion",
    )
    recorder = Recorder(cfg=rec_cfg, timebase=timebase)

    # ------------------------------------------------------------------ motion stack

    motion_cfg = MotionConfig()
    motion_engine = MotionEngine(timebase=timebase, config=motion_cfg)

    event_cfg = MotionEventConfig(
        score_on=args.motion_score_on,
        score_off=args.motion_score_off,
        min_event_ms=args.motion_min_event_ms,
        min_gap_ms=args.motion_min_gap_ms,
    )
    event_builder = MotionEventBuilder(timebase=timebase, config=event_cfg)

    force_rotate_url = None
    if args.enable_force_rotate:
        base = args.producer.rstrip("/")
        force_rotate_url = f"{base}/force_rotate"

    clip_cfg = MotionClipConfig(
        preroll_ms=args.clip_preroll_ms,
        postroll_ms=args.clip_postroll_ms,
        min_clip_ms=args.clip_min_ms,
        max_clip_ms=args.clip_max_ms,
        merge_if_gap_ms=args.clip_merge_gap_ms,
        cooldown_ms=0,
        label="motion",
        out_dir=out_dir_path,
        max_wait_ms=None,
        force_rotate_url=force_rotate_url,
        force_rotate_timeout_s=args.force_rotate_timeout_ms / 1000.0,
        require_force_rotate_ok=args.force_rotate_require_ok,
    )
    clip_mgr = MotionClipManager(recorder=recorder, config=clip_cfg)

    sidecar_path = Path(args.motion_sidecar)
    _LOG.info("Writing motion events to %s", sidecar_path)

    # ------------------------------------------------------------------ main loop

    reader.start()
    t0 = time.time()

    with MotionSidecarWriter(sidecar_path) as sidecar:
        try:
            while True:
                tup = reader.read()
                if tup is None:
                    # Allow cooperative cancellation and max-seconds limit while idle.
                    if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
                        _LOG.info("Reached max-seconds=%d, exiting loop.", args.max_seconds)
                        break
                    time.sleep(0.005)
                    continue

                frame_bgr, pts_raw, frame_id = tup

                # Interpret pts_raw as producer PTS in nanoseconds.
                # This should come directly from the native reader.
                if pts_raw is None:
                    # Fallback: we lose strict alignment, but keep running.
                    epoch_ms = now_ms()
                else:
                    pts_ns = int(pts_raw)
                    epoch_ms = float(timebase.pts_to_epoch_ms(pts_ns))

                frame = Frame(
                    img=frame_bgr,
                    pts_ms=epoch_ms,
                    frame_id=int(frame_id),
                )

                res = motion_engine.step(frame, recording_active=False)
                events = event_builder.consume(res)
                for ev in events:
                    sidecar.write_event(ev)
                    clip_mgr.handle_event(ev)

                if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
                    _LOG.info("Reached max-seconds=%d, exiting loop.", args.max_seconds)
                    break

            # End-of-stream: flush remaining events → sidecar + clips.
            final_events = event_builder.flush()
            for ev in final_events:
                sidecar.write_event(ev)
                clip_mgr.handle_event(ev)

        except KeyboardInterrupt:
            _LOG.info("KeyboardInterrupt received, flushing and shutting down.")
            final_events = event_builder.flush()
            for ev in final_events:
                sidecar.write_event(ev)
                clip_mgr.handle_event(ev)
        finally:
            clip_mgr.flush()
            with contextlib.suppress(Exception):
                reader.close()


if __name__ == "__main__":  # pragma: no cover
    main()
