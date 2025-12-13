from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests  # ⬅️ NEW

from common.timebase import Timebase
from record.recorder import ClipHttpError, ClipRequest

from .events import MotionEvent

_LOG = logging.getLogger(__name__)


@dataclass
class MotionClipConfig:
    """Configuration for turning MotionEvents into clip requests."""

    # Pre/post-roll around each event, in milliseconds (epoch-ms space).
    preroll_ms: int = 2000
    postroll_ms: int = 3000

    # Minimum and maximum clip duration (after preroll/postroll and merging).
    min_clip_ms: int = 3000
    max_clip_ms: int = 120_000

    # If two candidate clip windows are closer than this in time (gap between
    # stop_ms of the first and start_ms of the second), they are merged into a
    # single request.
    merge_if_gap_ms: int = 2000

    # Optional cooldown between clip *requests* in ms. If a new window starts
    # before last_clip_stop_ms + cooldown_ms, it is merged into the previous
    # window instead of creating a separate request.
    cooldown_ms: int = 0

    # Optional defaults for clip metadata / behaviour.
    label: str = "motion"
    out_dir: Optional[Path] = None
    max_wait_ms: Optional[int] = None

    # NEW: optional /force_rotate integration.
    force_rotate_url: Optional[str] = None
    force_rotate_timeout_s: float = 1.0
    # If True: skip the clip request when /force_rotate fails or returns non-2xx.
    # If False: log a warning but still try /clip.
    require_force_rotate_ok: bool = False


class MotionClipManager:
    """Consume MotionEvents and issue clip requests via a Recorder-like object.

    The recorder is expected to expose:

        - ``timebase`` attribute (Optional[Timebase])
        - ``request_clip(ClipRequest)`` method

    The manager keeps a single pending clip window that may absorb multiple
    events depending on ``merge_if_gap_ms`` and ``cooldown_ms``. At the end of
    the stream, call :meth:`flush` to ensure the final window is issued.
    """

    def __init__(
        self,
        recorder: Any,
        config: Optional[MotionClipConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._recorder = recorder
        self._cfg = config or MotionClipConfig()
        self._log = logger or _LOG

        self._timebase: Optional[Timebase] = getattr(recorder, "timebase", None)

        # Pending clip window in epoch-ms.
        self._pending_start_ms: Optional[float] = None
        self._pending_stop_ms: Optional[float] = None

        # Last issued clip stop time in epoch-ms (for cooldown logic).
        self._last_clip_stop_ms: Optional[float] = None

    # ------------------------------------------------------------------ helpers

    def _adjust_window(self, start_ms: float, stop_ms: float) -> tuple[float, float]:
        """Apply preroll/postroll and min/max duration constraints."""
        cfg = self._cfg

        start = start_ms - float(cfg.preroll_ms)
        stop = stop_ms + float(cfg.postroll_ms)

        if start < 0.0:
            start = 0.0

        # Enforce minimum duration.
        duration = stop - start
        if duration < float(cfg.min_clip_ms):
            stop = start + float(cfg.min_clip_ms)
            duration = stop - start

        # Enforce maximum duration, if configured.
        if cfg.max_clip_ms > 0 and duration > float(cfg.max_clip_ms):
            stop = start + float(cfg.max_clip_ms)

        return start, stop

    def _issue_clip(self, start_ms: float, stop_ms: float) -> None:
        """Convert a finalised window into a ClipRequest and send it."""
        cfg = self._cfg
        tb = self._timebase

        # Optionally rotate the current segment before asking for a clip.
        if not self._force_rotate():
            self._log.warning(
                "Skipping motion clip because force_rotate did not succeed "
                "and require_force_rotate_ok=True"
            )
            return

        # Map epoch-ms window into PTS if we have a timebase; otherwise fall
        # back to epoch-ms mode.
        if tb is not None:
            start_pts = int(tb.epoch_ms_to_pts(start_ms))
            stop_pts = int(tb.epoch_ms_to_pts(stop_ms))
            req = ClipRequest(
                start_pts_ns=start_pts,
                stop_pts_ns=stop_pts,
                start_ms=None,
                stop_ms=None,
                label=cfg.label,
                out_dir=cfg.out_dir,
                filename=None,
                preroll_ms=None,
                postroll_ms=None,
                max_wait_ms=cfg.max_wait_ms,
            )
        else:
            req = ClipRequest(
                start_pts_ns=None,
                stop_pts_ns=None,
                start_ms=start_ms,
                stop_ms=stop_ms,
                label=cfg.label,
                out_dir=cfg.out_dir,
                filename=None,
                preroll_ms=None,
                postroll_ms=None,
                max_wait_ms=cfg.max_wait_ms,
            )

        self._log.debug(
            "Issuing motion clip request: start_ms=%.3f stop_ms=%.3f label=%s",
            start_ms,
            stop_ms,
            cfg.label,
        )
        try:
            self._recorder.request_clip(req)
        except ClipHttpError as exc:
            # Non-fatal: log and continue. Most common failure is 409 when the
            # requested window no longer overlaps available segments.
            self._log.warning("Motion clip request failed (non-fatal): %s", exc)
        else:
            self._last_clip_stop_ms = stop_ms

    # ------------------------------------------------------------------ public

    def handle_event(self, ev: MotionEvent) -> None:
        """Consume a MotionEvent and immediately emit a clip request.

        We assume event-level merging has already happened upstream in
        MotionEventBuilder, so each MotionEvent maps to one clip.
        """
        event_start_ms = float(ev.start_ms)
        event_stop_ms = float(ev.stop_ms)

        start_ms, stop_ms = self._adjust_window(event_start_ms, event_stop_ms)

        self._log.debug(
            "Handling motion event window: start_ms=%.3f stop_ms=%.3f",
            start_ms,
            stop_ms,
        )
        self._issue_clip(start_ms, stop_ms)

    def flush(self) -> None:
        """No-op for now: clips are issued eagerly on each event."""
        # We keep this for API compatibility.
        self._pending_start_ms = None
        self._pending_stop_ms = None

    def _force_rotate(self) -> bool:
        """Optionally POST /force_rotate before requesting a clip.

        Returns True if it's OK to proceed with /clip, False if we should skip
        the clip (when require_force_rotate_ok=True).
        """
        url = self._cfg.force_rotate_url
        if not url:
            return True  # feature disabled

        timeout = float(self._cfg.force_rotate_timeout_s) or 1.0
        try:
            resp = requests.post(url, timeout=timeout)
        except Exception as exc:
            self._log.warning("force_rotate POST failed: %s", exc)
            return not self._cfg.require_force_rotate_ok

        if 200 <= resp.status_code < 300:
            self._log.debug("force_rotate succeeded: HTTP %s", resp.status_code)
            return True

        self._log.warning(
            "force_rotate returned HTTP %s: %r",
            resp.status_code,
            resp.text,
        )
        return not self._cfg.require_force_rotate_ok
