from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

from common.timebase import Timebase

_LOG = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    """Configuration for the segment-based recorder.

    Parameters
    ----------
    producer_base_url:
        Base URL for the native producer HTTP API (e.g. ``"http://127.0.0.1:8765"``).
    default_out_dir:
        Optional default directory where clips should be written. If ``None``,
        the producer's own default clip directory is used.
    default_label:
        Optional default label applied to all clip requests unless overridden.
    max_wait_ms:
        Default maximum time the recorder will wait for a clip job to finish
        before raising :class:`ClipTimeoutError`.
    poll_interval_ms:
        Interval between polling ``GET /clip/{id}`` in milliseconds.
    """

    producer_base_url: str
    default_out_dir: Optional[Path] = None
    default_label: Optional[str] = None
    max_wait_ms: int = 120_000
    poll_interval_ms: int = 500


@dataclass
class ClipRequest:
    """High-level clip request accepted by :class:`Recorder`.

    The recorder prefers PTS-based clipping. If PTS fields are present, they
    are used directly. Otherwise, if epoch-ms fields are provided and a
    :class:`Timebase` is available, they will be converted to PTS internally.
    """

    # Preferred: PTS-based window on the producer timeline.
    start_pts_ns: Optional[int] = None
    stop_pts_ns: Optional[int] = None

    # Epoch-based window in the same epoch-ms space as /health.ts_ms.
    start_ms: Optional[float] = None
    stop_ms: Optional[float] = None

    # Optional clip metadata / overrides.
    label: Optional[str] = None
    out_dir: Optional[Path] = None
    filename: Optional[str] = None

    # Optional per-request overrides (fall back to RecorderConfig defaults).
    preroll_ms: Optional[int] = None
    postroll_ms: Optional[int] = None
    max_wait_ms: Optional[int] = None


@dataclass
class ClipResult:
    """Result of a completed (or failed) clip job."""

    job_id: str
    status: str
    mode: str  # "pts" or "epoch"
    path: Optional[Path]
    start_pts_ns: Optional[int] = None
    stop_pts_ns: Optional[int] = None
    start_ms: Optional[float] = None
    stop_ms: Optional[float] = None
    raw: Optional[Mapping[str, Any]] = None


class ClipError(Exception):
    """Base class for recorder-related errors."""


class ClipRequestError(ClipError):
    """Invalid clip request parameters (missing or inconsistent time range)."""


class ClipTimeoutError(ClipError):
    """The clip job did not complete within the configured timeout."""


class ClipHttpError(ClipError):
    """HTTP or response-parsing failure while talking to the producer."""


class Recorder:
    """Segment-based recorder that talks to the native producer's ``/clip`` API.

    This helper abstracts away:

    - Building the correct JSON payload for ``POST /clip`` in PTS or epoch mode.
    - Polling ``GET /clip/{id}`` until the job reaches a terminal state.
    - Converting epoch-ms windows to PTS using :class:`Timebase` when available.
    """

    def __init__(
        self,
        cfg: RecorderConfig,
        timebase: Optional[Timebase] = None,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cfg = cfg
        self._timebase = timebase
        self._session = session or requests.Session()
        self._log = logger or _LOG

    # --------------------------------------------------------------------- utils

    @property
    def timebase(self) -> Optional[Timebase]:
        return self._timebase

    def with_timebase(self, timebase: Timebase) -> "Recorder":
        """Return a shallow copy of this recorder with a different timebase."""
        return Recorder(
            cfg=self._cfg,
            timebase=timebase,
            session=self._session,
            logger=self._log,
        )

    # ----------------------------------------------------------------- main API

    def request_clip(self, req: ClipRequest) -> ClipResult:
        """Request a clip and block until completion or failure.

        Parameters
        ----------
        req:
            High-level clip parameters. See :class:`ClipRequest`.

        Returns
        -------
        ClipResult
            Structured result containing the final status and path.

        Raises
        ------
        ClipRequestError
            If the request parameters are invalid.
        ClipTimeoutError
            If the job does not reach a terminal state within the timeout.
        ClipHttpError
            If HTTP or JSON parsing fails.
        ClipError
            For other producer-reported errors.
        """
        mode, start_pts, stop_pts, start_ms, stop_ms = self._normalize_window(req)

        body: dict[str, Any] = {}
        if mode == "pts":
            body["start_pts_ns"] = int(start_pts)  # type: ignore[arg-type]
            body["stop_pts_ns"] = int(stop_pts)  # type: ignore[arg-type]
        elif mode == "epoch":
            # /clip expects i64 for epoch-ms fields, so we send integers.
            body["start_epoch_ms"] = int(start_ms)  # type: ignore[arg-type]
            body["stop_epoch_ms"] = int(stop_ms)  # type: ignore[arg-type]
        else:  # pragma: no cover - defensive
            raise ClipRequestError(f"unsupported mode: {mode!r}")

        # Pre/postroll in ms (applied after mapping inside the producer).
        if req.preroll_ms is not None:
            body["preroll_ms"] = int(req.preroll_ms)
        if req.postroll_ms is not None:
            body["postroll_ms"] = int(req.postroll_ms)

        # Optional label and output location.
        label = req.label if req.label is not None else self._cfg.default_label
        if label is not None:
            body["label"] = str(label)

        out_dir = req.out_dir if req.out_dir is not None else self._cfg.default_out_dir
        if out_dir is not None:
            body["out_dir"] = str(out_dir)

        if req.filename is not None:
            body["filename"] = str(req.filename)

        # For now we pass through max_wait_ms as a hint; the producer may ignore it.
        # The recorder still enforces its own timeout via polling.
        if req.max_wait_ms is not None:
            body["max_wait_ms"] = int(req.max_wait_ms)

        base_url = self._cfg.producer_base_url.rstrip("/")
        clip_url = f"{base_url}/clip"

        self._log.info(
            "posting /clip request mode=%s start=%r stop=%r preroll_ms=%r postroll_ms=%r label=%r out_dir=%r filename=%r",
            mode,
            start_pts if mode == "pts" else start_ms,
            stop_pts if mode == "pts" else stop_ms,
            req.preroll_ms,
            req.postroll_ms,
            label,
            out_dir,
            req.filename,
        )
        job = self._post_json(clip_url, body)

        # Producer uses "clip_id" as the identifier; support both for robustness.
        job_id_raw = job.get("id") or job.get("clip_id")
        if not job_id_raw:
            raise ClipHttpError(f"/clip response missing clip_id/id: {job!r}")

        job_id = str(job_id_raw)

        # Resolve timeouts.
        max_wait_ms = (
            int(req.max_wait_ms) if req.max_wait_ms is not None else int(self._cfg.max_wait_ms)
        )

        status_payload = self._poll_clip_job(base_url, job_id, max_wait_ms)

        status, path_str = self._extract_status_and_path(status_payload)
        path = Path(path_str) if path_str is not None else None

        result = ClipResult(
            job_id=job_id,
            status=status,
            mode=mode,
            path=path,
            start_pts_ns=start_pts,
            stop_pts_ns=stop_pts,
            start_ms=start_ms,
            stop_ms=stop_ms,
            raw=status_payload,
        )

        if status != "done":
            raise ClipError(f"clip job {job_id} finished with status={status!r}")

        self._log.info("/clip job %s completed status=%s path=%s", job_id, status, path)
        return result

    # ------------------------------------------------------------ internal bits

    def _normalize_window(
        self, req: ClipRequest
    ) -> tuple[str, Optional[int], Optional[int], Optional[float], Optional[float]]:
        """Validate and normalize the requested time window.

        Returns
        -------
        (mode, start_pts_ns, stop_pts_ns, start_ms, stop_ms)

        Where:
        - ``mode`` is "pts" or "epoch".
        - PTS values are integers in the producer's PTS units.
        - Epoch values are floats in epoch-ms.
        """
        start_pts = req.start_pts_ns
        stop_pts = req.stop_pts_ns
        start_ms = req.start_ms
        stop_ms = req.stop_ms

        has_pts = start_pts is not None and stop_pts is not None
        has_epoch = start_ms is not None and stop_ms is not None

        if not has_pts and not has_epoch:
            raise ClipRequestError("must provide either PTS or epoch-ms clip window")

        mode: str
        if has_pts:
            mode = "pts"
        elif self._timebase is not None:
            # Convert epoch-ms to PTS using the configured timebase.
            mode = "pts"
            assert start_ms is not None and stop_ms is not None

            start_pts_f = self._timebase.epoch_ms_to_pts(start_ms)
            stop_pts_f = self._timebase.epoch_ms_to_pts(stop_ms)

            if start_pts_f < 0 or stop_pts_f < 0:
                raise ClipRequestError(
                    "epoch-ms window maps to negative PTS values; "
                    f"start_ms={start_ms} stop_ms={stop_ms} "
                    f"start_pts={start_pts_f} stop_pts={stop_pts_f}"
                )

            start_pts = int(start_pts_f)
            stop_pts = int(stop_pts_f)
        else:
            # No PTS provided and no timebase available; fall back to epoch mode.
            mode = "epoch"

        if mode == "pts":
            assert start_pts is not None and stop_pts is not None
            if stop_pts <= start_pts:
                raise ClipRequestError(
                    f"invalid PTS window: start_pts_ns={start_pts} stop_pts_ns={stop_pts}"
                )
        else:
            assert start_ms is not None and stop_ms is not None
            if stop_ms <= start_ms:
                raise ClipRequestError(
                    f"invalid epoch window: start_ms={start_ms} stop_ms={stop_ms}"
                )

        return mode, start_pts, stop_pts, start_ms, stop_ms

    def _post_json(self, url: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            resp = self._session.post(url, json=body, timeout=5.0)
        except Exception as exc:  # pragma: no cover - network failure is rare in tests
            raise ClipHttpError(f"POST {url!r} failed: {exc}") from exc

        if not (200 <= resp.status_code < 300):
            raise ClipHttpError(f"POST {url!r} returned HTTP {resp.status_code}: {resp.text!r}")

        try:
            data = resp.json()
        except Exception as exc:
            raise ClipHttpError(f"POST {url!r} returned invalid JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ClipHttpError(f"POST {url!r} returned non-object JSON: {data!r}")
        return data

    def _poll_clip_job(self, base_url: str, job_id: str, max_wait_ms: int) -> Mapping[str, Any]:
        """Poll ``GET /clip/{id}`` until a terminal status or timeout."""
        poll_url = f"{base_url}/clip/{job_id}"
        deadline = time.monotonic() + max_wait_ms / 1000.0
        interval_s = max(self._cfg.poll_interval_ms, 1) / 1000.0

        last_payload: Mapping[str, Any] = {}
        while True:
            now = time.monotonic()
            if now > deadline:
                raise ClipTimeoutError(
                    f"clip job {job_id} did not complete within {max_wait_ms} ms; last_status={last_payload!r}"
                )

            try:
                resp = self._session.get(poll_url, timeout=5.0)
            except Exception as exc:  # pragma: no cover - network failure is rare in tests
                raise ClipHttpError(f"GET {poll_url!r} failed: {exc}") from exc

            if not (200 <= resp.status_code < 300):
                raise ClipHttpError(
                    f"GET {poll_url!r} returned HTTP {resp.status_code}: {resp.text!r}"
                )
            try:
                data = resp.json()
            except Exception as exc:
                raise ClipHttpError(f"GET {poll_url!r} returned invalid JSON: {exc}") from exc
            if not isinstance(data, Mapping):
                raise ClipHttpError(f"GET {poll_url!r} returned non-object JSON: {data!r}")
            last_payload = data

            status, _ = self._extract_status_and_path(data)
            self._log.debug("poll /clip/%s → status=%r payload=%r", job_id, status, data)

            if status in {"done", "error"}:
                return data

            time.sleep(interval_s)

    def _extract_status_and_path(self, payload: Mapping[str, Any]) -> tuple[str, Optional[str]]:
        """Best-effort extraction of clip status and path from a /clip payload.

        Supports both:
        - Top-level fields: {"status": "...", "path": "..."}
        - Nested fields:     {"ok": true, "clip": {"status": "...", "path": "..."}}

        If no explicit status is present but ``ok`` is true and a path is
        available, we treat the job as "done".
        """
        status: Optional[str] = None
        path_value: Optional[str] = None

        clip_obj = payload.get("clip")
        if isinstance(clip_obj, Mapping):
            if "status" in clip_obj:
                status = str(clip_obj.get("status"))
            if "path" in clip_obj and isinstance(clip_obj["path"], str):
                path_value = clip_obj["path"]

        # Fallback to top-level fields if present.
        if status is None and "status" in payload:
            val = payload.get("status")
            if val is not None:
                status = str(val)

        if path_value is None and isinstance(payload.get("path"), str):
            path_value = payload["path"]  # type: ignore[assignment]

        # If we still don't have a status, but the server says ok=true and we
        # have a path, assume the clip is done.
        if status is None and payload.get("ok") is True and path_value is not None:
            status = "done"

        return status or "", path_value


def recorder_config_from_cfg(cfg_module: Any) -> RecorderConfig:
    """Build :class:`RecorderConfig` from an application config module.

    The provided module is expected to define at least:

    - PRODUCER_BASE_URL
    - (optional) CLIP_DEFAULT_OUT_DIR
    - (optional) CLIP_DEFAULT_LABEL
    - (optional) CLIP_MAX_WAIT_MS
    - (optional) CLIP_POLL_INTERVAL_MS
    """
    out_dir = getattr(cfg_module, "CLIP_DEFAULT_OUT_DIR", None)
    path_out_dir = Path(out_dir) if out_dir else None

    return RecorderConfig(
        producer_base_url=str(cfg_module.PRODUCER_BASE_URL),
        default_out_dir=path_out_dir,
        default_label=getattr(cfg_module, "CLIP_DEFAULT_LABEL", None),
        max_wait_ms=int(getattr(cfg_module, "CLIP_MAX_WAIT_MS", 120_000)),
        poll_interval_ms=int(getattr(cfg_module, "CLIP_POLL_INTERVAL_MS", 500)),
    )
