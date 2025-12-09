from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from common.timebase import Timebase
from record.recorder import (
    ClipHttpError,
    ClipRequest,
    ClipRequestError,
    ClipResult,
    ClipTimeoutError,
    Recorder,
    RecorderConfig,
)


class _FakeResp:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class _FakeSession:
    def __init__(
        self, post_responses: Sequence[_FakeResp], get_responses: Sequence[_FakeResp]
    ) -> None:
        self._post_responses = list(post_responses)
        self._get_responses = list(get_responses)
        self.post_calls: list[tuple[str, Mapping[str, Any], float]] = []
        self.get_calls: list[tuple[str, float]] = []

    def post(self, url: str, json: Mapping[str, Any], timeout: float) -> _FakeResp:
        self.post_calls.append((url, json, timeout))
        if not self._post_responses:
            raise AssertionError("no more fake POST responses configured")
        return self._post_responses.pop(0)

    def get(self, url: str, timeout: float) -> _FakeResp:
        self.get_calls.append((url, timeout))
        if not self._get_responses:
            raise AssertionError("no more fake GET responses configured")
        return self._get_responses.pop(0)


def _basic_config() -> RecorderConfig:
    return RecorderConfig(
        producer_base_url="http://producer:8765",
        default_out_dir=Path("/clips"),
        default_label="test",
        max_wait_ms=5_000,
        poll_interval_ms=10,
    )


def test_request_clip_pts_mode_builds_payload_and_polls_to_done():
    cfg = _basic_config()
    session = _FakeSession(
        post_responses=[
            _FakeResp(
                status_code=200,
                json_data={"id": "job-123", "status": "pending"},
            )
        ],
        get_responses=[
            _FakeResp(status_code=200, json_data={"id": "job-123", "status": "pending"}),
            _FakeResp(
                status_code=200,
                json_data={
                    "id": "job-123",
                    "status": "done",
                    "path": "/clips/clip-00001.mp4",
                },
            ),
        ],
    )
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    req = ClipRequest(
        start_pts_ns=1000,
        stop_pts_ns=2000,
        preroll_ms=100,
        postroll_ms=200,
        filename="custom_name",
    )

    result = recorder.request_clip(req)

    # We should have built a PTS-mode payload.
    assert session.post_calls, "expected a POST /clip call"
    url, body, timeout = session.post_calls[0]
    assert url == "http://producer:8765/clip"
    assert body["start_pts_ns"] == 1000
    assert body["stop_pts_ns"] == 2000
    assert "start_epoch_ms" not in body
    assert "stop_epoch_ms" not in body
    assert body["preroll_ms"] == 100
    assert body["postroll_ms"] == 200
    assert body["filename"] == "custom_name"

    # Result should reflect the terminal status.
    assert isinstance(result, ClipResult)
    assert result.job_id == "job-123"
    assert result.status == "done"
    assert result.mode == "pts"
    assert result.path == Path("/clips/clip-00001.mp4")


def test_request_clip_epoch_only_with_timebase_uses_pts_mode():
    cfg = _basic_config()
    # Synthetic timebase where pts units are ns and epoch_ms = base + (pts - base)/1e6
    tb = Timebase(
        base_epoch_ms=1_700_000_000_000.0, base_pts_ns=10_000_000_000_000.0, pts_units="ns"
    )

    session = _FakeSession(
        post_responses=[_FakeResp(status_code=200, json_data={"id": "job-1", "status": "pending"})],
        get_responses=[
            _FakeResp(
                status_code=200, json_data={"id": "job-1", "status": "done", "path": "/clips/x.mp4"}
            )
        ],
    )
    recorder = Recorder(cfg=cfg, timebase=tb, session=session)

    req = ClipRequest(
        start_ms=1_700_000_000_500.0,
        stop_ms=1_700_000_001_000.0,
    )
    result = recorder.request_clip(req)

    assert result.mode == "pts"
    assert session.post_calls
    _, body, _ = session.post_calls[0]
    assert "start_pts_ns" in body and "stop_pts_ns" in body
    assert "start_epoch_ms" not in body and "stop_epoch_ms" not in body


def test_request_clip_epoch_only_without_timebase_uses_epoch_mode():
    cfg = _basic_config()
    session = _FakeSession(
        post_responses=[_FakeResp(status_code=200, json_data={"id": "job-2", "status": "pending"})],
        get_responses=[
            _FakeResp(
                status_code=200, json_data={"id": "job-2", "status": "done", "path": "/clips/y.mp4"}
            )
        ],
    )
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    req = ClipRequest(start_ms=1000.0, stop_ms=2000.0)
    result = recorder.request_clip(req)

    assert result.mode == "epoch"
    assert session.post_calls
    _, body, _ = session.post_calls[0]
    assert body["start_epoch_ms"] == 1000.0
    assert body["stop_epoch_ms"] == 2000.0
    assert "start_pts_ns" not in body and "stop_pts_ns" not in body


def test_request_clip_rejects_invalid_windows():
    cfg = _basic_config()
    session = _FakeSession(post_responses=[], get_responses=[])
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    # No window at all.
    with pytest.raises(ClipRequestError):
        recorder.request_clip(ClipRequest())

    # stop <= start in PTS.
    with pytest.raises(ClipRequestError):
        recorder.request_clip(ClipRequest(start_pts_ns=1000, stop_pts_ns=500))

    # stop <= start in epoch.
    with pytest.raises(ClipRequestError):
        recorder.request_clip(ClipRequest(start_ms=2000.0, stop_ms=1000.0))


def test_post_http_error_raises():
    cfg = _basic_config()
    session = _FakeSession(
        post_responses=[_FakeResp(status_code=500, json_data={"error": "boom"}, text="boom")],
        get_responses=[],
    )
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    with pytest.raises(ClipHttpError):
        recorder.request_clip(ClipRequest(start_pts_ns=0, stop_pts_ns=1))


def test_post_invalid_json_raises():
    cfg = _basic_config()
    session = _FakeSession(
        post_responses=[_FakeResp(status_code=200, json_data=ValueError("bad json"))],
        get_responses=[],
    )
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    with pytest.raises(ClipHttpError):
        recorder.request_clip(ClipRequest(start_pts_ns=0, stop_pts_ns=1))


def test_poll_timeout_raises():
    cfg = _basic_config()
    # Use a small timeout so the test remains fast.
    cfg.max_wait_ms = 10
    # GET always returns "pending", never "done" or "error".
    session = _FakeSession(
        post_responses=[_FakeResp(status_code=200, json_data={"id": "job-9", "status": "pending"})],
        get_responses=[_FakeResp(status_code=200, json_data={"id": "job-9", "status": "pending"})]
        * 100,
    )
    recorder = Recorder(cfg=cfg, timebase=None, session=session)

    with pytest.raises(ClipTimeoutError):
        recorder.request_clip(ClipRequest(start_pts_ns=0, stop_pts_ns=1))
