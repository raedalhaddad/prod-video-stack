from __future__ import annotations

from capture.reader import ShmParams, ShmTransport


def make_fake_ring(frames=5, width=8, height=4):
    # Build synthetic BGRx slots with increasing pts and ids
    frame_bytes = bytearray(height * width * 4)

    def fill(fid):
        for i in range(0, len(frame_bytes), 4):
            frame_bytes[i : i + 4] = bytes([(fid % 255), 0, 0, 0])

    state = {"fid": 0, "last_age": 0.0}

    def reader():
        if state["fid"] >= frames:
            return None
        fid = state["fid"]
        fill(fid)
        pts_ns = 1_000_000_000 + fid * 33_000_000  # ~30 fps
        state["fid"] += 1
        return memoryview(frame_bytes), pts_ns, fid, state["last_age"]

    return reader


def test_bgrx_to_bgr_and_pts_mapping():
    params = ShmParams(
        map_name="Local\\x",
        width=8,
        height=4,
        stride=8 * 4,
        fmt="BGRx",
        ring_slots=3,
        base_epoch_ms=1000.0,
        base_pts_ns=1_000_000_000,
    )
    tr = ShmTransport(
        params, make_fake_ring(frames=2, width=8, height=4), queue_max=2, drop_policy="drop_old"
    )
    tr.start()
    out = []
    while True:
        item = tr.read()
        if item is None:
            # allow ingest thread to run
            for _ in range(1000):
                item = tr.read()
                if item is not None:
                    break
        if item is None:
            break
        frame, pts_ms, fid = item
        out.append((frame.copy(), pts_ms, fid))
    tr.close()
    assert len(out) >= 1
    f0, t0, id0 = out[0]
    assert f0.shape == (4, 8, 3)
    # All blue channel equal to id0 (as we filled fid into B plane)
    assert int(f0[0, 0, 0]) == (id0 % 255)
    # PTS mapping monotonic
    ts = [t for _, t, _ in out]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_drop_policy():
    params = ShmParams(
        map_name="Local\\x",
        width=4,
        height=2,
        stride=16,
        fmt="BGRx",
        ring_slots=3,
        base_epoch_ms=0.0,
        base_pts_ns=0,
    )
    tr = ShmTransport(
        params, make_fake_ring(frames=10, width=4, height=2), queue_max=1, drop_policy="drop_new"
    )
    tr.start()
    # Drain a few reads, with queue_max=1 we expect drops when ingest outruns reads
    n = 0
    while True:
        item = tr.read()
        if item is None:
            for _ in range(1000):
                item = tr.read()
                if item is not None:
                    break
        if item is None:
            break
        n += 1
    tr.close()
    assert tr.stats().drops >= 1
