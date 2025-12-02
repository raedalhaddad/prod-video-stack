from __future__ import annotations

import math

from capture.stats_adapter import make_pts_mapper_from_health_payload


def _payload(base_epoch_ms: float, base_pts: float, units: str) -> dict:
    # Minimal /health-style payload for the timebase block
    return {
        "timebase": {
            "base_epoch_ms": base_epoch_ms,
            "base_pts_ns": base_pts,
            "pts_units": units,
        }
    }


def test_pts_mapper_ns_units():
    payload = _payload(base_epoch_ms=1000.0, base_pts=1_000_000_000, units="ns")
    mapper = make_pts_mapper_from_health_payload(payload)

    # At base_pts we should land exactly on base_epoch_ms
    assert mapper(1_000_000_000) == 1000.0

    # 0.5s later → 500 ms increment
    ms = mapper(1_000_000_000 + 500_000_000)
    assert math.isclose(ms, 1500.0, rel_tol=1e-9, abs_tol=1e-6)


def test_pts_mapper_us_units():
    payload = _payload(base_epoch_ms=2000.0, base_pts=1_000_000, units="us")
    mapper = make_pts_mapper_from_health_payload(payload)

    assert mapper(1_000_000) == 2000.0  # base
    # 0.5s later → 500 ms increment
    ms = mapper(1_000_000 + 500_000)
    assert math.isclose(ms, 2500.0, rel_tol=1e-9, abs_tol=1e-6)


def test_pts_mapper_ms_units():
    payload = _payload(base_epoch_ms=3000.0, base_pts=1_000, units="ms")
    mapper = make_pts_mapper_from_health_payload(payload)

    assert mapper(1_000) == 3000.0  # base
    ms = mapper(1_000 + 500)
    assert math.isclose(ms, 3500.0, rel_tol=1e-9, abs_tol=1e-6)


def test_pts_mapper_epoch_ms_units():
    # For epoch_ms we expect an identity mapping: raw pts are already epoch-ms
    payload = _payload(base_epoch_ms=0.0, base_pts=0.0, units="epoch_ms")
    mapper = make_pts_mapper_from_health_payload(payload)

    assert mapper(12345.0) == 12345.0


def test_pts_mapper_epoch_s_units():
    # For epoch_s we expect seconds → ms
    payload = _payload(base_epoch_ms=0.0, base_pts=0.0, units="epoch_s")
    mapper = make_pts_mapper_from_health_payload(payload)

    assert mapper(1.0) == 1000.0
    assert mapper(12.5) == 12_500.0


def test_pts_mapper_invalid_payload_identity_fallback():
    # Missing timebase → default identity mapper (age may be meaningless)
    mapper = make_pts_mapper_from_health_payload({})
    # Should behave like float(x)
    assert mapper(123.4) == 123.4
