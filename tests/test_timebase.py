from __future__ import annotations

import math

from capture.stats_adapter import make_pts_mapper_from_health_payload
from common.timebase import timebase_from_health_payload


def _almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_timebase_matches_pts_mapper_for_all_units():
    base_epoch_ms = 1_700_000_000_000.0
    base_pts = 10_000_000_000_000.0
    units_list = ["ns", "us", "ms", "epoch_ms", "epoch_s"]

    for units in units_list:
        payload = {
            "timebase": {
                "base_epoch_ms": base_epoch_ms,
                "base_pts_ns": base_pts,
                "pts_units": units,
            }
        }
        tb = timebase_from_health_payload(payload)
        mapper = make_pts_mapper_from_health_payload(payload)

        # A small set of sample PTS values around the anchor.
        for delta in (-3, -1, 0, 1, 10, 100):
            pts = base_pts + delta
            epoch_via_tb = tb.pts_to_epoch_ms(pts)
            epoch_via_mapper = mapper(pts)
            assert _almost_equal(
                epoch_via_tb,
                epoch_via_mapper,
            ), f"units={units} pts={pts} tb={epoch_via_tb} mapper={epoch_via_mapper}"

            # Round-trip via epoch_ms_to_pts should recover the original PTS.
            pts_roundtrip = tb.epoch_ms_to_pts(epoch_via_tb)
            assert _almost_equal(
                pts_roundtrip,
                float(pts),
            ), f"units={units} pts={pts} roundtrip={pts_roundtrip}"
