from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Timebase:
    """Anchored mapping between producer PTS and epoch milliseconds.

    The producer exposes a single anchored timebase via /health.timebase:

    - base_epoch_ms: epoch milliseconds at the anchor
    - base_pts_ns:   producer PTS at the anchor (units described by pts_units)
    - pts_units:     one of {"ns", "us", "ms", "epoch_ms", "epoch_s"}

    This helper provides symmetric conversions between the producer's PTS
    timeline and epoch milliseconds, matching the semantics used by
    capture.stats_adapter._build_pts_mapper.
    """

    base_epoch_ms: float
    base_pts_ns: float
    pts_units: str  # e.g. "ns", "us", "ms", "epoch_ms", "epoch_s"

    # --- public mapping helpers -------------------------------------------------

    def pts_to_epoch_ms(self, pts: float) -> float:
        """Map a PTS value from the producer's PTS timeline into epoch-ms.

        Parameters
        ----------
        pts:
            PTS value in the units indicated by ``pts_units``. For example, when
            pts_units == "ns", this is nanoseconds on the producer's PTS clock.

        Returns
        -------
        float
            Epoch milliseconds corresponding to the given PTS value.
        """
        units = self.pts_units
        # Map the known units into a scale that converts "1 PTS unit" into ms.
        unit_scale_ms = {
            "ns": 1e-6,
            "us": 1e-3,
            "ms": 1.0,
            "epoch_ms": None,
            "epoch_s": "sec",
        }.get(units, 1e-6)

        if unit_scale_ms is None:
            # PTS is already in epoch-ms.
            return float(pts)
        if unit_scale_ms == "sec":
            # PTS is epoch seconds.
            return float(pts) * 1000.0

        # Generic base-epoch + scaled offset mapping.
        sc = float(unit_scale_ms)
        return self.base_epoch_ms + (float(pts) - self.base_pts_ns) * sc

    def epoch_ms_to_pts(self, epoch_ms: float) -> float:
        """Inverse of :meth:`pts_to_epoch_ms`.

        Given an epoch-ms timestamp, compute the corresponding PTS value in the
        producer's internal units (as described by ``pts_units``).
        """
        units = self.pts_units
        # Map the known units into a scale that converts "1 PTS unit" into ms.
        unit_scale_ms = {
            "ns": 1e-6,
            "us": 1e-3,
            "ms": 1.0,
            "epoch_ms": None,
            "epoch_s": "sec",
        }.get(units, 1e-6)

        if unit_scale_ms is None:
            # PTS is already in epoch-ms.
            return float(epoch_ms)
        if unit_scale_ms == "sec":
            # PTS is epoch seconds.
            return float(epoch_ms) / 1000.0

        sc = float(unit_scale_ms)
        # epoch_ms = base_epoch_ms + (pts - base_pts_ns) * sc
        # => pts = base_pts_ns + (epoch_ms - base_epoch_ms) / sc
        return self.base_pts_ns + (float(epoch_ms) - self.base_epoch_ms) / sc


# --- construction helpers -------------------------------------------------------


def _extract_timebase(data: Mapping[str, Any]) -> tuple[float, float, str]:
    """Extract (base_epoch_ms, base_pts, units) from a /health-style payload.

    Accepts either the full /health document or just its ``"timebase"`` sub-dict.
    This mirrors the helper used in ``capture.stats_adapter`` so that all
    components interpret the producer's timebase in a consistent way.
    """
    tb = data.get("timebase") or data
    base_epoch_ms = float(tb["base_epoch_ms"])
    base_pts = float(tb.get("base_pts_ns", tb.get("base_pts")))
    units = str(tb.get("pts_units", "ns")).lower()
    return base_epoch_ms, base_pts, units


def timebase_from_health_payload(data: Mapping[str, Any]) -> Timebase:
    """Build a :class:`Timebase` from an in-memory /health payload.

    Parameters
    ----------
    data:
        Either the full /health JSON object or its ``"timebase"`` sub-dict.

    Raises
    ------
    KeyError, ValueError
        If required fields are missing or malformed.
    """
    base_epoch_ms, base_pts, units = _extract_timebase(data)
    return Timebase(base_epoch_ms=base_epoch_ms, base_pts_ns=base_pts, pts_units=units)
