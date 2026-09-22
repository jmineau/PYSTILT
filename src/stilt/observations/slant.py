"""Slant line-of-sight geometry."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

_EARTH_RADIUS_M = 6_371_000.0


def slant_points(
    longitude: float,
    latitude: float,
    altitudes: ArrayLike,
    *,
    zenith: float,
    azimuth: float,
    anchor: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    ``(longitude, latitude, altitude)`` points along a line of sight.

    Each altitude sits ``(altitude - anchor) * tan(zenith)`` metres from
    ``(longitude, latitude)`` along the ``azimuth`` bearing, on a local flat
    tangent plane. ``zenith`` is degrees from the local vertical and
    ``azimuth`` is degrees clockwise from north: the bearing from the ground
    point toward the instrument or the sun, which is the direction the path
    rises toward. ``anchor`` is the altitude at which the path passes through
    ``(longitude, latitude)`` and defaults to the first altitude.

    Altitudes are returned unchanged, in whatever datum they were given. Use
    mean-sea-level altitudes for a slant; terrain-following (AGL) heights
    would bend the path. Pass the result to :meth:`stilt.Receptor.from_points`.
    """
    alts = np.asarray(altitudes, dtype=float).ravel()
    if alts.size == 0:
        raise ValueError("slant_points requires at least one altitude.")
    if not 0 <= zenith < 90:
        raise ValueError("slant_points zenith must be in [0, 90) degrees.")
    if anchor is None:
        anchor = float(alts[0])

    horizontal_m = (alts - anchor) * math.tan(math.radians(zenith))
    deg_per_m_lat = 1.0 / (math.radians(1.0) * _EARTH_RADIUS_M)
    deg_per_m_lon = deg_per_m_lat / math.cos(math.radians(latitude))
    lons = longitude + horizontal_m * math.sin(math.radians(azimuth)) * deg_per_m_lon
    lats = latitude + horizontal_m * math.cos(math.radians(azimuth)) * deg_per_m_lat
    return list(zip(lons.tolist(), lats.tolist(), alts.tolist(), strict=True))


__all__ = ["slant_points"]
