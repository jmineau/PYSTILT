"""
The checks a receptor must pass, each written once for many points.

Each check returns ``(bad, message)``: the points that fail it and why. A
receptor runs the checks on its own points and raises the first failure
(:mod:`stilt.receptors.models`). The receptor table runs them on a whole
file and names the first bad row (:func:`stilt.receptors.receptor_rows`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

#: Why a multipoint receptor's points may not share a horizontal location.
REPEATED_LOCATION = (
    "MultiPointReceptor points must have distinct horizontal locations "
    "(HYSPLIT collapses starting locations that share a lat/lon into a "
    "single vertical line source and releases only between the last two "
    "heights). Use ColumnReceptor for a vertical column, or one "
    "PointReceptor per height (distinct r_idx) for discrete release "
    "heights at one location."
)


def point_errors(
    lon: Any, lat: Any, alt: Any, altitude_ref: Any
) -> list[tuple[np.ndarray, str]]:
    """
    Return the checks every release point must pass, as ``(bad, message)``.

    *lon*, *lat*, and *alt* hold one value per point; *altitude_ref* holds
    one per point or one for all.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    alt = np.asarray(alt, dtype=float)
    agl = np.asarray(altitude_ref) == "agl"
    return [
        ((lon < -180) | (lon > 180), "longitude must be within [-180, 180]."),
        ((lat < -90) | (lat > 90), "latitude must be within [-90, 90]."),
        (agl & (alt < 0), "AGL altitudes must be >= 0."),
    ]


def column_errors(bottom: Any, top: Any) -> list[tuple[np.ndarray, str]]:
    """Return the check a column must pass, as ``(bad, message)``: bottom below top."""
    bad = np.asarray(bottom, dtype=float) >= np.asarray(top, dtype=float)
    return [(bad, "'bottom' must be less than 'top'.")]


def multipoint_errors(
    lon: Any, lat: Any, receptor: Any
) -> list[tuple[np.ndarray, str]]:
    """
    Return the check a multipoint receptor must pass, as ``(bad, message)``.

    No two points of one receptor may share a horizontal location, to 5
    decimal places. *receptor* labels each point's receptor. A point is bad
    when an earlier point of its receptor is at the same location.

    HYSPLIT joins consecutive starting locations at one latitude and
    longitude into a vertical line source and releases only from the last
    pair, and PYSTILT matches particles to their release point by
    horizontal position.
    """
    seen: set[tuple[Any, float, float]] = set()
    bad = np.zeros(len(lon), dtype=bool)
    for i, key in enumerate(
        zip(
            np.asarray(receptor).tolist(),
            (round(float(v), 5) for v in lon),
            (round(float(v), 5) for v in lat),
            strict=True,
        )
    ):
        bad[i] = key in seen
        seen.add(key)
    return [(bad, REPEATED_LOCATION)]


def _raise_first(errors: Iterable[tuple[np.ndarray, str]]) -> None:
    """Raise the message of the first check any point fails."""
    for bad, message in errors:
        if bad.any():
            raise ValueError(message)
