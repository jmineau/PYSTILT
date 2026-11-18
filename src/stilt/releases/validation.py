"""
The checks a release must pass, each written once for many points.

Each check returns ``(bad, message)``: the points that fail it and why. A
release runs the checks on its own points and raises the first failure
(:mod:`stilt.releases.models`). The release table runs them on a whole
file and names the first bad row (:func:`stilt.releases.table.release_rows`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

#: Why a multipoint release's points may not share a horizontal location.
REPEATED_LOCATION = (
    "MultiPointRelease points must have distinct horizontal locations "
    "(HYSPLIT collapses release locations that share a lat/lon into a "
    "single vertical line source and releases only between the last two "
    "heights). Use ColumnRelease for a vertical column, or one "
    "PointRelease per height (distinct group) for discrete release "
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
    # Written as "not inside", so a NaN, which compares False, fails too.
    return [
        (
            ~((lon >= -180) & (lon <= 180)),
            "longitude must be a number within [-180, 180].",
        ),
        (~((lat >= -90) & (lat <= 90)), "latitude must be a number within [-90, 90]."),
        (~np.isfinite(alt), "altitude must be a finite number."),
        (agl & (alt < 0), "AGL altitudes must be >= 0."),
    ]


def column_errors(bottom: Any, top: Any) -> list[tuple[np.ndarray, str]]:
    """Return the check a column must pass, as ``(bad, message)``: bottom below top."""
    bad = np.asarray(bottom, dtype=float) >= np.asarray(top, dtype=float)
    return [(bad, "'bottom' must be less than 'top'.")]


def multipoint_errors(lon: Any, lat: Any, release: Any) -> list[tuple[np.ndarray, str]]:
    """
    Return the check a multipoint release must pass, as ``(bad, message)``.

    No two points of one release may share a horizontal location, to 5
    decimal places. *release* labels each point's release. A point is bad
    when an earlier point of its release is at the same location.

    HYSPLIT joins consecutive release locations at one latitude and
    longitude into a vertical line source and releases only from the last
    pair, and PYSTILT matches particles to their release point by
    horizontal position.
    """
    points = pd.DataFrame(
        {
            "release": np.asarray(release),
            "lon": np.round(np.asarray(lon, dtype=float), 5),
            "lat": np.round(np.asarray(lat, dtype=float), 5),
        }
    )
    return [(points.duplicated().to_numpy(), REPEATED_LOCATION)]


def _raise_first(errors: Iterable[tuple[np.ndarray, str]]) -> None:
    """Raise the message of the first check any point fails."""
    for bad, message in errors:
        if bad.any():
            raise ValueError(message)
