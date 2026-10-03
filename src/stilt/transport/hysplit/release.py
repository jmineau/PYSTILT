"""
Release heights of HYSPLIT's particles.

HYSPLIT does not write the height a particle was released at, so PYSTILT
recovers it as ``xhgt`` from how HYSPLIT releases them: a column releases
its particles bottom to top in ``indx`` order, and a multipoint receptor's
particles are matched to its release points.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stilt.receptors import ColumnReceptor, MultiPointReceptor, Receptor

_MIN_RELIABLE_SPACING_M = 1000.0


def add_release_heights(particles: pd.DataFrame, receptor: Receptor) -> pd.DataFrame:
    """
    Return *particles* with each particle's release height ``xhgt``, for a column or multipoint receptor.

    A point receptor's particles are returned as they are.
    """
    if isinstance(receptor, ColumnReceptor):
        numpar = int(particles["indx"].max())  # type: ignore[arg-type]
        step = (receptor.top - receptor.bottom) / numpar
        return particles.assign(xhgt=(particles["indx"] - 0.5) * step + receptor.bottom)
    if isinstance(receptor, MultiPointReceptor):
        return particles.assign(xhgt=_multipoint_release_heights(particles, receptor))
    return particles


def _multipoint_release_heights(
    p: pd.DataFrame, receptor: MultiPointReceptor
) -> pd.Series:
    """
    Return each row's release altitude for a multipoint receptor.

    HYSPLIT does not record which starting location a particle came from.
    It is recovered from each particle's row nearest the release time:

    1. If the HYSPLIT build writes release-time (``t = 0``) rows, match the
       nearest release point horizontally. Nothing has moved yet, so this
       is exact.
    2. Otherwise the first row is one time step after release. Height
       drifts about 30 times less than horizontal position over that step,
       so when all release heights differ (as in a slanted column), match
       on height.
    3. Otherwise match on horizontal position, and warn when the release
       points are too close together for that to be reliable.
    """
    first = (
        p.assign(_age=p["time"].abs())
        .sort_values("_age", kind="stable")
        .drop_duplicates(subset="indx")
    )
    lons = np.asarray(receptor.longitudes, dtype=float)
    lats = np.asarray(receptor.latitudes, dtype=float)
    alts = np.asarray(receptor.altitudes, dtype=float)
    has_t0 = bool((first["_age"] == 0).all())

    # Height of each particle in the receptor's own vertical reference.
    height = None
    if "zagl" in first.columns:
        if receptor.altitude_ref == "agl":
            height = first["zagl"].to_numpy(dtype=float)
        elif "zsfc" in first.columns:
            height = (first["zagl"] + first["zsfc"]).to_numpy(dtype=float)

    if not has_t0 and height is not None and len(np.unique(alts)) == len(alts):
        nearest = np.argmin(np.abs(height[:, None] - alts[None, :]), axis=1)
    else:
        xy = first[["long", "lati"]].to_numpy(dtype=float)
        pts = np.column_stack((lons, lats))
        nearest = np.argmin(
            np.sum((xy[:, None, :] - pts[None, :, :]) ** 2, axis=2), axis=1
        )
        if not has_t0 and len(alts) > 1:
            x = lons * np.cos(np.radians(lats.mean())) * 111_320.0
            y = lats * 111_320.0
            gaps = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
            spacing = float(gaps[np.triu_indices(len(alts), k=1)].min())
            if spacing < _MIN_RELIABLE_SPACING_M:
                warnings.warn(
                    f"MultiPointReceptor release points are as close as "
                    f"{spacing:.0f} m and cannot be separated by altitude, and this "
                    "HYSPLIT build writes no t=0 row, so particles cannot be "
                    "reliably matched to their release points; 'xhgt' may be wrong. "
                    "Use a HYSPLIT build that writes release-time rows "
                    "(TransportParams.exe_dir) or space the points more than "
                    f"{_MIN_RELIABLE_SPACING_M:.0f} m apart.",
                    stacklevel=3,
                )

    mapping = dict(zip(first["indx"].to_numpy(), alts[nearest], strict=True))
    return pd.Series(p["indx"].to_numpy(), index=p.index).map(mapping.get)


__all__ = ["add_release_heights"]
