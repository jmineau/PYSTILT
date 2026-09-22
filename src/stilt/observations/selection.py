"""Overpass grouping, sounding selection, and pixel jitter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from random import Random
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from shapely.geometry import Point, Polygon

JitterMethod = Literal["regular", "random"]


# -- overpasses -----------------------------------------------------------------


def group_by_overpass(
    times: ArrayLike | pd.Series, *, max_gap: str | pd.Timedelta = "30min"
) -> pd.Series:
    """
    Label each time with the overpass it belongs to.

    This is the X-STILT overpass finder: soundings from one satellite pass
    are seconds apart and passes are hours apart, so a new group starts
    wherever consecutive times (in time order) differ by more than
    ``max_gap``. Each label is the group's first time as ``YYYYMMDDHHMM``.

    The result is aligned with the input, so on a table of soundings::

        df["overpass"] = group_by_overpass(df["time"])
        for label, scene in df.groupby("overpass"):
            ...
    """
    series = times if isinstance(times, pd.Series) else pd.Series(times)
    stamps = pd.to_datetime(series)
    if stamps.isna().any():
        raise ValueError("group_by_overpass: times contain NaT.")
    labels = pd.Series(np.empty(len(stamps), dtype=object), index=stamps.index)
    if stamps.empty:
        return labels

    values = stamps.to_numpy()
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    gap = np.timedelta64(pd.Timedelta(max_gap).value, "ns")
    starts = np.concatenate(([True], np.diff(ordered) > gap))
    group = np.cumsum(starts) - 1
    names = pd.DatetimeIndex(ordered[starts]).strftime("%Y%m%d%H%M").to_numpy()
    labels.iloc[order] = names[group]
    return labels


# -- spatial selection ------------------------------------------------------------


def _haversine_km(
    lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres from one point to many."""
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Return *n* evenly-spaced values from *start* to *stop* inclusive."""
    if n <= 0:
        return []
    if n == 1:
        return [(start + stop) / 2]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def select_observations_spatial(
    longitudes: ArrayLike,
    latitudes: ArrayLike,
    *,
    site_longitude: float,
    site_latitude: float,
    near_field_dlon: float,
    near_field_dlat: float,
    near_field_cols: int,
    near_field_rows: int,
    background_cols: int,
    background_rows: int,
    domain_lon_range: tuple[float, float],
    domain_lat_range: tuple[float, float],
) -> np.ndarray:
    """
    Select soundings on a near-field plus background grid.

    Ports X-STILT's ``sel.obs4recpv2``: lay a dense grid of
    ``near_field_cols × near_field_rows`` points over
    ``site ± near_field_dlon/dlat`` and a sparse grid of
    ``background_cols × background_rows`` points over the full domain, and
    keep the sounding nearest (great-circle) to each grid point. Dense
    coverage near the site is where the footprints matter most; the
    background soundings support a concentration-difference analysis.

    Returns the positional indices of the selected soundings, each once, in
    ascending latitude. Use them as ``df.iloc[selected]``.
    """
    lons = np.asarray(longitudes, dtype=float).ravel()
    lats = np.asarray(latitudes, dtype=float).ravel()
    if lons.shape != lats.shape:
        raise ValueError("longitudes and latitudes must have the same length.")
    if lons.size == 0:
        return np.array([], dtype=int)

    nf_lons = _linspace(
        site_longitude - near_field_dlon,
        site_longitude + near_field_dlon,
        near_field_cols,
    )
    nf_lats = _linspace(
        site_latitude - near_field_dlat,
        site_latitude + near_field_dlat,
        near_field_rows,
    )
    bg_lons = _linspace(domain_lon_range[0], domain_lon_range[1], background_cols)
    bg_lats = _linspace(domain_lat_range[0], domain_lat_range[1], background_rows)
    grid_points = [(lon, lat) for lat in nf_lats for lon in nf_lons] + [
        (lon, lat) for lat in bg_lats for lon in bg_lons
    ]

    selected = {
        int(np.argmin(_haversine_km(g_lon, g_lat, lons, lats)))
        for g_lon, g_lat in grid_points
    }
    return np.array(sorted(selected, key=lambda i: (lats[i], i)), dtype=int)


# -- jitter ------------------------------------------------------------------------


def _sample_regular(polygon: Polygon, n: int) -> list[tuple[float, float]]:
    """Sample approximately regular points inside a polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    if polygon.area <= 0:
        raise ValueError("Cannot jitter within a zero-area polygon.")
    resolution = max(2, math.ceil(math.sqrt(n)))
    for _ in range(12):
        xs = [minx + (i + 0.5) * (maxx - minx) / resolution for i in range(resolution)]
        ys = [miny + (j + 0.5) * (maxy - miny) / resolution for j in range(resolution)]
        points = [(x, y) for y in ys for x in xs if polygon.covers(Point(x, y))]
        if len(points) >= n:
            return points[:n]
        resolution += 1
    raise ValueError("Unable to generate enough regular jitter points inside polygon.")


def _sample_random(
    polygon: Polygon, n: int, *, seed: int | None = None
) -> list[tuple[float, float]]:
    """Sample random points inside a polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    rng = Random(seed)
    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(100, n * 200)
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if polygon.covers(Point(x, y)):
            points.append((x, y))
    if len(points) < n:
        raise ValueError(
            "Unable to generate enough random jitter points inside polygon."
        )
    return points


def jitter_points(
    polygon: Polygon | Sequence[tuple[float, float]],
    n: int,
    *,
    method: JitterMethod = "regular",
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """
    ``(longitude, latitude)`` points spread over one pixel.

    X-STILT's ``jitterTF``: instead of one receptor at a large pixel's
    centre, run several across it and average their footprints. ``polygon``
    is the pixel outline, as a shapely polygon or its corner coordinates.
    ``"regular"`` lays the points on a grid clipped to the polygon;
    ``"random"`` draws them uniformly with ``seed``.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    outline = polygon if isinstance(polygon, Polygon) else Polygon(polygon)
    if method == "regular":
        return _sample_regular(outline, n)
    if method == "random":
        return _sample_random(outline, n, seed=seed)
    raise ValueError(f"Unknown jitter method: {method!r}")


__all__ = ["group_by_overpass", "jitter_points", "select_observations_spatial"]
