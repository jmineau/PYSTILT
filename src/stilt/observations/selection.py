"""Overpass grouping and sounding selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# -- overpasses -----------------------------------------------------------------


def group_by_overpass(
    times: ArrayLike | pd.Series, *, max_gap: str | pd.Timedelta = "30min"
) -> pd.Series:
    """
    Label each time with the overpass it belongs to.

    Soundings from one satellite pass are seconds apart and passes are hours
    apart, so a new overpass starts wherever sorted times differ by more
    than ``max_gap``. This is X-STILT's overpass grouping.

    Parameters
    ----------
    times : array-like or pandas.Series
        Sounding times.
    max_gap : str or pandas.Timedelta, default "30min"
        Largest gap between soundings of one overpass.

    Returns
    -------
    pandas.Series
        The first time of each sounding's overpass, as ``YYYYMMDDHHMM``,
        aligned with ``times``.

    Examples
    --------
    >>> df["overpass"] = group_by_overpass(df["time"])
    >>> for label, scene in df.groupby("overpass"):
    ...     ...
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


def haversine_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Return the great-circle distance in km, broadcasting over the inputs."""
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Return ``n`` evenly spaced values from ``start`` to ``stop``, or their midpoint when ``n`` is 1."""
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
    Select soundings densely near a site and sparsely across the domain.

    Lays a dense grid of points around the site and a sparse grid over the
    whole domain, and keeps the sounding nearest to each grid point. The
    dense soundings cover the site, where footprints matter most, and the
    sparse ones give a background. This is X-STILT's ``sel.obs4recpv2``.

    Parameters
    ----------
    longitudes, latitudes : array-like
        Sounding positions, in degrees.
    site_longitude, site_latitude : float
        Center of the dense grid, in degrees.
    near_field_dlon, near_field_dlat : float
        Half-width and half-height of the dense grid, in degrees.
    near_field_cols, near_field_rows : int
        Number of dense grid points across and up.
    background_cols, background_rows : int
        Number of sparse grid points across and up.
    domain_lon_range, domain_lat_range : tuple of float
        ``(min, max)`` extent of the sparse grid, in degrees.

    Returns
    -------
    numpy.ndarray
        Positions of the selected soundings, each once, in order of
        latitude. Use them as ``df.iloc[selected]``.
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
        int(np.argmin(haversine_km(g_lon, g_lat, lons, lats)))
        for g_lon, g_lat in grid_points
    }
    return np.array(sorted(selected, key=lambda i: (lats[i], i)), dtype=int)


__all__ = ["group_by_overpass", "select_observations_spatial"]
