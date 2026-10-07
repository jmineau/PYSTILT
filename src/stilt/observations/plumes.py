"""
Background from the soundings outside a plume traced by forward runs.

Forward runs released from a city over the hours before a satellite overpass
show where the city's air is at overpass time. A kernel density of the
particle positions during the overpass, contoured at a fraction of its
maximum, outlines the plume. Soundings outside the outline saw background
air. X-STILT does this in ``fit.kde.plume`` and ``calc.bg.upwind``
(Wu et al. 2018, method M3).

:func:`plume_polygon` draws the outline and :func:`plume_background` picks
the background soundings and summarizes them. Both take plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from numpy.typing import ArrayLike
from shapely.geometry import MultiPolygon, Polygon

Side = Literal["north", "south", "east", "west"]
SIDES: tuple[Side, ...] = ("north", "south", "east", "west")

#: Points per chunk when accumulating the kernel density.
_KDE_CHUNK = 50_000


@dataclass(frozen=True)
class Plume:
    """
    Plume outline returned by :func:`plume_polygon`.

    Attributes
    ----------
    polygon : shapely.Polygon
        Plume outline, in longitude and latitude.
    density : xarray.DataArray
        Kernel density of the particle positions on ``lat`` and ``lon``,
        scaled to a maximum of 1.
    threshold : float
        Scaled density that the outline follows.
    """

    polygon: Polygon
    density: xr.DataArray
    threshold: float


def kernel_density(
    longitudes: ArrayLike,
    latitudes: ArrayLike,
    *,
    bandwidth: tuple[float, float] = (0.1, 0.15),
    n: int = 100,
) -> xr.DataArray:
    """
    Return the Gaussian kernel density of points on an ``n`` by ``n`` grid.

    As in R's ``MASS::kde2d``, the kernel's standard deviation is a quarter
    of ``bandwidth`` in each direction, so the defaults give 0.025° in
    longitude and 0.0375° in latitude. The grid spans the points plus one
    bandwidth on every side. The density is scaled to a maximum of 1.

    Parameters
    ----------
    longitudes, latitudes : array-like
        Point positions, in degrees. Non-finite positions are dropped.
    bandwidth : tuple of float, default (0.1, 0.15)
        Bandwidth in longitude and latitude, in degrees.
    n : int, default 100
        Number of grid points in each direction.

    Returns
    -------
    xarray.DataArray
        Density on ``lat`` and ``lon``.
    """
    lon = np.asarray(longitudes, dtype=float).ravel()
    lat = np.asarray(latitudes, dtype=float).ravel()
    if lon.shape != lat.shape:
        raise ValueError("longitudes and latitudes must have the same length")
    keep = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[keep], lat[keep]
    if lon.size == 0:
        raise ValueError("no finite particle positions")
    if n < 2:
        raise ValueError("n must be >= 2")
    bw_x, bw_y = (float(b) for b in bandwidth)
    if bw_x <= 0 or bw_y <= 0:
        raise ValueError("bandwidth must be > 0")
    sd_x, sd_y = bw_x / 4, bw_y / 4

    gx = np.linspace(lon.min() - bw_x, lon.max() + bw_x, n)
    gy = np.linspace(lat.min() - bw_y, lat.max() + bw_y, n)
    z = np.zeros((n, n))  # (lat, lon)
    for start in range(0, lon.size, _KDE_CHUNK):
        sl = slice(start, start + _KDE_CHUNK)
        kx = np.exp(-0.5 * ((gx[:, None] - lon[None, sl]) / sd_x) ** 2)
        ky = np.exp(-0.5 * ((gy[:, None] - lat[None, sl]) / sd_y) ** 2)
        z += ky @ kx.T
    z /= z.max()
    return xr.DataArray(
        z,
        dims=("lat", "lon"),
        coords={"lat": gy, "lon": gx},
        name="density",
        attrs={"bandwidth_lon": bw_x, "bandwidth_lat": bw_y},
    )


def density_polygon(density: xr.DataArray, threshold: float) -> Polygon:
    """
    Return the largest connected region where ``density >= threshold``.

    Grid cells at or above the threshold are merged. When they form several
    separate pieces, the largest is the plume, as X-STILT keeps the longest
    contour piece. The outline follows cell edges, so it is as fine as the
    density grid.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    z = density.transpose("lat", "lon").to_numpy()
    gx = np.asarray(density["lon"].to_numpy(), dtype=float)
    gy = np.asarray(density["lat"].to_numpy(), dtype=float)
    dx = float(gx[1] - gx[0]) if gx.size > 1 else 0.0
    dy = float(gy[1] - gy[0]) if gy.size > 1 else 0.0
    iy, ix = np.nonzero(z >= threshold)
    if ix.size == 0:
        raise ValueError(f"no density at or above threshold {threshold}")
    # Cell edges from one origin, so neighboring cells share exact coordinates
    # and merge cleanly.
    x_edges = gx[0] - dx / 2 + np.arange(gx.size + 1) * dx
    y_edges = gy[0] - dy / 2 + np.arange(gy.size + 1) * dy
    cells = shapely.box(x_edges[ix], y_edges[iy], x_edges[ix + 1], y_edges[iy + 1])
    merged = shapely.unary_union(cells)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    if not isinstance(merged, Polygon):
        raise ValueError("the thresholded density does not form a polygon")
    return merged


def plume_polygon(
    longitudes: ArrayLike,
    latitudes: ArrayLike,
    *,
    threshold: float = 0.1,
    bandwidth: tuple[float, float] = (0.1, 0.15),
    n: int = 100,
) -> Plume:
    """
    Return the plume outline from forward-run particle positions at overpass time.

    The plume is the region where the kernel density of the positions
    (:func:`kernel_density`) is at least ``threshold`` times its maximum.
    The defaults are X-STILT's (``td = 0.1``, ``h = c(0.1, 0.15)``,
    ``n = 100``).

    Parameters
    ----------
    longitudes, latitudes : array-like
        Positions of every particle row in the overpass window, pooled over
        all the forward runs for that overpass, in degrees.
    threshold : float, default 0.1
        Fraction of the maximum density that the outline follows. A larger
        value gives a tighter plume.
    bandwidth : tuple of float, default (0.1, 0.15)
        Kernel bandwidth in longitude and latitude, in degrees. A larger
        value gives a smoother plume.
    n : int, default 100
        Number of density grid points in each direction.

    Returns
    -------
    Plume
    """
    density = kernel_density(longitudes, latitudes, bandwidth=bandwidth, n=n)
    polygon = density_polygon(density, threshold)
    return Plume(polygon=polygon, density=density, threshold=float(threshold))


@dataclass(frozen=True)
class PlumeBackground:
    """
    Background from soundings beside a plume, returned by :func:`plume_background`.

    Attributes
    ----------
    value : float
        Background, the median of the soundings in ``used``.
    uncertainty : float
        Standard deviation of those soundings and their RMS retrieval
        uncertainty, added in quadrature. A part that is unknown (one
        sounding, or no retrieval uncertainties given) counts as zero.
    n : int
        Number of soundings in ``used``.
    in_plume : numpy.ndarray
        True for each sounding inside the plume.
    used : numpy.ndarray
        True for each sounding the background came from.
    sides : pandas.DataFrame
        The same summary for each side of the plume, one row per side
        (``north``, ``south``, ``east``, ``west``), with columns ``n``,
        ``mean``, ``median``, ``std``, ``retrieval_std``, and
        ``uncertainty``. Use it to check whether the sides agree before
        picking one with ``side=``.
    """

    value: float
    uncertainty: float
    n: int
    in_plume: np.ndarray
    used: np.ndarray
    sides: pd.DataFrame


def _stats(values: np.ndarray, uncertainties: np.ndarray) -> dict[str, float]:
    """Return the count, mean, median, spread, and uncertainty of some soundings."""
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "retrieval_std": np.nan,
            "uncertainty": np.nan,
        }
    std = float(np.std(values, ddof=1)) if n > 1 else np.nan
    retrieval = float(np.sqrt(np.mean(uncertainties**2)))
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": std,
        "retrieval_std": retrieval,
        "uncertainty": float(
            np.sqrt(np.nan_to_num(std) ** 2 + np.nan_to_num(retrieval) ** 2)
        ),
    }


def plume_background(
    longitudes: ArrayLike,
    latitudes: ArrayLike,
    values: ArrayLike,
    plume: Plume | Polygon,
    *,
    uncertainties: ArrayLike | None = None,
    side: Side | None = None,
    width: float = 0.5,
    pad: float = 0.1,
    trim: float | None = 0.9,
) -> PlumeBackground:
    """
    Return the background from the soundings beside a plume.

    The soundings inside ``plume`` are enhanced. A box is drawn around them,
    and the soundings outside the plume within ``width`` degrees of the box
    on each side are the background candidates. The background is their
    median. Pass only good-quality soundings. Raises when no sounding is
    inside the plume, since then the overpass did not see it.

    Parameters
    ----------
    longitudes, latitudes : array-like
        Sounding positions, in degrees.
    values : array-like
        Sounding values, such as XCH4.
    plume : Plume or shapely.Polygon
        Plume outline, usually from :func:`plume_polygon`.
    uncertainties : array-like, optional
        Retrieval uncertainty of each sounding, in the units of ``values``.
    side : {"north", "south", "east", "west"}, optional
        Use only one side. Choose the upwind side, which the forward
        particles' drift shows. ``None`` pools all four.
    width : float, default 0.5
        Width of the band on each side of the box, in degrees.
    pad : float, default 0.1
        Padding of the box around the in-plume soundings, as a fraction of
        its size.
    trim : float or None, default 0.9
        Drop soundings outside the plume above this quantile of their
        values, to guard against enhanced air the outline missed, as
        X-STILT does. ``None`` keeps them all.

    Returns
    -------
    PlumeBackground
    """
    lon = np.asarray(longitudes, dtype=float).ravel()
    lat = np.asarray(latitudes, dtype=float).ravel()
    val = np.asarray(values, dtype=float).ravel()
    if not lon.shape == lat.shape == val.shape:
        raise ValueError("longitudes, latitudes and values must have the same length")
    if uncertainties is None:
        unc = np.full(val.shape, np.nan)
    else:
        unc = np.asarray(uncertainties, dtype=float).ravel()
        if unc.shape != val.shape:
            raise ValueError("uncertainties must have the same length as values")
    if width <= 0:
        raise ValueError("width must be > 0")
    if pad < 0:
        raise ValueError("pad must be >= 0")
    if trim is not None and not 0 < trim <= 1:
        raise ValueError("trim must be in (0, 1]")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    polygon = plume.polygon if isinstance(plume, Plume) else plume
    in_plume = shapely.contains_xy(polygon, lon, lat) & np.isfinite(val)
    if not in_plume.any():
        raise ValueError("no sounding falls inside the plume")

    outside = ~in_plume & np.isfinite(val)
    if trim is not None and outside.any():
        edge = np.quantile(val[outside], trim)
        outside &= val <= edge

    # The padded box of the soundings the plume covers.
    xmin, xmax = lon[in_plume].min(), lon[in_plume].max()
    ymin, ymax = lat[in_plume].min(), lat[in_plume].max()
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    xmin, xmax, ymin, ymax = xmin - dx, xmax + dx, ymin - dy, ymax + dy

    in_x = (lon >= xmin) & (lon <= xmax)
    in_y = (lat >= ymin) & (lat <= ymax)
    masks: dict[Side, np.ndarray] = {
        "north": outside & in_x & (lat >= ymax) & (lat <= ymax + width),
        "south": outside & in_x & (lat <= ymin) & (lat >= ymin - width),
        "east": outside & in_y & (lon >= xmax) & (lon <= xmax + width),
        "west": outside & in_y & (lon <= xmin) & (lon >= xmin - width),
    }
    sides = pd.DataFrame(
        [_stats(val[m], unc[m]) for m in masks.values()],
        index=pd.Index(SIDES, name="side"),
    )
    sides["n"] = sides["n"].astype(int)

    used = (
        masks[side] if side is not None else np.logical_or.reduce(list(masks.values()))
    )
    stats = _stats(val[used], unc[used])
    return PlumeBackground(
        value=stats["median"],
        uncertainty=stats["uncertainty"],
        n=int(stats["n"]),
        in_plume=in_plume,
        used=used,
        sides=sides,
    )


__all__ = [
    "Plume",
    "PlumeBackground",
    "density_polygon",
    "kernel_density",
    "plume_background",
    "plume_polygon",
]
