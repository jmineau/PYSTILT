"""
Where to release particles for a sounding.

Points spread over a pixel (:func:`jitter_points`), points along a slant
line of sight (:func:`slant_points`), and the altitudes of a retrieval's
pressure levels to place them at (:func:`pressure_altitudes`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from random import Random
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import Point, Polygon

JitterMethod = Literal["regular", "random"]

_EARTH_RADIUS_M = 6_371_000.0
_R_DRY = 287.05  # J kg^-1 K^-1, dry air
_GRAVITY = 9.80665  # m s^-2
_STD_LAPSE_RATE = 0.0065  # K m^-1
_STD_SEA_LEVEL_TEMPERATURE = 288.15  # K


# -- slant line of sight -----------------------------------------------------------


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
    Return ``(longitude, latitude, altitude)`` points along a line of sight.

    Each point is ``(altitude - anchor) * tan(zenith)`` meters from
    ``(longitude, latitude)`` along the ``azimuth`` bearing, on a flat local
    tangent plane. Pass the result to :meth:`stilt.Receptor.from_points`.

    Parameters
    ----------
    longitude, latitude : float
        Where the path passes through ``anchor``, in degrees.
    altitudes : array-like
        Altitudes of the points, in meters. They are returned unchanged. Use
        altitudes above sea level, since heights above ground would bend the
        path with the terrain.
    zenith : float
        Angle of the path from the local vertical, in degrees, from 0 up to
        90.
    azimuth : float
        Bearing toward the instrument or the sun, in degrees clockwise from
        north. The path rises in this direction.
    anchor : float, optional
        Altitude at which the path passes through ``(longitude, latitude)``.
        Defaults to the first altitude.

    Returns
    -------
    list of (float, float, float)
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


def pressure_altitudes(
    pressures: ArrayLike,
    *,
    surface_pressure: float,
    surface_altitude: float,
    temperature: ArrayLike | None = None,
    top: float | None = None,
) -> np.ndarray:
    """
    Return the altitudes above sea level of a retrieval's pressure levels.

    Use the result as the ``altitudes`` of :func:`slant_points`, so the
    slant follows the retrieval's own layers. The pressure levels are, for
    example, OCO-2's ``pressure_levels`` or TROPOMI's ``surface_pressure``
    minus multiples of ``pressure_interval``. Altitudes come from the
    hypsometric equation, integrated up from the surface.

    Parameters
    ----------
    pressures : array-like
        Pressure levels, in hPa, in any order.
    surface_pressure : float
        The sounding's surface pressure, in hPa. Levels at higher pressure
        are below the surface and are dropped.
    surface_altitude : float
        The sounding's surface altitude, in meters above sea level.
    temperature : array-like, optional
        Temperature at each pressure level, in K. The altitudes are
        integrated layer by layer with each layer's mean temperature, and the
        layer between the surface and the first level takes the first
        level's temperature. Pass it when the retrieval or its prior gives a
        profile. ``None`` uses the standard atmosphere lapse rate of 6.5 K/km
        from a surface temperature of 288.15 K minus 6.5 K/km times the
        surface altitude; this matches the U.S. Standard Atmosphere below
        11 km.
    top : float, optional
        Drop levels above this altitude, in meters above sea level, such as
        the top of the meteorology.

    Returns
    -------
    numpy.ndarray
        Altitudes in meters above sea level, from the surface upward, so the
        first one anchors the slant at the sounding's location.
    """
    p = np.asarray(pressures, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("pressure_altitudes requires at least one pressure level.")
    if not np.all(p > 0):
        raise ValueError("pressure_altitudes pressures must be positive (hPa).")
    if not surface_pressure > 0:
        raise ValueError("pressure_altitudes surface_pressure must be positive (hPa).")

    t = None if temperature is None else np.asarray(temperature, dtype=float).ravel()
    if t is not None and t.shape != p.shape:
        raise ValueError(
            "pressure_altitudes temperature profile must have one value per "
            f"pressure level ({t.size} temperatures for {p.size} levels)."
        )

    keep = p <= surface_pressure
    order = np.argsort(-p[keep], kind="stable")
    p = p[keep][order]
    if p.size == 0:
        raise ValueError(
            "pressure_altitudes found no pressure levels at or above the surface "
            f"({surface_pressure} hPa)."
        )
    if t is not None:
        t = t[keep][order]

    z_sfc = float(surface_altitude)
    if t is not None:
        edges_p = np.concatenate(([surface_pressure], p))
        edges_t = np.concatenate(([t[0]], t))
        t_mean = 0.5 * (edges_t[:-1] + edges_t[1:])
        dz = _R_DRY * t_mean / _GRAVITY * np.log(edges_p[:-1] / edges_p[1:])
        z = z_sfc + np.cumsum(dz)
    else:
        t_sfc = _STD_SEA_LEVEL_TEMPERATURE - _STD_LAPSE_RATE * z_sfc
        exponent = _R_DRY * _STD_LAPSE_RATE / _GRAVITY
        z = z_sfc + t_sfc / _STD_LAPSE_RATE * (1 - (p / surface_pressure) ** exponent)

    if top is not None:
        z = z[z <= top]
        if z.size == 0:
            raise ValueError(
                f"pressure_altitudes found no pressure levels below top={top} m."
            )
    return z


# -- jitter ------------------------------------------------------------------------


def _sample_regular(polygon: Polygon, n: int) -> list[tuple[float, float]]:
    """Return ``n`` points on a regular grid inside a polygon."""
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
    """Return ``n`` uniformly random points inside a polygon."""
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
    Return ``(longitude, latitude)`` points spread over one pixel.

    Running several receptors across a large pixel and averaging their
    footprints represents the pixel better than one receptor at its center.
    This is X-STILT's ``jitterTF``.

    Parameters
    ----------
    polygon : shapely.Polygon or sequence of (float, float)
        Pixel outline, as a polygon or its corner coordinates.
    n : int
        Number of points.
    method : {"regular", "random"}, default "regular"
        ``"regular"`` places the points on a grid clipped to the polygon.
        ``"random"`` draws them uniformly.
    seed : int, optional
        Random seed for ``method="random"``.

    Returns
    -------
    list of (float, float)
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    outline = polygon if isinstance(polygon, Polygon) else Polygon(polygon)
    if method == "regular":
        return _sample_regular(outline, n)
    if method == "random":
        return _sample_random(outline, n, seed=seed)
    raise ValueError(f"Unknown jitter method: {method!r}")


__all__ = ["jitter_points", "pressure_altitudes", "slant_points"]
