"""Receptor points along a slant line of sight."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

_EARTH_RADIUS_M = 6_371_000.0
_R_DRY = 287.05  # J kg^-1 K^-1, dry air
_GRAVITY = 9.80665  # m s^-2
_STD_LAPSE_RATE = 0.0065  # K m^-1
_STD_SEA_LEVEL_TEMPERATURE = 288.15  # K


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
    temperature: float | ArrayLike | None = None,
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
    temperature : float or array-like, optional
        Temperature in K. ``None`` uses the standard atmosphere lapse rate of
        6.5 K/km from a surface temperature of 288.15 K minus 6.5 K/km times
        the surface altitude; this matches the U.S. Standard Atmosphere below
        11 km. A single value assumes an isothermal atmosphere, with scale
        height ``R_d T / g`` (7.3 km at 250 K). One value per level
        integrates layer by layer with each layer's mean temperature, and the
        layer between the surface and the first level takes the first
        level's temperature. Use a profile when the retrieval or its prior
        gives one.
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

    t_in = None if temperature is None else np.asarray(temperature, dtype=float)
    t = t_in.ravel() if t_in is not None and t_in.ndim > 0 else None
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
    elif t_in is not None:
        scale_height = _R_DRY * float(t_in) / _GRAVITY
        z = z_sfc + scale_height * np.log(surface_pressure / p)
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


__all__ = ["pressure_altitudes", "slant_points"]
