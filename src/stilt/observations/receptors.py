"""Observation-to-receptor conversion helpers."""

from __future__ import annotations

import math

import numpy as np

from stilt.config import VerticalReference
from stilt.observations.geometry import LineOfSight
from stilt.observations.observation import Observation
from stilt.receptor import Receptor

_EARTH_RADIUS_M = 6_371_000.0


def build_point_receptor(
    observation: Observation,
    *,
    altitude: float | None = None,
) -> Receptor:
    """Build a point receptor for one observation.

    If ``altitude`` is omitted, the observation altitude is used.
    """
    receptor_altitude = observation.altitude if altitude is None else altitude
    if receptor_altitude is None:
        raise ValueError(
            "A point receptor altitude is required. Pass `altitude=` or set "
            "`Observation.altitude`."
        )
    return Receptor(
        time=observation.time,
        longitude=observation.longitude,
        latitude=observation.latitude,
        altitude=receptor_altitude,
        altitude_ref=observation.altitude_ref,
    )


def build_column_receptor(
    observation: Observation,
    *,
    bottom: float,
    top: float,
    altitude_ref: VerticalReference = "agl",
) -> Receptor:
    """Build a vertical-column receptor centered on one observation."""
    return Receptor.from_column(
        time=observation.time,
        longitude=observation.longitude,
        latitude=observation.latitude,
        bottom=bottom,
        top=top,
        altitude_ref=altitude_ref,
    )


def build_multipoint_receptor(
    observation: Observation,
    *,
    points: list[tuple[float, float, float]],
    altitude_ref: VerticalReference = "agl",
) -> Receptor:
    """Build a multipoint receptor from explicit release points."""
    return Receptor.from_points(
        observation.time,
        points,
        altitude_ref=altitude_ref,
    )


def _sample_los_altitudes(
    line_of_sight: LineOfSight,
    *,
    surface_altitude: float | None = None,
    max_altitude: float | None = None,
) -> list[float]:
    """Return altitude samples for one LOS definition."""
    if line_of_sight.altitude_levels:
        altitudes = [float(v) for v in line_of_sight.altitude_levels]
    else:
        assert line_of_sight.start_altitude is not None
        assert line_of_sight.end_altitude is not None
        if line_of_sight.count is not None:
            if line_of_sight.count < 2:
                raise ValueError("LineOfSight.count must be >= 2.")
            altitudes = np.linspace(
                line_of_sight.start_altitude,
                line_of_sight.end_altitude,
                line_of_sight.count,
            ).tolist()
        else:
            assert line_of_sight.frequency is not None
            if line_of_sight.frequency <= 0:
                raise ValueError("LineOfSight.frequency must be positive.")
            altitude_range = abs(
                line_of_sight.end_altitude - line_of_sight.start_altitude
            )
            count = max(2, int(round(altitude_range / line_of_sight.frequency)) + 1)
            altitudes = np.linspace(
                line_of_sight.start_altitude,
                line_of_sight.end_altitude,
                count,
            ).tolist()

    lower_bound = 0.0 if line_of_sight.altitude_ref == "agl" else None
    if surface_altitude is not None and line_of_sight.altitude_ref == "msl":
        lower_bound = surface_altitude
    if lower_bound is not None:
        altitudes = [v for v in altitudes if v >= lower_bound]
    if max_altitude is not None:
        altitudes = [v for v in altitudes if v <= max_altitude]
    if not altitudes:
        raise ValueError("LOS sampling produced no receptor points after clipping.")
    return altitudes


def _resolved_los_surface_altitude(
    observation: Observation,
    line_of_sight: LineOfSight,
    *,
    surface_altitude: float | None = None,
) -> float | None:
    """Return the effective lower clipping altitude for one LOS."""
    if surface_altitude is not None:
        return surface_altitude
    if line_of_sight.surface_altitude is not None:
        return line_of_sight.surface_altitude
    if (
        line_of_sight.altitude_ref == "msl"
        and observation.altitude_ref == "msl"
        and observation.altitude is not None
    ):
        return float(observation.altitude)
    return None


def _resolved_los_max_altitude(
    line_of_sight: LineOfSight,
    *,
    model_top_altitude: float | None = None,
) -> float | None:
    """Return the effective upper clipping altitude for one LOS."""
    if model_top_altitude is not None:
        return model_top_altitude
    return line_of_sight.max_altitude


def _offset_lon_lat(
    *,
    longitude: float,
    latitude: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    """Move a point in local tangent-plane ENU coordinates."""
    lat_rad = math.radians(latitude)
    dlat = north_m / _EARTH_RADIUS_M
    dlon = east_m / (_EARTH_RADIUS_M * math.cos(lat_rad))
    return (
        longitude + math.degrees(dlon),
        latitude + math.degrees(dlat),
    )


def build_slant_receptor(
    observation: Observation,
    *,
    line_of_sight: LineOfSight | None = None,
    surface_altitude: float | None = None,
    model_top_altitude: float | None = None,
) -> Receptor:
    """Build a slant multipoint receptor from observation LOS geometry.

    When explicit clipping values are omitted, the builder uses the strongest
    available hints:

    - anchor altitude: ``LineOfSight.anchor_altitude`` -> ``Observation.altitude``
    - surface clipping for MSL LOS: explicit ``surface_altitude`` ->
      ``LineOfSight.surface_altitude`` -> ``Observation.altitude`` when the
      observation altitude is also MSL
    - model-top clipping: explicit ``model_top_altitude`` ->
      ``LineOfSight.max_altitude``
    """
    los = line_of_sight or observation.line_of_sight
    if los is None:
        raise ValueError(
            "A slant receptor requires Observation.line_of_sight or an explicit "
            "`line_of_sight=` argument."
        )
    if observation.viewing is None:
        raise ValueError("A slant receptor requires Observation.viewing geometry.")
    if observation.viewing.viewing_zenith_angle is None:
        raise ValueError(
            "A slant receptor requires ViewingGeometry.viewing_zenith_angle."
        )
    if observation.viewing.viewing_azimuth_angle is None:
        raise ValueError(
            "A slant receptor requires ViewingGeometry.viewing_azimuth_angle."
        )

    viewing_zenith = float(observation.viewing.viewing_zenith_angle)
    viewing_azimuth = float(observation.viewing.viewing_azimuth_angle)
    if viewing_zenith < 0 or viewing_zenith >= 90:
        raise ValueError(
            "Slant receptor viewing_zenith_angle must be in [0, 90) degrees."
        )

    altitudes = _sample_los_altitudes(
        los,
        surface_altitude=_resolved_los_surface_altitude(
            observation,
            los,
            surface_altitude=surface_altitude,
        ),
        max_altitude=_resolved_los_max_altitude(
            los,
            model_top_altitude=model_top_altitude,
        ),
    )
    anchor_altitude = (
        los.anchor_altitude
        if los.anchor_altitude is not None
        else observation.altitude
        if observation.altitude is not None
        else altitudes[0]
    )

    zenith_rad = math.radians(viewing_zenith)
    azimuth_rad = math.radians(viewing_azimuth)

    points: list[tuple[float, float, float]] = []
    for altitude in altitudes:
        delta_altitude = altitude - anchor_altitude
        horizontal_distance = delta_altitude * math.tan(zenith_rad)
        east_m = horizontal_distance * math.sin(azimuth_rad)
        north_m = horizontal_distance * math.cos(azimuth_rad)
        longitude, latitude = _offset_lon_lat(
            longitude=observation.longitude,
            latitude=observation.latitude,
            east_m=east_m,
            north_m=north_m,
        )
        points.append((longitude, latitude, altitude))

    return Receptor.from_points(
        observation.time,
        points,
        altitude_ref=los.altitude_ref,
    )
