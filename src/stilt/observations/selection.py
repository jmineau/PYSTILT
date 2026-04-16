"""Observation selection and jitter helpers."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from random import Random
from typing import Literal

import pandas as pd
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon
from shapely.ops import transform

from .geometry import HorizontalGeometry
from .observation import Observation

JitterMethod = Literal["regular", "random"]
ObservationPredicate = Callable[[Observation], bool]
TimestampLike = pd.Timestamp | dt.datetime | str


def _projection_transforms(
    geometry: HorizontalGeometry,
):
    """Return forward/inverse local-projection transforms for one geometry."""
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise ImportError(
            "Synthetic observation geometry requires pyproj. "
            "Install it with: pip install 'pystilt[projection]'"
        ) from exc

    local_crs = CRS.from_proj4(
        "+proj=aeqd "
        f"+lat_0={geometry.center_latitude} "
        f"+lon_0={geometry.center_longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    inverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform
    return forward, inverse


def _rectangle_from_geometry(geometry: HorizontalGeometry) -> Polygon:
    """Approximate one horizontal geometry as a local projected rectangle."""
    if geometry.resolution_km is None:
        raise ValueError(
            "HorizontalGeometry.resolution_km is required when vertices are absent."
        )
    width_km, height_km = geometry.resolution_km
    half_width = width_km * 500
    half_height = height_km * 500
    rect = Polygon(
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]
    )
    if geometry.orientation_deg:
        rect = rotate(rect, geometry.orientation_deg, origin=(0, 0), use_radians=False)
    _, inverse = _projection_transforms(geometry)
    return transform(inverse, rect)


def _ellipse_from_geometry(geometry: HorizontalGeometry) -> Polygon:
    """Approximate one horizontal geometry as an ellipse polygon."""
    if geometry.major_axis_km is None or geometry.minor_axis_km is None:
        raise ValueError(
            "Ellipse jitter requires major_axis_km and minor_axis_km on the geometry."
        )
    a = geometry.major_axis_km * 500
    b = geometry.minor_axis_km * 500
    points = []
    for idx in range(72):
        theta = 2 * math.pi * idx / 72
        x = a * math.cos(theta)
        y = b * math.sin(theta)
        if geometry.orientation_deg:
            angle = math.radians(geometry.orientation_deg)
            x_rot = x * math.cos(angle) - y * math.sin(angle)
            y_rot = x * math.sin(angle) + y * math.cos(angle)
            x, y = x_rot, y_rot
        points.append((x, y))
    _, inverse = _projection_transforms(geometry)
    return transform(inverse, Polygon(points))


def geometry_polygon(geometry: HorizontalGeometry) -> Polygon:
    """Build a shapely polygon approximation for an observation geometry."""
    if geometry.vertices:
        return Polygon(geometry.vertices)
    if geometry.kind == "ellipse":
        return _ellipse_from_geometry(geometry)
    if geometry.kind in {"polygon", "swath_cell", "point", "line"}:
        return _rectangle_from_geometry(geometry)
    raise ValueError(f"Unsupported geometry kind: {geometry.kind!r}")


def _sample_regular(polygon: Polygon, n: int) -> list[tuple[float, float]]:
    """Sample approximately regular points inside a polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    if polygon.area <= 0:
        raise ValueError("Cannot jitter within a zero-area geometry.")
    resolution = max(2, math.ceil(math.sqrt(n)))
    for _ in range(12):
        xs = [minx + (i + 0.5) * (maxx - minx) / resolution for i in range(resolution)]
        ys = [miny + (j + 0.5) * (maxy - miny) / resolution for j in range(resolution)]
        points = [(x, y) for y in ys for x in xs if polygon.covers(Point(x, y))]
        if len(points) >= n:
            return points[:n]
        resolution += 1
    raise ValueError("Unable to generate enough regular jitter points inside geometry.")


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
            "Unable to generate enough random jitter points inside geometry."
        )
    return points


def jitter_observation(
    observation: Observation,
    *,
    n: int,
    method: JitterMethod = "regular",
    seed: int | None = None,
) -> list[Observation]:
    """Expand one observation into jittered observation realizations.

    The returned observations have point geometry at the sampled locations and
    preserve the original observation metadata.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if observation.geometry is None:
        raise ValueError("Observation.geometry is required for jitter.")

    polygon = geometry_polygon(observation.geometry)
    if method == "regular":
        samples = _sample_regular(polygon, n)
    elif method == "random":
        samples = _sample_random(polygon, n, seed=seed)
    else:
        raise ValueError(f"Unknown jitter method: {method!r}")

    parent_id = observation.observation_id
    jittered: list[Observation] = []
    for idx, (lon, lat) in enumerate(samples, start=1):
        jitter_geometry = HorizontalGeometry(
            kind="point",
            center_longitude=lon,
            center_latitude=lat,
            metadata={
                "jitter_index": idx,
                "jitter_method": method,
                "parent_geometry_kind": observation.geometry.kind,
                "parent_observation_id": parent_id,
            },
        )
        jittered.append(
            replace(
                observation,
                longitude=lon,
                latitude=lat,
                observation_id=f"{parent_id}-j{idx}" if parent_id is not None else None,
                geometry=jitter_geometry,
                metadata={
                    **observation.metadata,
                    "parent_observation_id": parent_id,
                    "jitter_index": idx,
                    "jitter_method": method,
                },
            )
        )
    return jittered


# ---------------------------------------------------------------------------
# Spatial observation selection
# ---------------------------------------------------------------------------


def _coerce_string_set(
    value: str | Sequence[str] | None,
) -> set[str] | None:
    """Normalize one string or string sequence into a set."""
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def filter_observations(
    observations: Sequence[Observation],
    *,
    sensors: str | Sequence[str] | None = None,
    species: str | Sequence[str] | None = None,
    observation_ids: str | Sequence[str] | None = None,
    time_range: tuple[TimestampLike, TimestampLike] | None = None,
    metadata: Mapping[str, object] | None = None,
    quality: Mapping[str, float | int | bool] | None = None,
    predicate: ObservationPredicate | None = None,
) -> list[Observation]:
    """Return observations that match simple generic filtering criteria.

    This is the non-spatial counterpart to :func:`select_observations_spatial`.
    It preserves input order and only applies exact-match checks so it stays
    lightweight and predictable for the first observation-layer pass.
    """
    allowed_sensors = _coerce_string_set(sensors)
    allowed_species = _coerce_string_set(species)
    allowed_ids = _coerce_string_set(observation_ids)

    if time_range is None:
        start = end = None
    else:
        start = pd.Timestamp(time_range[0])
        end = pd.Timestamp(time_range[1])
        if start is pd.NaT or end is pd.NaT:
            raise ValueError("time_range values must be valid timestamps.")

    filtered: list[Observation] = []
    for observation in observations:
        if allowed_sensors is not None and observation.sensor not in allowed_sensors:
            continue
        if allowed_species is not None and observation.species not in allowed_species:
            continue
        if allowed_ids is not None and observation.observation_id not in allowed_ids:
            continue
        observation_time = pd.Timestamp(observation.time)
        if observation_time is pd.NaT:
            continue
        if start is not None and observation_time < start:
            continue
        if end is not None and observation_time > end:
            continue
        if metadata is not None and any(
            observation.metadata.get(key) != value for key, value in metadata.items()
        ):
            continue
        if quality is not None and any(
            observation.quality.get(key) != value for key, value in quality.items()
        ):
            continue
        if predicate is not None and not predicate(observation):
            continue
        filtered.append(observation)
    return filtered


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between two lon/lat points."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def select_observations_spatial(
    observations: Sequence[Observation],
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
) -> list[Observation]:
    """Select satellite observations using a near-field + background grid pattern.

    Ports the ``sel.obs4recpv2`` strategy from X-STILT: construct a dense
    near-field sampling grid centred on *site* and a sparse background grid
    over the full domain, then select the closest observation to each grid
    point.  Duplicate selections are collapsed so each observation appears
    once.

    This is the primary spatial selection strategy for satellite overpass
    workflows.  It ensures dense coverage near the measurement site (where
    footprint sensitivity is highest) while retaining background soundings
    needed for concentration-difference analysis.

    Parameters
    ----------
    observations:
        Candidate observations to select from (e.g., all soundings from one
        overpass).
    site_longitude, site_latitude:
        Central location around which the near-field grid is placed.
    near_field_dlon, near_field_dlat:
        Half-widths of the near-field bounding box in degrees.
    near_field_cols, near_field_rows:
        Number of grid columns / rows in the near-field zone.
    background_cols, background_rows:
        Number of grid columns / rows in the background zone.
    domain_lon_range, domain_lat_range:
        Full domain extent ``(min, max)`` in degrees for the background grid.

    Returns
    -------
    list[Observation]
        Unique selected observations ordered by latitude (ascending).

    Notes
    -----
    Selection is by nearest great-circle distance (haversine).  When two grid
    points map to the same observation, only one copy is returned.

    The near-field grid spans
    ``[site_lon ± near_field_dlon] × [site_lat ± near_field_dlat]`` with
    ``near_field_cols × near_field_rows`` evenly-spaced points.  The
    background grid spans the full *domain_lon/lat_range* with
    ``background_cols × background_rows`` evenly-spaced points.
    """
    obs_list = list(observations)
    if not obs_list:
        return []

    obs_lons = [o.longitude for o in obs_list]
    obs_lats = [o.latitude for o in obs_list]

    # Build the combined near-field + background target grid.
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

    grid_points: list[tuple[float, float]] = [
        (lon, lat) for lat in nf_lats for lon in nf_lons
    ] + [(lon, lat) for lat in bg_lats for lon in bg_lons]

    selected_indices: set[int] = set()
    for g_lon, g_lat in grid_points:
        nearest_idx = min(
            range(len(obs_list)),
            key=lambda i: _haversine_km(g_lon, g_lat, obs_lons[i], obs_lats[i]),
        )
        selected_indices.add(nearest_idx)

    return sorted(
        (obs_list[i] for i in selected_indices),
        key=lambda o: o.latitude,
    )


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Return *n* evenly-spaced values from *start* to *stop* inclusive."""
    if n == 1:
        return [(start + stop) / 2]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]
