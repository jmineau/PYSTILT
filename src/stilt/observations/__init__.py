"""
Observation-domain models and sensor interfaces for science workflows.

Particle weighting (averaging kernel, pressure weighting, lifetime decay) lives
in :mod:`stilt.transforms`.
"""

from .builders import (
    build_column_receptor,
    build_multipoint_receptor,
    build_point_receptor,
    build_slant_receptor,
)
from .geometry import HorizontalGeometry, LineOfSight, ViewingGeometry
from .observation import Observation
from .scenes import Scene, group_by_overpass, group_observations
from .selection import (
    filter_observations,
    jitter_observation,
    select_observations_spatial,
)

__all__ = [
    "HorizontalGeometry",
    "LineOfSight",
    "Observation",
    "Scene",
    "group_by_overpass",
    "group_observations",
    "ViewingGeometry",
    "build_column_receptor",
    "build_multipoint_receptor",
    "build_point_receptor",
    "build_slant_receptor",
    "filter_observations",
    "jitter_observation",
    "select_observations_spatial",
]
