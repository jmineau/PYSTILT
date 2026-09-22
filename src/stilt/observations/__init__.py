"""
Observation-domain models and sensor interfaces for science workflows.

Particle weighting (averaging kernel, pressure weighting, lifetime decay) lives
in :mod:`stilt.transforms`.
"""

from .builders import build_slant_receptor, slant_points
from .geometry import HorizontalGeometry, ViewingGeometry
from .observation import Observation
from .scenes import Scene, group_by_overpass, group_observations
from .selection import (
    filter_observations,
    jitter_observation,
    select_observations_spatial,
)

__all__ = [
    "HorizontalGeometry",
    "Observation",
    "Scene",
    "group_by_overpass",
    "group_observations",
    "ViewingGeometry",
    "build_slant_receptor",
    "filter_observations",
    "jitter_observation",
    "select_observations_spatial",
    "slant_points",
]
