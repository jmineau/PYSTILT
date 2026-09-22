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
from .scenes import (
    Scene,
    group_scenes_by_key,
    group_scenes_by_metadata,
    group_scenes_by_swath,
    group_scenes_by_time_gap,
    make_scene,
)
from .selection import (
    filter_observations,
    jitter_observation,
    select_observations_spatial,
)
from .sensors import BaseSensor, ColumnSensor, PointSensor, Sensor
from .uncertainty import UncertaintyBudget, UncertaintyComponent

__all__ = [
    "BaseSensor",
    "ColumnSensor",
    "HorizontalGeometry",
    "LineOfSight",
    "Observation",
    "PointSensor",
    "Scene",
    "Sensor",
    "ViewingGeometry",
    "UncertaintyBudget",
    "UncertaintyComponent",
    "build_column_receptor",
    "build_multipoint_receptor",
    "build_point_receptor",
    "build_slant_receptor",
    "filter_observations",
    "group_scenes_by_key",
    "group_scenes_by_metadata",
    "group_scenes_by_swath",
    "group_scenes_by_time_gap",
    "jitter_observation",
    "make_scene",
    "select_observations_spatial",
]
