"""Configuration models and runtime bootstrap helpers for STILT runs."""

from .footprint import FootprintConfig, foot_names
from .geometry import (
    FileGeometrySpec,
    GeometrySpec,
    H3GeometrySpec,
    WindowsGeometrySpec,
)
from .meteorology import MetConfig
from .model import ModelConfig
from .params import ErrorParams, ModelParams, STILTParams, TransportParams
from .runtime import RuntimeSettings
from .spatial import (
    Bounds,
    Grid,
    VerticalReference,
    kmsl_from_vertical_reference,
    validate_vertical_reference,
)

__all__ = [
    "Bounds",
    "ErrorParams",
    "FileGeometrySpec",
    "FootprintConfig",
    "GeometrySpec",
    "H3GeometrySpec",
    "Grid",
    "MetConfig",
    "ModelConfig",
    "ModelParams",
    "RuntimeSettings",
    "STILTParams",
    "TransportParams",
    "VerticalReference",
    "WindowsGeometrySpec",
    "foot_names",
    "kmsl_from_vertical_reference",
    "validate_vertical_reference",
]
