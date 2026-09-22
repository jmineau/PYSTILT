"""Configuration models and runtime bootstrap helpers for STILT runs."""

from .fields import cfg_field
from .footprint import FootprintConfig, foot_names
from .geometry import (
    FileGeometrySpec,
    GeometrySpec,
    H3GeometrySpec,
    WindowsGeometrySpec,
)
from .meteorology import MetConfig
from .model import (
    ModelConfig,
    build_control_entries,
    build_setup_entries,
    iter_documented_config_fields,
)
from .params import ErrorParams, ModelParams, STILTParams, TransportParams
from .runtime import RuntimeSettings, resolve_runtime_settings
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
    "build_control_entries",
    "build_setup_entries",
    "cfg_field",
    "foot_names",
    "iter_documented_config_fields",
    "kmsl_from_vertical_reference",
    "resolve_runtime_settings",
    "validate_vertical_reference",
]
