"""Footprint config models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .fields import cfg_field
from .spatial import Grid
from .transforms import ParticleTransformSpec


class FootprintConfig(BaseModel):
    """Settings for a single named footprint product."""

    model_config = ConfigDict(frozen=True)

    grid: Grid = cfg_field(
        ...,
        description="Spatial domain and resolution for the footprint.",
    )
    smooth_factor: float = cfg_field(
        1.0,
        description="Factor by which to linearly scale footprint smoothing. Defaults to 1",
    )
    time_integrate: bool = cfg_field(
        False,
        description="If True, sum the footprint over all time steps to produce a single 2-D layer.",
    )
    error: bool = cfg_field(
        False,
        description=(
            "If True, also compute an error footprint from the error trajectories "
            'and store it alongside the main footprint under "{name}_error".'
        ),
        visibility="advanced",
    )
    transforms: list[ParticleTransformSpec] = cfg_field(
        description="Declarative particle transforms applied before rasterizing the footprint.",
        default_factory=list,
        visibility="advanced",
    )


def foot_names(foot_configs: dict[str, FootprintConfig]) -> list[str]:
    """Return all requested footprint output names, including error outputs."""
    names: list[str] = []
    for name, cfg in foot_configs.items():
        names.append(name)
        if cfg.error:
            names.append(f"{name}_error")
    return names


__all__ = [
    "FootprintConfig",
    "foot_names",
]
