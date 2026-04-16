"""Typed pre-footprint particle transform helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from stilt.config import (
    FirstOrderLifetimeTransformSpec,
    ParticleTransformSpec,
    VerticalOperatorTransformSpec,
)
from stilt.observations import (
    ChemistryContext,
    FirstOrderLifetimeChemistry,
    VerticalOperator,
    VerticalOperatorWeighting,
    WeightingContext,
)
from stilt.receptor import Receptor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stilt.config import FootprintConfig
    from stilt.observations import Observation


@dataclass(frozen=True, slots=True)
class ParticleTransformContext:
    """Context passed into pre-footprint particle transforms."""

    receptor: Receptor
    footprint_name: str
    footprint_config: FootprintConfig
    is_error: bool = False
    observation: Observation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ParticleTransform(Protocol):
    """Behavioral interface for typed particle transforms."""

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ParticleTransformContext | None = None,
    ) -> pd.DataFrame:
        """Return a transformed particle table."""
        ...


class _ConfiguredVerticalOperatorTransform:
    """Configured vertical-operator weighting built from a config spec."""

    def __init__(self, spec: VerticalOperatorTransformSpec) -> None:
        operator = VerticalOperator(
            mode=spec.mode,
            levels=list(spec.levels),
            values=list(spec.values),
            pressure_levels=list(spec.pressure_levels),
            metadata=dict(spec.metadata),
        )
        self._coordinate = spec.coordinate
        self._weighting = VerticalOperatorWeighting(operator)
        self._metadata = dict(spec.metadata)

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ParticleTransformContext | None = None,
    ) -> pd.DataFrame:
        """Apply the configured vertical operator weighting."""
        weighting_context = WeightingContext(
            coordinate=self._coordinate,
            observation=context.observation if context is not None else None,
            metadata={
                **(context.metadata if context is not None else {}),
                **self._metadata,
            },
        )
        return self._weighting.apply(particles, context=weighting_context)


class _ConfiguredFirstOrderLifetimeTransform:
    """Configured first-order lifetime chemistry built from a config spec."""

    def __init__(self, spec: FirstOrderLifetimeTransformSpec) -> None:
        self._chemistry = FirstOrderLifetimeChemistry(spec.lifetime_hours)
        self._time_column = spec.time_column
        self._time_unit = spec.time_unit
        self._metadata = dict(spec.metadata)

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ParticleTransformContext | None = None,
    ) -> pd.DataFrame:
        """Apply the configured first-order lifetime decay."""
        chemistry_context = ChemistryContext(
            observation=context.observation if context is not None else None,
            species=(
                context.observation.species
                if context is not None and context.observation is not None
                else None
            ),
            time_column=self._time_column,
            time_unit=self._time_unit,
            metadata={
                **(context.metadata if context is not None else {}),
                **self._metadata,
            },
        )
        return self._chemistry.apply(particles, context=chemistry_context)


def build_particle_transform(spec: ParticleTransformSpec) -> ParticleTransform:
    """Build one typed particle transform from a declarative config spec."""
    if isinstance(spec, VerticalOperatorTransformSpec):
        return _ConfiguredVerticalOperatorTransform(spec)
    if isinstance(spec, FirstOrderLifetimeTransformSpec):
        return _ConfiguredFirstOrderLifetimeTransform(spec)
    raise TypeError(f"Unsupported particle transform spec: {type(spec)!r}")


def build_particle_transforms(
    specs: Iterable[ParticleTransformSpec],
) -> list[ParticleTransform]:
    """Build a list of typed particle transforms from config specs."""
    return [build_particle_transform(spec) for spec in specs]


def apply_particle_transforms(
    particles: pd.DataFrame,
    transforms: Iterable[ParticleTransform],
    *,
    context: ParticleTransformContext | None = None,
) -> pd.DataFrame:
    """Apply particle transforms sequentially before footprint rasterization."""
    result = particles
    for transform in transforms:
        result = transform.apply(result, context=context)
    return result


__all__ = [
    "ParticleTransform",
    "ParticleTransformContext",
    "apply_particle_transforms",
    "build_particle_transform",
    "build_particle_transforms",
]
