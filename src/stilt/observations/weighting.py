"""Generic particle-weighting interfaces for observation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from .operators import VerticalOperator

if TYPE_CHECKING:
    from .observation import Observation


# Modes that require scaling by n_particles.
#
# The footprint calculator divides the aggregated particle influence by
# n_particles, which means that fractional pressure-weight profiles (PWF sums
# to 1 across the column) must be rescaled to preserve the column-integrated
# signal. AK-only modes do not need this correction because AK_norm is already
# dimensionless and approximately normalized without a per-particle factor.
_PWF_MODES: frozenset[str] = frozenset({"pwf", "ak_pwf", "integration", "tccon"})


@dataclass(frozen=True, slots=True)
class WeightingContext:
    """Context passed into a particle-weighting model.

    This is intentionally small for the first pass. It gives the weighting interface
    a place to carry observation/operator metadata today while leaving room for
    later chemistry/lifetime extensions without coupling them to transport code.
    """

    coordinate: str = "xhgt"
    observation: Observation | None = None
    operator: VerticalOperator | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WeightingModel(Protocol):
    """Behavioral interface for models that reweight particle sensitivities."""

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: WeightingContext | None = None,
    ) -> pd.DataFrame:
        """Return particles after applying a weighting model."""
        ...


class NoOpWeighting:
    """Weighting model that returns an unchanged copy of the particles."""

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: WeightingContext | None = None,
    ) -> pd.DataFrame:
        """Return an unchanged copy of the particles."""
        return particles.copy()


class VerticalOperatorWeighting:
    """Apply a :class:`VerticalOperator` to particle ``foot`` values."""

    def __init__(self, operator: VerticalOperator | None = None) -> None:
        self._operator = operator

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: WeightingContext | None = None,
    ) -> pd.DataFrame:
        """Apply a vertical operator to the particle ``foot`` values."""
        operator = self._operator or (context.operator if context is not None else None)
        if operator is None:
            raise ValueError(
                "VerticalOperatorWeighting requires a VerticalOperator either "
                "at construction time or in WeightingContext.operator."
            )
        coordinate = context.coordinate if context is not None else "xhgt"
        return _apply_vertical_operator_impl(
            particles,
            operator,
            coordinate=coordinate,
        )


def apply_weighting(
    particles: pd.DataFrame,
    weighting: WeightingModel,
    *,
    context: WeightingContext | None = None,
) -> pd.DataFrame:
    """Apply a weighting model to a particle DataFrame."""
    return weighting.apply(particles, context=context)


def _apply_vertical_operator_impl(
    particles: pd.DataFrame,
    operator: VerticalOperator,
    *,
    coordinate: str = "xhgt",
) -> pd.DataFrame:
    """Internal implementation used by the vertical-operator weighting interface."""
    if operator.mode == "none":
        return particles

    p = particles.copy()

    # Re-weighting guard: restore the original foot before applying new weights.
    if "foot_before_weight" in p.columns:
        p["foot"] = p["foot_before_weight"]
        p = p.drop(columns=["foot_before_weight"])

    if operator.mode == "uniform":
        return p

    if coordinate not in p.columns:
        raise ValueError(
            f"Particle DataFrame has no column {coordinate!r}. "
            "Assign release heights ('xhgt') before applying a vertical operator, "
            "or pass coordinate='pres' for pressure-based interpolation."
        )

    if not operator.levels or not operator.values:
        raise ValueError(
            "VerticalOperator.levels and .values must both be non-empty "
            f"for mode={operator.mode!r}."
        )

    levels = np.asarray(operator.levels, dtype=float)
    values = np.asarray(operator.values, dtype=float)

    if len(levels) != len(values):
        raise ValueError(
            f"VerticalOperator.levels ({len(levels)}) and .values "
            f"({len(values)}) must have the same length."
        )

    sort_idx = np.argsort(levels)
    levels = levels[sort_idx]
    values = values[sort_idx]

    coords = p[coordinate].to_numpy(dtype=float)
    weights = np.interp(coords, levels, values, left=values[0], right=values[-1])

    if operator.mode in _PWF_MODES:
        n_particles = int(cast(pd.Series, p["indx"]).nunique())
        weights = weights * n_particles

    p["foot_before_weight"] = p["foot"]
    p["foot"] = p["foot"] * weights
    return p


__all__ = [
    "NoOpWeighting",
    "VerticalOperatorWeighting",
    "WeightingContext",
    "WeightingModel",
    "apply_weighting",
]
