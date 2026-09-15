"""Generic particle-weighting interfaces for observation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from .operators import VerticalOperator

if TYPE_CHECKING:
    from .observation import Observation


# Modes whose profile values are *layer weights* (a PWF sums to 1 across the
# column) rather than dimensionless per-particle scalings.
#
# The footprint calculator divides the aggregated particle influence by
# n_particles. To preserve the column-integrated signal, each operator level's
# weight must be shared among the particles released nearest to that level:
# ``weight_i = value(z_i) * n_particles / n_particles_at_level(i)``. When the
# profile is sampled once per particle (the X-STILT convention) the level
# counts are all one and this reduces to ``value * n_particles``; when a coarse
# retrieval profile is supplied, the per-level count keeps the magnitude
# independent of ``numpar``. AK-only modes do not need this correction because
# AK_norm is already dimensionless.
_PWF_MODES: frozenset[str] = frozenset({"pwf", "ak_pwf", "integration", "tccon"})


def _release_coordinate(p: pd.DataFrame, coordinate: str) -> pd.Series:
    """
    Return each particle's *release* value of ``coordinate`` indexed by ``indx``.

    Release coordinates such as ``xhgt`` are constant along a trajectory, but
    HYSPLIT diagnostics such as ``pres`` are written at every time step.  The
    row nearest the receptor time (smallest ``|time|`` when ``time`` is in
    minutes; otherwise the first row) defines the release value.
    """
    if "time" in p.columns and pd.api.types.is_numeric_dtype(p["time"]):
        ordered = p.assign(_age=p["time"].abs()).sort_values("_age", kind="stable")
    else:
        ordered = p
    first = ordered.drop_duplicates(subset="indx")
    return pd.Series(
        first[coordinate].to_numpy(dtype=float), index=first["indx"].to_numpy()
    )


def _nearest_level_index(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return, for each coordinate in *x*, the index of the nearest ascending level."""
    if len(levels) == 1:
        return np.zeros(len(x), dtype=int)
    idx = np.clip(np.searchsorted(levels, x), 1, len(levels) - 1)
    lower = levels[idx - 1]
    upper = levels[idx]
    return np.where(np.abs(x - lower) <= np.abs(upper - x), idx - 1, idx)


@dataclass(frozen=True, slots=True)
class WeightingContext:
    """
    Context passed into a particle-weighting model.

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

    # One release value per particle, broadcast along its trajectory rows.
    per_particle = _release_coordinate(p, coordinate)
    coords = per_particle.reindex(p["indx"].to_numpy()).to_numpy(dtype=float)

    if operator.mode in _PWF_MODES:
        # Layer weights: each particle takes the value of its nearest level
        # (piecewise constant), shared among the particles at that level.
        n_particles = len(per_particle)
        level_counts = np.bincount(
            _nearest_level_index(per_particle.to_numpy(dtype=float), levels),
            minlength=len(levels),
        )
        row_levels = _nearest_level_index(coords, levels)
        weights = (
            values[row_levels] * n_particles / np.maximum(level_counts[row_levels], 1)
        )
    else:
        weights = np.interp(coords, levels, values, left=values[0], right=values[-1])

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
