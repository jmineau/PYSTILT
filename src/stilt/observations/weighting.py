"""Generic particle-weighting interfaces for observation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from .operators import VerticalOperator

if TYPE_CHECKING:
    from .observation import Observation


# Modes that weight each particle by the fraction of the column's air mass it
# represents. Following X-STILT, the pressure weighting function (PWF) is
# derived from the particles' own release pressures rather than from a
# user-supplied profile; see ``_particle_pwf``.
#
# ``Footprint.calculate`` divides the aggregated influence by ``n_particles``,
# so the weights are multiplied by ``n_particles`` to keep the weighted
# footprint's magnitude independent of ``numpar``.
_PWF_MODES: frozenset[str] = frozenset({"pwf", "ak_pwf"})
_AK_MODES: frozenset[str] = frozenset({"ak", "ak_pwf"})


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


def _particle_pwf(
    p: pd.DataFrame, surface_pressure: float | None
) -> tuple[pd.Series, pd.Series]:
    """
    Derive each particle's release pressure and pressure weight.

    Follows X-STILT's ``get.wgt.*.func``: fit a hypsometric curve
    ``ln p = b + a·z`` to the particles' first-step heights and pressures
    (this smooths the one-time-step offset from the true release state and
    yields a surface pressure when none is supplied), evaluate it at each
    particle's release height, and turn the spacing between neighbouring
    release pressures into the air mass each particle represents.

    HYSPLIT spreads column particles evenly over height, each one randomized
    within its own ``1/numpar`` slab, so a particle stands for the slab
    centred on it: the cell edges sit midway between adjacent release
    pressures, the surface closes the bottom, and the topmost cell mirrors its
    lower half-width.  (X-STILT instead gives each particle the layer *below*
    it, which shifts every weight down by half a cell and leaves the lowest
    particle with almost none.)

    Returns ``(xpres, pwf)`` indexed by ``indx``. ``pwf`` sums to the fraction
    of the atmosphere's mass the column covers, ``(p_sfc - p_top) / p_sfc``;
    the rest lies above the column top, where surface fluxes cannot reach the
    receptor within the back-trajectory.
    """
    for col in ("pres", "zagl"):
        if col not in p.columns:
            raise ValueError(
                f"Pressure weighting requires the {col!r} particle variable; "
                "include it in STILTParams.varsiwant."
            )
    pres = _release_coordinate(p, "pres")
    zagl = _release_coordinate(p, "zagl")
    z_release = _release_coordinate(p, "xhgt") if "xhgt" in p.columns else zagl

    if zagl.nunique() < 2:
        raise ValueError(
            "Pressure weighting needs particles released over a range of heights "
            "(a ColumnReceptor); all particles share one release height."
        )
    a, b = np.polyfit(zagl.to_numpy(), np.log(pres.to_numpy()), 1)
    if a >= 0:
        raise ValueError(
            "Could not fit a pressure profile to the particles (pressure does "
            "not decrease with height)."
        )
    p_sfc = (
        float(surface_pressure) if surface_pressure is not None else float(np.exp(b))
    )

    xpres = pd.Series(p_sfc * np.exp(a * z_release.to_numpy()), index=z_release.index)
    ordered = xpres.sort_values(ascending=False)  # surface upward
    levels = ordered.to_numpy()

    mids = (levels[:-1] + levels[1:]) / 2.0
    lower_edges = np.concatenate(([p_sfc], mids))
    upper_edges = np.concatenate((mids, [2 * levels[-1] - mids[-1]]))
    pwf = pd.Series((lower_edges - upper_edges) / p_sfc, index=ordered.index)
    return xpres, pwf


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


def _ak_weights(
    p: pd.DataFrame, operator: VerticalOperator, coordinate: str
) -> np.ndarray:
    """Interpolate the averaging kernel to each row's particle release coordinate."""
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
    return np.interp(coords, levels, values, left=values[0], right=values[-1])


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

    weights = np.ones(len(p))
    if operator.mode in _AK_MODES:
        weights = weights * _ak_weights(p, operator, coordinate)

    if operator.mode in _PWF_MODES:
        xpres, pwf = _particle_pwf(p, operator.surface_pressure)
        indx = p["indx"].to_numpy()
        p["xpres"] = xpres.reindex(indx).to_numpy()
        p["pwf"] = pwf.reindex(indx).to_numpy()
        weights = weights * p["pwf"].to_numpy() * len(pwf)

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
