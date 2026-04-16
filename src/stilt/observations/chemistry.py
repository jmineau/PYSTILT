"""Generic chemistry/lifetime interfaces for observation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .observation import Observation


_TIME_UNIT_TO_HOURS = {
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "m": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "d": 24.0,
    "day": 24.0,
    "days": 24.0,
}


@dataclass(frozen=True, slots=True)
class ChemistryContext:
    """Context passed into a chemistry model.

    The first pass stays intentionally small: chemistry models operate on the
    particle table and may optionally use observation/species metadata plus the
    transport-age column. This keeps the interface portable across later chemistry
    implementations without importing application orchestration into core.
    """

    observation: Observation | None = None
    species: str | None = None
    time_column: str = "time"
    time_unit: str = "min"
    metadata: dict[str, Any] = field(default_factory=dict)


class ChemistryModel(Protocol):
    """Behavioral interface for chemistry transforms on particle sensitivities."""

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ChemistryContext | None = None,
    ) -> pd.DataFrame:
        """Return particles after applying chemistry or lifetime logic."""
        ...


class NoOpChemistry:
    """Chemistry model that returns an unchanged copy of the particles."""

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ChemistryContext | None = None,
    ) -> pd.DataFrame:
        """Return an unchanged copy of the particles."""
        return particles.copy()


class FirstOrderLifetimeChemistry:
    """Apply a first-order exponential lifetime decay to particle ``foot``.

    This is a small portable chemistry model rather than a domain-specific NO2
    workflow.  It uses the transport-age column (minutes by default) and a
    species lifetime in hours:

    ``foot <- foot * exp(-abs(age) / tau)``

    where ``age`` is the particle transport time and ``tau`` is the lifetime.
    """

    def __init__(self, lifetime_hours: float) -> None:
        if lifetime_hours <= 0:
            raise ValueError("lifetime_hours must be > 0.")
        self.lifetime_hours = lifetime_hours

    def apply(
        self,
        particles: pd.DataFrame,
        *,
        context: ChemistryContext | None = None,
    ) -> pd.DataFrame:
        """Apply first-order lifetime decay to the particle ``foot`` column."""
        p = particles.copy()

        if "foot_before_chemistry" in p.columns:
            p["foot"] = p["foot_before_chemistry"]
            p = p.drop(columns=["foot_before_chemistry"])

        time_column = context.time_column if context is not None else "time"
        time_unit = context.time_unit if context is not None else "min"

        if time_column not in p.columns:
            raise ValueError(
                f"Particle DataFrame has no column {time_column!r} required for "
                "chemistry/lifetime weighting."
            )
        if time_unit not in _TIME_UNIT_TO_HOURS:
            raise ValueError(
                "Unsupported chemistry time unit "
                f"{time_unit!r}. Expected one of: "
                f"{', '.join(sorted(_TIME_UNIT_TO_HOURS))}."
            )

        ages = np.abs(p[time_column].to_numpy(dtype=float))
        tau = self.lifetime_hours / _TIME_UNIT_TO_HOURS[time_unit]
        scale = np.exp(-ages / float(tau))

        p["foot_before_chemistry"] = p["foot"]
        p["foot"] = p["foot"] * scale
        return p


def apply_chemistry(
    particles: pd.DataFrame,
    chemistry: ChemistryModel,
    *,
    context: ChemistryContext | None = None,
) -> pd.DataFrame:
    """Apply a chemistry model to a particle DataFrame."""
    return chemistry.apply(particles, context=context)


__all__ = [
    "ChemistryContext",
    "ChemistryModel",
    "FirstOrderLifetimeChemistry",
    "NoOpChemistry",
    "apply_chemistry",
]
