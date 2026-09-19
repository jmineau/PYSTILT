"""Observation vertical-operator models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VerticalOperatorMode = Literal[
    "none",
    "uniform",
    "ak",
    "pwf",
    "ak_pwf",
]


@dataclass(slots=True)
class VerticalOperator:
    """
    Vertical weighting or averaging-kernel operator for one observation.

    Parameters
    ----------
    mode:
        ``"none"`` / ``"uniform"`` leave particle weights alone. ``"ak"``
        multiplies each particle's ``foot`` by a normalized averaging kernel
        interpolated from ``levels`` / ``values``. ``"pwf"`` weights each
        particle by the fraction of the column's air mass it represents,
        derived from the particles' own release pressures. ``"ak_pwf"`` does
        both.
    levels, values:
        Averaging-kernel profile (dimensionless ``values`` at vertical
        ``levels``). Required for ``"ak"`` and ``"ak_pwf"``; ignored otherwise.
        The coordinate of ``levels`` is chosen where the operator is applied
        (release height AGL in metres by default, or pressure in hPa).
    surface_pressure:
        Surface pressure in hPa used as the bottom of the column for the
        pressure weighting. When omitted it is estimated from the particles'
        first-step pressures and heights.
    """

    mode: VerticalOperatorMode
    levels: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    surface_pressure: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
