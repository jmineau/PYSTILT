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

VERTICAL_OPERATOR_MODES: frozenset[str] = frozenset(
    ("none", "uniform", "ak", "pwf", "ak_pwf")
)

# Modes that existed before 0.1.0a10, mapped to their replacement, so an
# upgrading caller gets a pointed error instead of a silent no-op.
_RETIRED_MODES: dict[str, str] = {
    "integration": "pwf",
    "tccon": "ak_pwf",
}


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

    def __post_init__(self) -> None:
        """
        Reject unknown modes at construction.

        A dataclass does not enforce its ``Literal`` annotation, so without
        this an unrecognised mode would pass silently through
        ``apply_vertical_operator`` leaving ``foot`` unweighted — while still
        adding ``foot_before_weight``, so it would look like weighting had
        been applied.
        """
        if self.mode in VERTICAL_OPERATOR_MODES:
            return
        replacement = _RETIRED_MODES.get(str(self.mode))
        if replacement is not None:
            raise ValueError(
                f"The {self.mode!r} vertical-operator mode was removed in "
                f"0.1.0a10. Use {replacement!r} instead, folding any "
                "instrument-specific factor (for example TCCON's wet-air "
                "scaling) into 'values'. The pressure weighting is now "
                "derived from the particles, so 'levels' and 'values' hold "
                "only the averaging kernel."
            )
        raise ValueError(
            f"Unknown vertical-operator mode {self.mode!r}. "
            f"Valid modes: {', '.join(sorted(VERTICAL_OPERATOR_MODES))}."
        )
