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
    "integration",
    "tccon",
]


@dataclass(slots=True)
class VerticalOperator:
    """Vertical weighting or averaging-kernel operator for one observation."""

    mode: VerticalOperatorMode
    levels: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    pressure_levels: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
