"""Generic observation-level uncertainty models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class UncertaintyComponent:
    """One named uncertainty term attached to an observation."""

    name: str
    sigma: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("UncertaintyComponent.sigma must be >= 0.")


@dataclass(frozen=True, slots=True)
class UncertaintyBudget:
    """Portable collection of named observation-uncertainty components."""

    components: tuple[UncertaintyComponent, ...] = ()
    units: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("UncertaintyBudget component names must be unique.")

    @property
    def total(self) -> float:
        """Root-sum-square total of all component sigmas."""
        return math.sqrt(sum(component.sigma**2 for component in self.components))

    def component(self, name: str) -> UncertaintyComponent | None:
        """Return one named uncertainty component if present."""
        for component in self.components:
            if component.name == name:
                return component
        return None

    def to_mapping(self) -> dict[str, float]:
        """Return a simple ``name -> sigma`` mapping."""
        return {component.name: component.sigma for component in self.components}

    @classmethod
    def from_mapping(
        cls,
        components: Mapping[str, float],
        *,
        units: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UncertaintyBudget:
        """Build a budget from a simple ``name -> sigma`` mapping."""
        return cls(
            components=tuple(
                UncertaintyComponent(name=name, sigma=sigma)
                for name, sigma in components.items()
            ),
            units=units,
            metadata=dict(metadata or {}),
        )


__all__ = [
    "UncertaintyBudget",
    "UncertaintyComponent",
]
