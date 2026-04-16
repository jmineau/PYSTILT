"""Base sensor interfaces for normalized observation workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from stilt.observations.observation import Observation
from stilt.observations.scenes import Scene, make_scene
from stilt.receptor import Receptor


class Sensor(Protocol):
    """Behavioral interface for sensor families in the observation layer."""

    name: str
    supported_species: tuple[str, ...]

    def make_observation(
        self,
        *,
        species: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Build one normalized observation for this sensor."""
        ...

    def build_receptor(self, observation: Observation) -> Receptor:
        """Build the transport receptor used for one observation."""
        ...

    def group_scenes(self, observations: Sequence[Observation]) -> list[Scene]:
        """Group sensor observations into retrieval scenes."""
        ...


class BaseSensor:
    """Small base class for sensors that construct observations and receptors."""

    name: str = "sensor"
    supported_species: tuple[str, ...] = ()

    def _resolve_species(self, species: str | None) -> str:
        """Resolve and validate the observation species for this sensor."""
        if species is None:
            if len(self.supported_species) == 1:
                return self.supported_species[0]
            raise ValueError(
                f"{type(self).__name__} requires `species=` unless exactly one "
                "supported species is configured."
            )
        if self.supported_species and species not in self.supported_species:
            raise ValueError(
                f"{type(self).__name__} does not support species {species!r}. "
                f"Supported species: {self.supported_species!r}."
            )
        return species

    def make_observation(
        self,
        *,
        species: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Construct one normalized observation stamped with this sensor name.

        This is a lightweight normalization helper for code that already has
        structured values in hand. Product-specific file readers remain
        intentionally outside core ``stilt.observations``.
        """
        if "sensor" in kwargs:
            raise TypeError("make_observation() sets `sensor` automatically.")
        if "species" in kwargs:
            raise TypeError("Pass `species=` directly to make_observation().")
        return Observation(
            sensor=self.name,
            species=self._resolve_species(species),
            **kwargs,
        )

    def build_receptor(self, observation: Observation) -> Receptor:
        """Return the receptor corresponding to one observation."""
        raise NotImplementedError

    def group_scenes(self, observations: Sequence[Observation]) -> list[Scene]:
        """Group observations into scenes.

        The default behavior is intentionally conservative: keep the given
        collection together as a single scene.
        """
        observations = list(observations)
        if not observations:
            return []
        return [make_scene(observations, sensor=self.name)]
