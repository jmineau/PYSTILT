"""Point-sensor implementation for in-situ and simple point measurements."""

from __future__ import annotations

from stilt.observations.observation import Observation
from stilt.observations.receptors import build_point_receptor
from stilt.observations.sensors.base import BaseSensor
from stilt.receptor import Receptor


class PointSensor(BaseSensor):
    """
    Sensor family for point-like observations.

    This fits fixed towers, moving in-situ platforms, and other measurements
    that should normalize to a single location and map to a point receptor.
    """

    def __init__(
        self,
        *,
        name: str = "point",
        supported_species: tuple[str, ...] = (),
        default_height: float | None = None,
    ) -> None:
        self.name = name
        self.supported_species = supported_species
        self.default_height = default_height

    def build_receptor(self, observation: Observation) -> Receptor:
        """Build a point receptor for one point-like observation."""
        geometry = observation.geometry
        if geometry is not None and geometry.kind != "point":
            raise ValueError(
                "PointSensor requires point geometry or no explicit geometry. "
                f"Got {geometry.kind!r}."
            )
        return build_point_receptor(observation, altitude=self.default_height)
