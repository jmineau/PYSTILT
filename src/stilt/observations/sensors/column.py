"""Column-sensor implementation for vertical and slanted column observations."""

from __future__ import annotations

from typing import Literal

from stilt.config import VerticalReference
from stilt.observations.observation import Observation
from stilt.observations.receptors import (
    build_column_receptor,
    build_slant_receptor,
)
from stilt.observations.sensors.base import BaseSensor
from stilt.receptors import Receptor

ColumnMode = Literal["vertical", "slant"]


class ColumnSensor(BaseSensor):
    """Sensor family for column and slant-column observation geometries."""

    def __init__(
        self,
        *,
        name: str = "column",
        supported_species: tuple[str, ...] = (),
        mode: ColumnMode = "vertical",
        bottom: float | None = None,
        top: float | None = None,
        altitude_ref: VerticalReference = "agl",
    ) -> None:
        self.name = name
        self.supported_species = supported_species
        self.mode = mode
        self.bottom = bottom
        self.top = top
        self.altitude_ref: VerticalReference = altitude_ref

    def build_receptor(self, observation: Observation) -> Receptor:
        """Build a vertical or slanted column receptor for one observation."""
        if self.mode == "vertical":
            if self.bottom is None or self.top is None:
                raise ValueError(
                    "ColumnSensor(mode='vertical') requires bottom and top heights."
                )
            return build_column_receptor(
                observation,
                bottom=self.bottom,
                top=self.top,
                altitude_ref=self.altitude_ref,
            )

        if self.mode == "slant":
            return build_slant_receptor(observation)

        raise ValueError(f"Unknown column sensor mode: {self.mode!r}")
