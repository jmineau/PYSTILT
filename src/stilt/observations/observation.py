"""Normalized observation model."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from stilt.config import VerticalReference, validate_vertical_reference

from .geometry import HorizontalGeometry, LineOfSight, ViewingGeometry
from .operators import VerticalOperator
from .uncertainty import UncertaintyBudget


@dataclass(slots=True)
class Observation:
    """One normalized measurement record independent of raw file format.

    Observations can be constructed directly by external parsers or via
    ``Sensor.make_observation()`` helpers in ``stilt.observations``.
    """

    sensor: str
    species: str
    time: pd.Timestamp | dt.datetime | str
    latitude: float
    longitude: float
    value: float | None = None
    units: str | None = None
    uncertainty: float | None = None
    uncertainty_budget: UncertaintyBudget | None = None
    observation_id: str | None = None
    platform: str | None = None
    altitude: float | None = None
    altitude_ref: VerticalReference = "agl"
    geometry: HorizontalGeometry | None = None
    line_of_sight: LineOfSight | None = None
    viewing: ViewingGeometry | None = None
    operator: VerticalOperator | None = None
    quality: dict[str, float | int | bool] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = pd.Timestamp(self.time)
        if pd.isna(parsed):
            raise ValueError("Observation.time must be a valid timestamp.")
        self.time = cast(pd.Timestamp, parsed)
        self.altitude_ref = validate_vertical_reference(self.altitude_ref)
        if self.uncertainty is None and self.uncertainty_budget is not None:
            self.uncertainty = self.uncertainty_budget.total
