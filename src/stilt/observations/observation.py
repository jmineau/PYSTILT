"""Normalized observation model."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from stilt.config import VerticalReference, validate_vertical_reference

from .geometry import HorizontalGeometry, LineOfSight, ViewingGeometry


@dataclass(slots=True)
class Observation:
    """
    One normalized measurement record independent of raw file format.

    Product readers live outside PYSTILT; they produce these. Anything the
    core does not model (retrieval quality flags, orbit numbers, an
    uncertainty decomposition) goes in ``quality`` or ``metadata``.

    ``transforms`` carries per-observation particle transforms — typically the
    retrieval's own :class:`~stilt.transforms.AveragingKernel` — for the caller
    to pass to :meth:`stilt.Simulation.generate_footprint`.
    """

    sensor: str
    species: str
    time: pd.Timestamp | dt.datetime | str
    latitude: float
    longitude: float
    value: float | None = None
    units: str | None = None
    uncertainty: float | None = None
    observation_id: str | None = None
    platform: str | None = None
    altitude: float | None = None
    altitude_ref: VerticalReference = "agl"
    geometry: HorizontalGeometry | None = None
    line_of_sight: LineOfSight | None = None
    viewing: ViewingGeometry | None = None
    transforms: list[Any] = field(default_factory=list)
    quality: dict[str, float | int | bool] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = pd.Timestamp(self.time)
        if parsed is pd.NaT:
            raise ValueError("Observation.time must be a valid timestamp.")
        self.time = cast(pd.Timestamp, parsed)
        self.altitude_ref = validate_vertical_reference(self.altitude_ref)
