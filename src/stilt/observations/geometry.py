"""Observation geometry models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GeometryKind = Literal["point", "polygon", "ellipse", "swath_cell", "line"]


@dataclass(slots=True)
class ViewingGeometry:
    """
    Line-of-sight angles for one observation, in degrees.

    ``zenith_angle`` is measured from the local vertical. ``azimuth_angle`` is
    measured clockwise from north and is the bearing *from the ground point
    toward the instrument* (the satellite, or the sun for a solar-tracking
    spectrometer), which is the direction the line of sight rises toward. For
    a solar tracker these are the solar zenith and solar azimuth angles. Check
    your product's definition: some report the reverse bearing.
    """

    zenith_angle: float
    azimuth_angle: float

    def __post_init__(self) -> None:
        if not 0 <= self.zenith_angle < 90:
            raise ValueError("ViewingGeometry.zenith_angle must be in [0, 90) degrees.")


@dataclass(slots=True)
class HorizontalGeometry:
    """Horizontal measurement geometry for one observation."""

    kind: GeometryKind
    center_longitude: float
    center_latitude: float
    vertices: list[tuple[float, float]] = field(default_factory=list)
    major_axis_km: float | None = None
    minor_axis_km: float | None = None
    orientation_deg: float | None = None
    along_track_index: int | None = None
    across_track_index: int | None = None
    swath: int | str | None = None
    resolution_km: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
