"""Observation geometry models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from stilt.config import VerticalReference, validate_vertical_reference

GeometryKind = Literal["point", "polygon", "ellipse", "swath_cell", "line"]


@dataclass(slots=True)
class ViewingGeometry:
    """Solar/view geometry attached to one observation."""

    solar_zenith_angle: float | None = None
    viewing_zenith_angle: float | None = None
    solar_azimuth_angle: float | None = None
    viewing_azimuth_angle: float | None = None
    relative_azimuth_angle: float | None = None
    scan_angle: float | None = None
    glint_angle: float | None = None


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


@dataclass(slots=True)
class LineOfSight:
    """Physical line-of-sight geometry and sampling choices for one observation."""

    altitude_ref: VerticalReference = "msl"
    altitude_levels: list[float] = field(default_factory=list)
    start_altitude: float | None = None
    end_altitude: float | None = None
    count: int | None = None
    frequency: float | None = None
    anchor_altitude: float | None = None
    surface_altitude: float | None = None
    max_altitude: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.altitude_ref = validate_vertical_reference(self.altitude_ref)
        if self.altitude_levels and (
            self.start_altitude is not None
            or self.end_altitude is not None
            or self.count is not None
            or self.frequency is not None
        ):
            raise ValueError(
                "LineOfSight.altitude_levels cannot be combined with "
                "start/end/count/frequency sampling arguments."
            )
        if self.altitude_levels:
            return
        if (self.start_altitude is None) != (self.end_altitude is None):
            raise ValueError(
                "LineOfSight requires both start_altitude and end_altitude "
                "when explicit altitude_levels are not provided."
            )
        if self.start_altitude is None and self.end_altitude is None:
            raise ValueError(
                "LineOfSight requires altitude_levels or a start/end altitude range."
            )
        if (self.count is None) == (self.frequency is None):
            raise ValueError(
                "LineOfSight requires exactly one of count or frequency when "
                "sampling from a start/end altitude range."
            )
