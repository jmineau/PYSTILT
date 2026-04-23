"""Spatial config models."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from .fields import cfg_field

VerticalReference = Literal["agl", "msl"]


def validate_vertical_reference(reference: str) -> VerticalReference:
    """Return a normalized vertical reference or raise for invalid input."""
    normalized = reference.lower()
    if normalized not in {"agl", "msl"}:
        raise ValueError(
            f"Vertical reference must be 'agl' or 'msl'. Got {reference!r}."
        )
    return cast(VerticalReference, normalized)


def kmsl_from_vertical_reference(reference: VerticalReference) -> int:
    """Map a vertical reference onto the HYSPLIT ``KMSL`` control value."""
    return 0 if reference == "agl" else 1


class Bounds(BaseModel):
    """Immutable geographic bounding box used for spatial subsetting."""

    model_config = ConfigDict(frozen=True)

    xmin: float = cfg_field(
        ...,
        description="Minimum x-coordinate (longitude for geographic CRS).",
    )
    xmax: float = cfg_field(
        ...,
        description="Maximum x-coordinate (longitude for geographic CRS).",
    )
    ymin: float = cfg_field(
        ...,
        description="Minimum y-coordinate (latitude for geographic CRS).",
    )
    ymax: float = cfg_field(
        ...,
        description="Maximum y-coordinate (latitude for geographic CRS).",
    )
    projection: str = cfg_field(
        "+proj=longlat",
        description="PROJ string defining the coordinate reference system.",
    )


class Grid(Bounds):
    """Gridded spatial domain for footprint computation."""

    model_config = ConfigDict(frozen=True)

    xres: float = cfg_field(
        ...,
        description="Cell width in the projection's x units (degrees for geographic CRS).",
    )
    yres: float = cfg_field(
        ...,
        description="Cell height in the projection's y units (degrees for geographic CRS).",
    )

    @property
    def resolution(self) -> str:
        """Human-readable cell resolution string, e.g. ``'0.01x0.01'``."""
        return f"{self.xres}x{self.yres}"


__all__ = ["Bounds", "Grid"]
