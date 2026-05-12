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
    """Immutable geographic bounding box (always lon/lat degrees)."""

    model_config = ConfigDict(frozen=True)

    xmin: float = cfg_field(..., description="Western longitude (degrees).")
    xmax: float = cfg_field(..., description="Eastern longitude (degrees).")
    ymin: float = cfg_field(..., description="Southern latitude (degrees).")
    ymax: float = cfg_field(..., description="Northern latitude (degrees).")


class Grid(Bounds):
    """Footprint grid: lon/lat bounds, optional output projection, cell resolution."""

    model_config = ConfigDict(frozen=True)

    xres: float = cfg_field(
        ...,
        description="Cell width in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    yres: float = cfg_field(
        ...,
        description="Cell height in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    projection: str = cfg_field(
        "+proj=longlat",
        description=(
            "Output CRS as a PROJ string.  Bounds are always lon/lat; "
            "particles and bounds are projected to this CRS before gridding."
        ),
    )

    @property
    def resolution(self) -> str:
        """Human-readable cell resolution string, e.g. ``'0.01x0.01'``."""
        return f"{self.xres}x{self.yres}"


__all__ = ["Bounds", "Grid"]
