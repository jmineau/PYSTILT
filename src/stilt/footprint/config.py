"""
Footprint config.

A footprint's ``grid`` is the raster it is computed on. Its ``geometry``,
when given, names the polygons it will be aggregated to: a vector file, H3
hexagons, or windows around points. Without a ``grid``, the grid is derived
from the geometry (:meth:`stilt.Mesh.from_spec`, then
:meth:`stilt.Mesh.to_grid`).

Loading a config does not read the geometry. ``project.variants`` reads it
(:meth:`stilt.ProjectConfig.resolve`), and the grid and the geometry's hash are
recorded with the footprints.

.. code-block:: yaml

   geometry:                     # the default footprint's geometry
     kind: file
     path: counties.shp
     ids: NAME

   variants:
     hrrr: {}
     hrrr-hexes:                   # a second footprint from the same particles
       geometry: {kind: h3, resolution: 8, bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}}
       cells_per_target: 4
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from stilt.spatial import Bounds, Grid
from stilt.transforms import dump_transform, load_transform


class _GeometrySpec(BaseModel):
    """Base of the geometry specs: a frozen description of the polygons."""

    model_config = ConfigDict(frozen=True)


class FileGeometrySpec(_GeometrySpec):
    """Polygons read from a vector file such as a shapefile or GeoPackage."""

    kind: Literal["file"] = "file"
    path: str = Field(..., description="Path to the vector file.")
    ids: str | None = Field(
        None,
        description="Attribute column holding the cell ids. Unset uses the row number.",
    )
    layer: str | None = Field(
        None, description="Layer to read from a multi-layer file."
    )
    where: str | None = Field(
        None, description="Attribute filter as an OGR SQL WHERE clause."
    )


class H3GeometrySpec(_GeometrySpec):
    """H3 hexagons of one resolution covering a longitude/latitude box."""

    kind: Literal["h3"] = "h3"
    resolution: int = Field(..., description="H3 resolution (0-15).", ge=0, le=15)
    bounds: Bounds = Field(..., description="Longitude/latitude box to cover.")


class WindowsGeometrySpec(_GeometrySpec):
    """Rectangular windows centered on points, such as known point sources."""

    kind: Literal["windows"] = "windows"
    coords: list[tuple[float, float]] = Field(
        ..., description="Window centers as (x, y) pairs in ``crs`` units."
    )
    size: float | tuple[float, float] = Field(
        ..., description="Window width, or (width, height), in ``crs`` units."
    )
    ids: list[str] | None = Field(None, description="Label for each window.")
    crs: str = Field(
        "+proj=longlat", description="Coordinate reference system of ``coords``."
    )


GeometrySpec = Annotated[
    FileGeometrySpec | H3GeometrySpec | WindowsGeometrySpec,
    Field(discriminator="kind"),
]
"""Any geometry spec accepted by ``FootprintConfig.geometry``."""


class FootprintConfig(BaseModel):
    """
    Footprint settings: the grid, smoothing, and particle transforms.

    The config and each variant hold these as their defaults and overrides,
    and every footprint keeps the settings it was calculated with.

    ``grid`` is the raster the footprint is computed on. Leave both ``grid``
    and ``geometry`` unset for a variant that only produces particles.
    Give ``geometry`` to name the polygons the footprint will be aggregated
    to, and the grid is derived from them (:meth:`stilt.Mesh.to_grid`) when
    the project's variants are resolved, not when the config loads. When
    both are given, ``grid`` is used as is and ``geometry`` is recorded with
    the footprints.
    """

    grid: Grid | None = Field(
        None,
        description=(
            "Domain and resolution of the footprint. Unset with ``geometry`` "
            "derives it from the geometry; unset without it gives a run that "
            "produces particles only."
        ),
    )
    geometry: GeometrySpec | None = Field(
        None,
        description=(
            "Polygons the footprint will be aggregated to (``kind`` of "
            "``file``, ``h3``, or ``windows``). Used to derive ``grid`` when it "
            "is unset, and stored with the footprint."
        ),
    )
    cells_per_target: float = Field(
        default=4.0,
        description="Grid cells across the smallest ``geometry`` cell when ``grid`` is derived.",
        gt=0,
    )
    smooth_factor: float = Field(
        1.0,
        description=(
            "Factor on the width of the Gaussian smoothing kernel. 0 turns smoothing off."
        ),
    )
    time_integrate: bool = Field(
        False,
        description="Sum the footprint over time into a single layer instead of hourly layers.",
    )
    transforms: list[Any] = Field(
        description=(
            "Particle transforms applied in order before the footprint is "
            "computed. Each entry's ``kind`` is a built-in name "
            "(``averaging_kernel``, ``pressure_weighting``, "
            "``first_order_lifetime``) or the import path of your own class."
        ),
        default_factory=list,
    )

    @field_validator("transforms", mode="before")
    @classmethod
    def _load_transforms(cls, value: Any) -> list[Any]:
        """Build transform objects from their configured mappings."""
        if value is None:
            return []
        return [load_transform(item) for item in value]

    @field_serializer("transforms")
    def _dump_transforms(self, value: list[Any]) -> list[dict[str, Any]]:
        """Serialize the transforms to plain mappings."""
        return [dump_transform(item) for item in value]


__all__ = [
    "FileGeometrySpec",
    "FootprintConfig",
    "GeometrySpec",
    "H3GeometrySpec",
    "WindowsGeometrySpec",
]
