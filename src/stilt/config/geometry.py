"""
Geometry settings for footprints.

A footprint's ``geometry`` names the polygons it will be aggregated to: a
vector file, H3 hexagons, or windows around points. Each spec's ``build``
method returns a :class:`stilt.Mesh`. When ``grid`` is not given, the grid
is derived from the geometry with :meth:`stilt.Grid.from_geometry`.

.. code-block:: yaml

   geometry:                     # the default footprint's geometry
     kind: file
     path: counties.shp
     ids: NAME

   variants:
     hrrr: {}
     hrrr-hexes:                   # a second footprint from the same particles
       from: hrrr
       geometry: {kind: h3, resolution: 8, bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}}
       cells_per_target: 4
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .spatial import Bounds

if TYPE_CHECKING:
    from stilt.geometry import Mesh


class FileGeometrySpec(BaseModel):
    """Polygons read from a vector file such as a shapefile or GeoPackage."""

    model_config = ConfigDict(frozen=True)

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

    def build(self) -> Mesh:
        """Read the file into a :class:`stilt.Mesh`. Requires geopandas."""
        from stilt.geometry import Mesh

        kwargs = {}
        if self.layer is not None:
            kwargs["layer"] = self.layer
        if self.where is not None:
            kwargs["where"] = self.where
        return Mesh.from_file(self.path, ids=self.ids, **kwargs)


class H3GeometrySpec(BaseModel):
    """H3 hexagons of one resolution covering a longitude/latitude box."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["h3"] = "h3"
    resolution: int = Field(..., description="H3 resolution (0-15).", ge=0, le=15)
    bounds: Bounds = Field(..., description="Longitude/latitude box to cover.")

    def build(self) -> Mesh:
        """Build the hexagons as a :class:`stilt.Mesh`. Requires ``h3``."""
        from stilt.geometry import Mesh

        return Mesh.from_h3(self.resolution, self.bounds)


class WindowsGeometrySpec(BaseModel):
    """Rectangular windows centered on points, such as known point sources."""

    model_config = ConfigDict(frozen=True)

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

    def build(self) -> Mesh:
        """Build the windows as a :class:`stilt.Mesh`."""
        from stilt.geometry import Mesh

        return Mesh.from_windows(self.coords, self.size, ids=self.ids, crs=self.crs)


GeometrySpec = Annotated[
    FileGeometrySpec | H3GeometrySpec | WindowsGeometrySpec,
    Field(discriminator="kind"),
]
"""Any geometry spec accepted by ``FootprintConfig.geometry``."""


__all__ = [
    "FileGeometrySpec",
    "GeometrySpec",
    "H3GeometrySpec",
    "WindowsGeometrySpec",
]
