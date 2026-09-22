"""
Declarative geometry specs for footprint configs.

A :class:`~stilt.config.FootprintConfig` may name the *state geometry* it is
meant to serve (a shapefile, H3 hexagons, point windows) instead of, or as
well as, its native raster ``grid``.  The spec is plain YAML-able data; its
:meth:`build` method returns the live :class:`stilt.Mesh`, and when ``grid``
is omitted the config derives one with :meth:`stilt.Grid.from_geometry`.

.. code-block:: yaml

   footprints:
     counties:
       geometry:
         kind: file
         path: counties.shp
         ids: NAME
     hexes:
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
    """Polygons read from a vector file (shapefile, GeoPackage, GeoJSON)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["file"] = "file"
    path: str = Field(..., description="Path to the vector file.")
    ids: str | None = Field(
        None, description="Attribute column to use as cell ids (default: row index)."
    )
    layer: str | None = Field(None, description="Layer name for multi-layer files.")
    where: str | None = Field(
        None, description="Optional attribute filter (OGR SQL WHERE clause)."
    )

    def build(self) -> Mesh:
        """Read the file into a :class:`stilt.Mesh` (requires geopandas)."""
        from stilt.geometry import Mesh

        kwargs = {}
        if self.layer is not None:
            kwargs["layer"] = self.layer
        if self.where is not None:
            kwargs["where"] = self.where
        return Mesh.from_file(self.path, ids=self.ids, **kwargs)


class H3GeometrySpec(BaseModel):
    """H3 hexagons of one resolution covering a lon/lat bounding box."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["h3"] = "h3"
    resolution: int = Field(..., description="H3 resolution (0-15).", ge=0, le=15)
    bounds: Bounds = Field(..., description="Lon/lat box the hexagons must cover.")

    def build(self) -> Mesh:
        """Generate the hexagons (requires the ``h3`` package)."""
        from stilt.geometry import Mesh

        return Mesh.from_h3(self.resolution, self.bounds)


class WindowsGeometrySpec(BaseModel):
    """Rectangular windows centred on points, e.g. named point sources."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["windows"] = "windows"
    coords: list[tuple[float, float]] = Field(
        ..., description="Window centres as (x, y) pairs."
    )
    size: float | tuple[float, float] = Field(
        ..., description="Window width, or (width, height), in CRS units."
    )
    ids: list[str] | None = Field(None, description="Optional label per point.")
    crs: str = Field("+proj=longlat", description="CRS of the coordinates.")

    def build(self) -> Mesh:
        """Build the windows as a :class:`stilt.Mesh`."""
        from stilt.geometry import Mesh

        return Mesh.from_windows(self.coords, self.size, ids=self.ids, crs=self.crs)


GeometrySpec = Annotated[
    FileGeometrySpec | H3GeometrySpec | WindowsGeometrySpec,
    Field(discriminator="kind"),
]
"""Any declarative geometry accepted by ``FootprintConfig.geometry``."""


__all__ = [
    "FileGeometrySpec",
    "GeometrySpec",
    "H3GeometrySpec",
    "WindowsGeometrySpec",
]
