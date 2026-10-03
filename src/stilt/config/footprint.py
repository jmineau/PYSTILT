"""
Footprint settings.

A footprint's ``grid`` is the raster it is computed on. Its ``geometry``,
when given, names the polygons it will be aggregated to: a vector file, H3
hexagons, or windows around points. Each geometry spec's ``build`` method
returns a :class:`stilt.Mesh`, and without a ``grid`` the grid is derived
from the mesh with :meth:`stilt.Mesh.to_grid`.

Loading a config does not read the geometry. The mesh is built the first
time a variant's settings are resolved (:meth:`FootprintConfig.resolve`),
and the grid and hash derived from it are stored with the footprints.

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

from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from stilt.spatial import Bounds, Grid
from stilt.transforms import dump_transform, load_transform

if TYPE_CHECKING:
    from stilt.spatial import Mesh


class _GeometrySpec(BaseModel):
    """Base of the geometry specs: a frozen description that builds a mesh."""

    model_config = ConfigDict(frozen=True)

    def build(self) -> Mesh:
        """Build the polygons as a :class:`stilt.Mesh`, reading the source now."""
        raise NotImplementedError

    @cached_property
    def mesh(self) -> Mesh:
        """
        The polygons, built by :meth:`build` on first use and kept.

        Variants that inherit the same geometry share one spec, so they
        share one build.
        """
        return self.build()


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

    def build(self) -> Mesh:
        """Read the file into a :class:`stilt.Mesh`. Requires geopandas."""
        from stilt.spatial import Mesh

        kwargs = {}
        if self.layer is not None:
            kwargs["layer"] = self.layer
        if self.where is not None:
            kwargs["where"] = self.where
        return Mesh.from_file(self.path, ids=self.ids, **kwargs)


class H3GeometrySpec(_GeometrySpec):
    """H3 hexagons of one resolution covering a longitude/latitude box."""

    kind: Literal["h3"] = "h3"
    resolution: int = Field(..., description="H3 resolution (0-15).", ge=0, le=15)
    bounds: Bounds = Field(..., description="Longitude/latitude box to cover.")

    def build(self) -> Mesh:
        """Build the hexagons as a :class:`stilt.Mesh`. Requires ``h3``."""
        from stilt.spatial import Mesh

        return Mesh.from_h3(self.resolution, self.bounds)


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

    def build(self) -> Mesh:
        """Build the windows as a :class:`stilt.Mesh`."""
        from stilt.spatial import Mesh

        return Mesh.from_windows(self.coords, self.size, ids=self.ids, crs=self.crs)


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
    to, and the grid is derived from them with
    :meth:`stilt.Mesh.to_grid` when the settings are resolved
    (:meth:`resolve`), not when they are loaded. When both are given,
    ``grid`` is used as is and ``geometry`` is kept with the footprint.
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
    geometry_hash: str | None = Field(
        None,
        description=(
            "Hash of the built ``geometry`` (``Mesh.hash``), used to tell whether "
            "the geometry changed after a footprint was made. Filled in "
            "when the settings are resolved."
        ),
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

    def resolve(self) -> Self:
        """
        Return these settings with ``grid`` and ``geometry_hash`` filled in from ``geometry``.

        Loading settings does not read the geometry. This method builds the
        mesh, derives the grid when none was given, and records the mesh
        hash. Each geometry spec builds its mesh once and keeps it (its
        ``mesh`` attribute). Settings with nothing to fill in are returned
        unchanged, so settings read back from a footprint folder never read
        the geometry source.

        Returns
        -------
        FootprintConfig
            The settings with ``grid`` and ``geometry_hash`` set whenever
            ``geometry`` is.

        Examples
        --------
        >>> spec = {"kind": "windows", "coords": [(-111.97, 40.515)], "size": 0.01}
        >>> FootprintConfig(geometry=spec).grid is None
        True
        >>> FootprintConfig(geometry=spec).resolve().grid.xres
        0.002
        """
        if self.geometry is None:
            return self
        update: dict[str, Any] = {}
        if self.grid is None:
            update["grid"] = self.geometry.mesh.to_grid(
                cells_per_target=self.cells_per_target
            )
        if self.geometry_hash is None:
            update["geometry_hash"] = self.geometry.mesh.hash
        return self.model_copy(update=update) if update else self

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
