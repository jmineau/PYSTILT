"""Footprint settings."""

from __future__ import annotations

from typing import Any, Self

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
)

from stilt.transforms import dump_transform, load_transform

from .geometry import GeometrySpec
from .spatial import Grid


class FootprintConfig(BaseModel):
    """
    Footprint settings: the grid, smoothing, and particle transforms.

    The config and each variant hold these as their defaults and overrides,
    and every footprint keeps the settings it was calculated with.

    ``grid`` is the raster the footprint is computed on. Leave both ``grid``
    and ``geometry`` unset for a variant that only produces particles.
    Give ``geometry`` to name the polygons the footprint will be aggregated
    to, and the grid is derived from them with
    :meth:`stilt.Grid.from_geometry` when the settings are resolved
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
            update["grid"] = Grid.from_geometry(
                self.geometry.mesh, cells_per_target=self.cells_per_target
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


__all__ = ["FootprintConfig"]
