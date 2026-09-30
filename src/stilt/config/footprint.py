"""Footprint settings."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from stilt.transforms import dump_transform, load_transform

from .geometry import GeometrySpec
from .spatial import Grid

_GEOMETRY_ADAPTER: TypeAdapter[Any] = TypeAdapter(GeometrySpec)


class FootprintConfig(BaseModel):
    """
    Footprint settings: the grid, smoothing, and particle transforms.

    The config and each variant hold these as their defaults and overrides,
    and every footprint keeps the settings it was calculated with.

    ``grid`` is the raster the footprint is computed on. Leave both ``grid``
    and ``geometry`` unset for a variant that only produces trajectories.
    Give ``geometry`` to name the polygons the footprint will be aggregated
    to, and the grid is derived from them with
    :meth:`stilt.Grid.from_geometry`. When both are given, ``grid`` is used
    as is and ``geometry`` is kept with the footprint.
    """

    grid: Grid | None = Field(
        None,
        description=(
            "Domain and resolution of the footprint. Leaving it unset with no "
            "``geometry`` gives a run that produces trajectories only."
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
            "automatically when ``geometry`` is set."
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

    @model_validator(mode="before")
    @classmethod
    def _derive_from_geometry(cls, data: Any) -> Any:
        """
        Fill ``grid`` and ``geometry_hash`` from ``geometry`` when they are missing.

        The geometry is built only when one of them is missing, so reloading a
        stored config never reads the geometry source.
        """
        if not isinstance(data, dict):
            return data
        spec_raw = data.get("geometry")
        if spec_raw is None:
            return data
        need_grid = data.get("grid") is None
        need_hash = data.get("geometry_hash") is None
        if not (need_grid or need_hash):
            return data
        spec = (
            spec_raw
            if hasattr(spec_raw, "build")
            else _GEOMETRY_ADAPTER.validate_python(spec_raw)
        )
        mesh = spec.build()
        out = dict(data)
        if need_grid:
            cells = float(data.get("cells_per_target", 4.0))
            out["grid"] = Grid.from_geometry(mesh, cells_per_target=cells)
        if need_hash:
            out["geometry_hash"] = mesh.hash
        return out

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
