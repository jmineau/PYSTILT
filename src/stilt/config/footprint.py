"""Footprint settings: the fields a variant carries and the resolved product."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
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


class FootprintParams(BaseModel):
    """
    Footprint settings as they appear in a config: defaults and per-variant.

    ``grid`` is the native raster. Leave it ``None`` (and give no
    ``geometry``) for a variant that produces only a trajectory. Give
    ``geometry`` instead to name the state geometry the footprint serves and
    let the raster be derived from it (:meth:`stilt.Grid.from_geometry`); when
    both are given the explicit ``grid`` wins and ``geometry`` is kept as a
    record.
    """

    grid: Grid | None = Field(
        None,
        description=(
            "Spatial domain and resolution of the footprint. ``None`` with no "
            "``geometry`` means no footprint: a trajectory-only run."
        ),
    )
    geometry: GeometrySpec | None = Field(
        None,
        description=(
            "State geometry this footprint serves (file, h3, windows). Used to "
            "derive ``grid`` when that is omitted, and recorded for aggregation."
        ),
    )
    cells_per_target: float = Field(
        default=4.0,
        description="Native cells across the smallest geometry cell when deriving ``grid``.",
        gt=0,
    )
    geometry_hash: str | None = Field(
        None,
        description=(
            "Content hash of the built ``geometry`` (``Mesh.hash``), recorded so "
            "a stored footprint can detect that the geometry file changed later. "
            "Filled automatically; not needed when ``geometry`` is unset."
        ),
    )
    smooth_factor: float = Field(
        1.0,
        description="Factor by which to linearly scale footprint smoothing. Defaults to 1",
    )
    time_integrate: bool = Field(
        False,
        description="If True, sum the footprint over all time steps to produce a single 2-D layer.",
    )
    transforms: list[Any] = Field(
        description=(
            "Particle transforms applied in order before rasterizing the footprint. "
            "Each entry is a built-in kind (averaging_kernel, pressure_weighting, "
            "first_order_lifetime) or a dotted import path to a user transform class."
        ),
        default_factory=list,
    )

    #: The footprint fields, the only ones a derived variant may override.
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "grid",
            "geometry",
            "cells_per_target",
            "geometry_hash",
            "smooth_factor",
            "time_integrate",
            "transforms",
        }
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_from_geometry(cls, data: Any) -> Any:
        """
        Build the geometry once to fill ``grid`` (when omitted) and ``geometry_hash``.

        Nothing is built when there is no ``geometry``, or when both ``grid``
        and ``geometry_hash`` are already present (e.g. reloading a stored
        config), so reading a footprint never touches the geometry source.
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
        """Serialise the transforms back to plain mappings."""
        return [dump_transform(item) for item in value]

    @property
    def footprint(self) -> FootprintConfig | None:
        """The footprint product these settings describe, or ``None`` without a grid."""
        if self.grid is None:
            return None
        return FootprintConfig(
            grid=self.grid,
            geometry=self.geometry,
            cells_per_target=self.cells_per_target,
            geometry_hash=self.geometry_hash,
            smooth_factor=self.smooth_factor,
            time_integrate=self.time_integrate,
            transforms=list(self.transforms),
        )


class FootprintConfig(FootprintParams):
    """
    One footprint product: :class:`FootprintParams` with the grid resolved.

    This is what :meth:`stilt.Footprint.calculate` takes and what a stored
    footprint's attributes round-trip to.
    """

    model_config = ConfigDict(frozen=True)

    grid: Grid = Field(
        ..., description="Spatial domain and resolution of the footprint."
    )

    def replace(self, **updates: object) -> FootprintConfig:
        """Return a copy with updated fields for interactive iteration."""
        return self.model_copy(update=updates)


__all__ = ["FootprintConfig", "FootprintParams"]
