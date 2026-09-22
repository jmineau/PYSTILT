"""Footprint config models."""

from __future__ import annotations

from typing import Any

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


class FootprintConfig(BaseModel):
    """
    Settings for a single named footprint product.

    Give ``grid`` to set the native raster directly, or ``geometry`` to name
    the state geometry the footprint serves and let the raster be derived
    from it (:meth:`stilt.Grid.from_geometry`).  When both are given the
    explicit ``grid`` wins and ``geometry`` is kept as a record.
    """

    model_config = ConfigDict(frozen=True)

    grid: Grid = Field(
        ...,
        description=(
            "Spatial domain and resolution for the footprint. May be omitted "
            "when ``geometry`` is given, in which case it is derived."
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

    smooth_factor: float = Field(
        1.0,
        description="Factor by which to linearly scale footprint smoothing. Defaults to 1",
    )
    time_integrate: bool = Field(
        False,
        description="If True, sum the footprint over all time steps to produce a single 2-D layer.",
    )
    error: bool = Field(
        default=False,
        description=(
            "If True, also compute an error footprint from the error trajectories "
            'and store it alongside the main footprint under "{name}_error".'
        ),
    )
    transforms: list[Any] = Field(
        description=(
            "Particle transforms applied in order before rasterizing the footprint. "
            "Each entry is a built-in kind (averaging_kernel, pressure_weighting, "
            "first_order_lifetime) or a dotted import path to a user transform class."
        ),
        default_factory=list,
    )

    @field_validator("transforms", mode="before")
    @classmethod
    def _load_transforms(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        return [load_transform(item) for item in value]

    @field_serializer("transforms")
    def _dump_transforms(self, value: list[Any]) -> list[dict[str, Any]]:
        return [dump_transform(item) for item in value]

    def replace(self, **updates: object) -> FootprintConfig:
        """Return a copy with updated fields for interactive iteration."""
        return self.model_copy(update=updates)


def foot_names(foot_configs: dict[str, FootprintConfig]) -> list[str]:
    """Return all requested footprint output names, including error outputs."""
    names: list[str] = []
    for name, cfg in foot_configs.items():
        names.append(name)
        if cfg.error:
            names.append(f"{name}_error")
    return names


__all__ = [
    "FootprintConfig",
    "foot_names",
]
