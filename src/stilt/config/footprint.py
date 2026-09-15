"""Footprint config models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from .fields import cfg_field
from .geometry import GeometrySpec
from .spatial import Grid
from .transforms import ParticleTransformSpec

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

    grid: Grid = cfg_field(
        ...,
        description=(
            "Spatial domain and resolution for the footprint. May be omitted "
            "when ``geometry`` is given, in which case it is derived."
        ),
    )
    geometry: GeometrySpec | None = cfg_field(
        None,
        description=(
            "State geometry this footprint serves (file, h3, windows). Used to "
            "derive ``grid`` when that is omitted, and recorded for aggregation."
        ),
        visibility="advanced",
    )
    cells_per_target: float = cfg_field(
        4.0,
        description="Native cells across the smallest geometry cell when deriving ``grid``.",
        visibility="advanced",
        gt=0,
    )
    geometry_hash: str | None = cfg_field(
        None,
        description=(
            "Content hash of the built ``geometry`` (``Mesh.hash``), recorded so "
            "a stored footprint can detect that the geometry file changed later. "
            "Filled automatically; not needed when ``geometry`` is unset."
        ),
        visibility="advanced",
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

    smooth_factor: float = cfg_field(
        1.0,
        description="Factor by which to linearly scale footprint smoothing. Defaults to 1",
    )
    time_integrate: bool = cfg_field(
        False,
        description="If True, sum the footprint over all time steps to produce a single 2-D layer.",
    )
    error: bool = cfg_field(
        False,
        description=(
            "If True, also compute an error footprint from the error trajectories "
            'and store it alongside the main footprint under "{name}_error".'
        ),
        visibility="advanced",
    )
    transforms: list[ParticleTransformSpec] = cfg_field(
        description="Declarative particle transforms applied before rasterizing the footprint.",
        default_factory=list,
        visibility="advanced",
    )

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
