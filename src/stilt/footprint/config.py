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

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       geometry:                   # the default footprint's geometry
         kind: file
         path: counties.shp
         ids: NAME

   variants:
     hrrr: {}
     hrrr-hexes:                   # a second footprint from the same particles
       stages:
         footprint:
           geometry: {kind: h3, resolution: 8, bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}}
           cells_per_target: 4
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import timedelta
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from stilt.identity import canonical
from stilt.spatial import Bounds, Grid
from stilt.transforms import dump_transform, load_transform, transform_kind


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


#: What to do instead of ``time_integrate``, which was removed.
TIME_INTEGRATE_REMOVED = (
    "time_integrate was removed. Sum the layers instead: foot.sum('time'), "
    "or project.footprints().foot.sum('hour') (the layer dimension is named for tres's unit). For one layer as long as the "
    "run, set tres to the run's length (tres: 1D for a 24-hour run)."
)


def _minutes(tres: str | timedelta) -> int:
    """
    Return the length of a footprint time layer, *tres*, in whole minutes.

    *tres* is a duration such as ``"15min"``, ``"1h"`` or ``"1D"`` (any
    string :class:`pandas.Timedelta` reads), or a :class:`datetime.timedelta`.

    Raises
    ------
    ValueError
        If *tres* is a number, is not positive, or is not whole minutes.

    Examples
    --------
    >>> _minutes("15min"), _minutes("1h"), _minutes("1D")
    (15, 60, 1440)
    """
    if not isinstance(tres, str | timedelta):
        raise ValueError(
            f"tres is a duration such as '15min', '1h' or '1D', not {tres!r}."
        )
    step = pd.Timedelta(tres)
    if pd.isna(step) or step <= pd.Timedelta(0) or step % pd.Timedelta(minutes=1):
        raise ValueError(
            f"tres must be a positive whole number of minutes, not {tres!r}."
        )
    return int(step / pd.Timedelta(minutes=1))


class FootprintConfig(BaseModel):
    """
    Footprint settings: the grid, smoothing, the near-field correction, ``foot``, and particle transforms.

    The config and each variant hold these as their defaults and overrides,
    and every footprint keeps the settings it was calculated with.

    ``grid`` is the raster the footprint is computed on. Leave both ``grid``
    and ``geometry`` unset for a variant that only produces particles.
    Give ``geometry`` to name the polygons the footprint will be aggregated
    to, and the grid is derived from them (:meth:`stilt.Mesh.to_grid`) when
    the project first reads its variants (``project.variants``), not when
    the config loads. When
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
        ge=0,
    )
    tres: str = Field(
        "1h",
        description=(
            "Length of each time layer of the footprint, counted from the "
            "release time, as a duration in whole minutes: ``1h`` for hourly "
            "layers, ``15min`` for 15-minute data, ``1D`` for days."
        ),
    )
    hnf_plume: bool = Field(
        True,
        description=(
            "Correct ``foot`` near the release with a vertical Gaussian plume "
            "model before the footprint is computed "
            "(:func:`stilt.footprint.correct_near_field`). Close to the "
            "release the plume is thinner than the layer surface fluxes are "
            "mixed through, so this raises the influence of nearby fluxes. "
            "Needs the particle columns ``dens``, ``tlgr``, ``sigw``, ``foot``, "
            "``mlht``, and ``samt``, and ``zsfc`` for a release above sea "
            "level; particles without them get no correction."
        ),
    )
    footprint_depth: float = Field(
        0.5,
        gt=0,
        description=(
            "Height below which a particle's time counts toward the ``foot`` "
            "PYSTILT computes: the depth surface fluxes are mixed through. A "
            "value of 1 or less is a fraction of the mixed-layer height; a "
            "larger value is meters above ground. It is used when PYSTILT "
            "computes ``foot`` (``pbl_height`` set, or particles without "
            "``foot``). A footprint from HYSPLIT's own ``foot`` takes HYSPLIT's "
            "``veght`` instead, and records it here; a different value written "
            "for such a footprint is refused."
        ),
    )
    pbl_height: Literal["met", "richardson"] | float | None = Field(
        None,
        description=(
            "How PYSTILT finds the mixed-layer height (``mlht``) when it "
            "computes ``foot``: ``met`` reads the met's ``PBLH``, ``richardson`` "
            "finds it from the met's profile as HYSPLIT's ``kmixd: 3`` does, "
            "and a number is a fixed height in meters above ground. Set, the "
            "footprint is made from a ``foot`` PYSTILT computes from the met "
            "files the run read, in place of the transport model's own. Unset, "
            "it uses the model's ``foot``; for particles without one it "
            "computes ``foot`` from the model's ``mlht``, or else the met's "
            "``PBLH``."
        ),
    )
    pbl_height_min: float = Field(
        0.0,
        ge=0,
        description=(
            "Lowest mixed-layer height, in meters, for a height PYSTILT finds "
            "itself (``pbl_height``, or the met's ``PBLH`` when unset), as "
            "``kmix0`` is for HYSPLIT. 0 sets no minimum."
        ),
    )
    transforms: list[Any] = Field(
        description=(
            "Particle transforms applied in order before the footprint is "
            "computed. Each entry's ``kind`` is a built-in name "
            "(``averaging_kernel``, ``pressure_weighting``, "
            "``first_order_lifetime``, ``dry_deposition``, ``oxidation``) or "
            "the import path of your own class."
        ),
        default_factory=list,
    )
    geometry_hash: str | None = Field(
        None,
        description=(
            "Hash of the polygons ``geometry`` names (:attr:`stilt.Mesh.hash`), "
            "recorded with each footprint so that it is summed onto the same "
            "polygons. Not a setting to write: the project sets it when it "
            "reads the geometry, and replaces one written in ``config.yaml``."
        ),
    )

    @property
    def zsfc_reasons(self) -> list[str]:
        """
        What goes wrong for releases above sea level whose particles lack ``zsfc``, a clause each.

        ``stilt check`` lists them when the transport model is not asked for
        ``zsfc`` (HYSPLIT's ``varsiwant``): the near-field correction is
        skipped for such a release, and a transform that reads the terrain
        height fails.
        """
        reasons = []
        if self.hnf_plume:
            reasons.append("the near-field correction (hnf_plume) is skipped for them")
        kinds = [
            transform_kind(t)
            for t in self.transforms
            if getattr(t, "reads_zsfc", False)
        ]
        if kinds:
            named = " and ".join(dict.fromkeys(kinds))
            noun = "transform fails" if len(set(kinds)) == 1 else "transforms fail"
            reasons.append(f"the {named} {noun} their footprints")
        return reasons

    @classmethod
    def override(cls, block: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
        """
        Return a variant's footprint settings: the block's, changed by the variant's *own*.

        A ``grid`` mapping updates the block's grid field by field, so a
        variant can change only the resolution. A variant that sets its own
        ``geometry`` drops the block's ``grid`` unless it sets ``grid`` too,
        so the grid is derived from its geometry.

        Raises
        ------
        ValueError
            If the variant changes part of a grid that is derived from the
            block's ``geometry``. That grid is not known until the geometry
            is read.
        """
        merged = {**block, **own}
        if isinstance(own.get("grid"), dict):
            if isinstance(block.get("grid"), dict):
                merged["grid"] = {**block["grid"], **own["grid"]}
            elif block.get("geometry") is not None and "geometry" not in own:
                raise ValueError(
                    "it changes part of a grid that is derived from the "
                    "block's geometry. Give a full grid, or set "
                    "cells_per_target to change its resolution."
                )
        if own.get("geometry") is not None and "grid" not in own:
            merged.pop("grid", None)
        return merged

    @property
    def tres_minutes(self) -> int:
        """Length of each time layer of the footprint, in minutes."""
        return _minutes(self.tres)

    @property
    def layer_unit(self) -> tuple[str, int]:
        """
        Name and length in minutes of the unit ``tres`` is written in.

        It names the layer dimension of :meth:`stilt.Project.footprints`, which
        counts in it: ``hour`` for ``1h``, ``day`` for ``1D``, ``minute`` for
        ``15min``.

        Examples
        --------
        >>> FootprintConfig(tres="6h").layer_unit
        ('hour', 60)
        >>> FootprintConfig(tres="90min").layer_unit
        ('minute', 1)
        """
        if self.tres.endswith("D"):
            return "day", 1440
        if self.tres.endswith("h"):
            return "hour", 60
        return "minute", 1

    @field_validator("tres", mode="before")
    @classmethod
    def _one_spelling(cls, value: Any) -> str:
        """Write the layer length one way, so ``60min`` and ``1h`` name one settings folder."""
        minutes = _minutes(value)
        if minutes % 1440 == 0:
            return f"{minutes // 1440}D"
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}min"

    @model_validator(mode="before")
    @classmethod
    def _no_time_integrate(cls, data: Any) -> Any:
        """
        Refuse ``time_integrate: true``, which was removed.

        A stored record of an hourly footprint may still hold
        ``time_integrate: false``, which is ignored, so its folder is found.
        """
        if isinstance(data, dict) and data.get("time_integrate"):
            raise ValueError(TIME_INTEGRATE_REMOVED)
        return data

    @field_validator("pbl_height")
    @classmethod
    def _pbl_height_is_positive(
        cls, pbl_height: str | float | None
    ) -> str | float | None:
        """Refuse a fixed mixed-layer height that is not above the ground."""
        if isinstance(pbl_height, float) and not pbl_height > 0:
            raise ValueError(
                "pbl_height must be met, richardson, or a height above 0 m."
            )
        return pbl_height

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


def _record_transform(transform: Any) -> dict[str, Any]:
    """Return a transform as recorded, its ``kind`` alone if it has no settings to write."""
    try:
        return dump_transform(transform)
    except TypeError:
        return {"kind": transform_kind(transform)}


def _read_transform(spec: dict[str, Any], source: str) -> Any:
    """Return a recorded transform, or its mapping when it cannot be rebuilt here."""
    try:
        return load_transform(spec)
    except (ImportError, TypeError, ValueError) as exc:
        warnings.warn(
            f"{source}: transform {spec.get('kind')!r} could not be rebuilt "
            f"({exc}). It is kept as its settings and cannot be applied.",
            stacklevel=3,
        )
        return spec


def footprint_settings(footprint: FootprintConfig) -> dict[str, Any]:
    """
    Return the settings that identify a footprint, besides its particles, in canonical form.

    The same record goes in a footprint folder's ``_settings.yaml`` and in
    each footprint file and array: the config, with the grid the footprint
    is computed on and the hash of the geometry the grid was derived for. A
    transform that is not a pydantic model is recorded by its ``kind``
    alone.
    """
    data = footprint.model_dump(mode="json", exclude={"transforms"})
    data["transforms"] = [_record_transform(t) for t in footprint.transforms]
    return canonical(data)


def read_footprint_settings(
    stored: Mapping[str, Any], source: str = "footprint settings"
) -> FootprintConfig:
    """
    Return the footprint config that :func:`footprint_settings` recorded.

    A transform whose class cannot be imported here is kept as its mapping,
    with a warning naming *source*, so the record still reads.
    """
    data = dict(stored)
    specs = data.pop("transforms", [])
    config = FootprintConfig.model_validate(data)
    # model_copy skips validation, so a mapping stays a mapping.
    return config.model_copy(
        update={"transforms": [_read_transform(s, source) for s in specs]}
    )


__all__ = [
    "FileGeometrySpec",
    "FootprintConfig",
    "GeometrySpec",
    "H3GeometrySpec",
    "WindowsGeometrySpec",
    "footprint_settings",
    "read_footprint_settings",
]
