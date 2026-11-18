"""
The footprint stage: what a ``stages`` block of ``config.yaml`` names.

A stage is what is made from each simulation's results after the transport
model runs (:class:`stilt.stages.Stage`). A project lists its stages by name
under ``stages:``, each block giving the stage's import path as ``class``
and then the stage's settings. :class:`FootprintStage` is the one stage
PYSTILT has.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from pydantic import TypeAdapter

from stilt._paths import absolute
from stilt.exceptions import EmptyFootprint
from stilt.particles import HNF_PLUME_COLUMNS
from stilt.receptors import Receptor
from stilt.spatial import Grid

from .config import (
    FileGeometrySpec,
    FootprintConfig,
    GeometrySpec,
    footprint_settings,
)
from .gridding import calc_footprint
from .io import write_empty_footprint, write_footprint
from .targets import Mesh

if TYPE_CHECKING:
    from stilt._paths import Location
    from stilt.config import Variant
    from stilt.output import Output
    from stilt.simulation import Simulation

#: One thing :meth:`FootprintStage.check` found: the check, the receptor it is about (or ``None``), and what is wrong.
_Finding = tuple[str, str | None, str]

_GEOMETRY_SPEC: TypeAdapter[GeometrySpec] = TypeAdapter(GeometrySpec)


class FootprintStage:
    """
    The stage that grids each simulation's particles into its footprint.

    Name it as the ``class`` of a block under ``stages:`` in
    ``config.yaml``. The block's other keys are the footprint settings
    (:class:`FootprintConfig`), the defaults of every variant; a variant
    changes some under its own ``stages`` key, or sets the stage to null to
    make particles only.

    .. code-block:: yaml

       stages:
         footprint:
           class: stilt.footprint.FootprintStage
           grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}

       variants:
         hrrr: {}
         hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}
         hrrr-particles: {stages: {footprint: null}}

    Attributes
    ----------
    name : str
        ``footprint``, what its results are called.
    config_class : type
        :class:`FootprintConfig`, which checks the block's settings.
    suffix : str
        ``.parquet``, the extension of a stored footprint.
    reads : tuple of str
        Empty: a footprint is made from the particles alone.
    """

    name = "footprint"
    config_class = FootprintConfig
    suffix = ".parquet"
    reads: tuple[str, ...] = ()

    def resolve(
        self,
        config: FootprintConfig,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> FootprintConfig:
        """
        Return the footprint settings a variant's simulations run with.

        The geometry is read, once however many variants name it, and its
        hash recorded (``geometry_hash``); the grid of a footprint given
        only by a geometry is derived from it (:meth:`stilt.Mesh.to_grid`).
        A relative geometry file starts from *directory*.

        Raises
        ------
        ValueError
            If the settings have neither a grid nor a geometry, or the
            near-field correction is on and the transport model is not
            asked for a particle column it reads (HYSPLIT's ``varsiwant``).
        """
        if config.grid is None and config.geometry is None:
            raise ValueError(
                "the footprint has no grid. Give it a grid or a geometry, or "
                "set the stage to null under the variant's stages: to make "
                "particles only."
            )
        columns = model_settings.get("varsiwant")
        if config.hnf_plume and columns is not None:
            missing = sorted(set(HNF_PLUME_COLUMNS) - set(columns))
            if missing:
                raise ValueError(
                    "the near-field correction (hnf_plume) reads the particle "
                    f"columns {missing}. Add them to varsiwant, or set "
                    "hnf_plume: false."
                )
        update: dict[str, Any] = {"geometry_hash": None}
        if config.geometry is not None:
            mesh = _mesh(config.geometry, directory)
            update["geometry_hash"] = mesh.hash
            if config.grid is None:
                update["grid"] = mesh.to_grid(cells_per_target=config.cells_per_target)
        return config.model_copy(update=update)

    def settings(self, config: FootprintConfig) -> dict[str, Any]:
        """Return the settings that identify a footprint, besides its particles (:func:`~stilt.footprint.config.footprint_settings`)."""
        return footprint_settings(config)

    def check(
        self, config: FootprintConfig, receptors: pd.DataFrame, directory: Path | None
    ) -> list[_Finding]:
        """
        Return what would stop the footprints of *receptors*, found without a run.

        The receptors with a release point outside the footprint grid
        (``grid``), and what each transform finds wrong with its own inputs,
        such as an averaging kernel's table. A missing ``zsfc`` is the
        project check's to find (:attr:`FootprintConfig.zsfc_reasons` says
        what needs it).
        """
        found: list[_Finding] = []
        if config.grid is not None:
            found += _outside(receptors, config.grid)
        ids = list(dict.fromkeys(str(r) for r in receptors["receptor"]))
        for transform in config.transforms:
            check = getattr(transform, "check", None)
            if check is None:
                continue
            found += [
                (str(name), None if receptor is None else str(receptor), str(message))
                for name, receptor, message in check(ids, directory)
            ]
        return found

    def run(self, config: FootprintConfig, sim: Simulation, path: Location) -> None:
        """
        Calculate a simulation's footprint from its particles and write it at *path*.

        The near-field correction reads the run's ``footprint_depth``. A
        relative file name in a transform's settings starts from the
        project directory (``sim.directory``). When no particle reaches the
        grid, the file written has no rows and is marked ``stilt:empty``: a
        finished result, which ``sim.footprint`` reads as ``None``.
        :meth:`stilt.Simulation.calc_footprint` makes footprints with other
        settings, without writing them.
        """
        if config.grid is None:  # resolve gives every footprint its grid
            raise ValueError(f"The footprint of {sim} has no grid.")
        try:
            foot = calc_footprint(
                sim.particles,
                sim.receptor,
                config.grid,
                smooth_factor=config.smooth_factor,
                tres=config.tres,
                hnf_plume=config.hnf_plume,
                footprint_depth=sim.variant.transport.footprint_depth,
                transforms=config.transforms,
                name=sim.variant.name,
                directory=sim.directory,
                geometry_hash=config.geometry_hash,
            )
        except EmptyFootprint:
            foot = None
        _write(path, sim.variant, sim.receptor, foot, sim.realization)


def store_footprint(
    output: Output, variant: Variant, foot: xr.DataArray, realization: int | None = None
) -> Location:
    """
    Write a footprint into *variant*'s footprint folder of an output directory, without calculating it.

    For a footprint made elsewhere, as one imported from STILT-R is; a run
    calculates its own (:meth:`FootprintStage.run`). The footprint must be
    on the variant's grid. The file records the footprint settings, the
    folder's hash, and the hash of the particles it was made from, so it
    reads alone.
    """
    receptor = foot.stilt.receptor
    path = output.make_path("footprint", variant, str(receptor.id), realization)
    return _write(path, variant, receptor, foot, realization)


def store_empty_footprint(
    output: Output, variant: Variant, receptor: Receptor, realization: int | None = None
) -> Location:
    """Record in *variant*'s footprint folder that a receptor's footprint is empty: no particle reached the grid."""
    path = output.make_path("footprint", variant, str(receptor.id), realization)
    return _write(path, variant, receptor, None, realization)


def _write(
    path: Location,
    variant: Variant,
    receptor: Receptor,
    foot: xr.DataArray | None,
    realization: int | None,
) -> Location:
    """Write a receptor's footprint under *variant*'s settings at *path*, the empty one when *foot* is ``None``."""
    from importlib.metadata import PackageNotFoundError, version

    config = variant.stages["footprint"]
    try:
        pystilt = version("pystilt")
    except PackageNotFoundError:
        pystilt = "0+unknown"
    # What a footprint file records besides its contents: its folder's
    # settings hash, the PYSTILT version, the hash of the particles it was
    # made from, and its realization in an ensemble.
    stamp = {
        b"stilt:hash": variant.stage_hashes["footprint"].encode(),
        b"stilt:pystilt": pystilt.encode(),
        b"stilt:particles_hash": variant.particles_hash.encode(),
    }
    if realization is not None:
        stamp[b"stilt:realization"] = str(realization).encode()
    if foot is None:
        return write_empty_footprint(path, receptor, config, variant.name, stamp)
    return write_footprint(path, foot, config, stamp)


def _mesh(spec: GeometrySpec, directory: Path | None) -> Mesh:
    """
    Return the mesh a geometry spec describes, reading each spec once.

    A relative file starts from *directory*. A file is read again when it
    has changed. The spec in the settings keeps the path as written, and a
    mesh's hash is of its polygons, so moving the project changes no hash.
    """
    modified = None
    if isinstance(spec, FileGeometrySpec):
        if directory is not None:
            spec = spec.model_copy(update={"path": str(absolute(spec.path, directory))})
        try:
            modified = os.stat(spec.path).st_mtime_ns
        except OSError:
            modified = None  # reading it says what is wrong
    return _read_mesh(spec.model_dump_json(), modified)


@functools.lru_cache(maxsize=16)
def _read_mesh(spec: str, modified: int | None) -> Mesh:
    """Return the mesh of a geometry spec given as JSON; *modified* is its file's time, so a changed file is read again."""
    return Mesh.from_spec(_GEOMETRY_SPEC.validate_json(spec))


def _outside(rows: pd.DataFrame, grid: Grid) -> list[_Finding]:
    """List the receptors with a release point outside *grid*, which is bounded in longitude and latitude."""
    outside = rows[~grid.contains(rows["longitude"], rows["latitude"])]
    box = (
        f"{grid.xmin:g} to {grid.xmax:g} longitude, "
        f"{grid.ymin:g} to {grid.ymax:g} latitude"
    )
    return [
        (
            "grid",
            str(row.receptor),
            f"longitude {row.longitude:g} and latitude {row.latitude:g} are "
            f"outside the footprint grid ({box})",
        )
        for row in outside.drop_duplicates("receptor").itertuples()
    ]


__all__ = ["FootprintStage", "store_empty_footprint", "store_footprint"]
