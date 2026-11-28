"""
The footprint stage: what a ``stages`` block of ``config.yaml`` names.

A stage is what is made from each simulation's results after the transport
model runs (:class:`runtraj.stages.Stage`). A project lists its stages by name
under ``stages:``, each block giving the stage's import path as ``class``
and then the stage's settings. :class:`FootprintStage` is the one stage
PYSTILT has.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import TypeAdapter
from runtraj import Bounds
from runtraj.identity import file_stamp
from runtraj.models import MetConfig, get_model
from runtraj.particles import particles_metadata
from runtraj.releases import Release
from runtraj.stages import Stage

from stilt._paths import absolute
from stilt.exceptions import EmptyFootprint
from stilt.footprint.foot import HNF_PLUME_COLUMNS, add_foot
from stilt.footprint.grid import Grid

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
    from runtraj.config import Variant
    from runtraj.output import Output
    from runtraj.simulation import Simulation
    from upath import UPath

logger = logging.getLogger(__name__)

#: The import path of the footprint stage, the ``class`` of its ``stages:`` block.
FOOTPRINT_STAGE = "stilt.footprint.FootprintStage"

#: One thing :meth:`FootprintStage.check` found: the check, the release it is about (or ``None``), and what is wrong.
_Finding = tuple[str, str | None, str]

_GEOMETRY_SPEC: TypeAdapter[GeometrySpec] = TypeAdapter(GeometrySpec)


class FootprintStage(Stage[FootprintConfig]):
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

    It reads no other stage's files, only the particles, and sets no
    ``version``, so its settings folders are named as before.
    """

    name = "footprint"
    config_class = FootprintConfig
    suffix = ".parquet"

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

        When the footprint is made from the transport model's own ``foot``
        (``pbl_height`` unset, and a model that records the depth it made
        ``foot`` with, HYSPLIT's ``veght``), ``footprint_depth`` becomes
        that depth: the near-field correction reads it, and the footprint
        records it.

        Raises
        ------
        ValueError
            If the settings have neither a grid nor a geometry, the
            near-field correction is on and the transport model is not asked
            for a particle column it reads (HYSPLIT's ``varsiwant``), or a
            ``footprint_depth`` is written for a footprint made from the
            model's ``foot`` and differs from the model's depth.
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
        depth = model_foot_depth(config, model_settings)
        if depth is not None:
            written = "footprint_depth" in config.model_fields_set
            if written and config.footprint_depth != depth:
                raise ValueError(
                    f"footprint_depth is {config.footprint_depth:g}, and the "
                    f"footprint is made from the {model} model's own foot, "
                    f"which it made with veght {depth:g}. Set veght: "
                    f"{config.footprint_depth:g} in the model's settings for "
                    "that depth, or set pbl_height so that PYSTILT computes "
                    "foot with footprint_depth."
                )
            update["footprint_depth"] = depth
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
        self, config: FootprintConfig, releases: pd.DataFrame, directory: Path | None
    ) -> list[_Finding]:
        """
        Return what would stop the footprints of *releases*, found without a run.

        The releases with a release point outside the footprint grid
        (``grid``), and what each transform finds wrong with its own inputs,
        such as an averaging kernel's table. A missing ``zsfc`` is the
        project check's to find (:attr:`FootprintConfig.zsfc_reasons` says
        what needs it).
        """
        found: list[_Finding] = []
        if (grid := config.grid) is not None:
            found += _outside(releases, grid)
        for transform in config.transforms:
            check = getattr(transform, "check", None)
            if check is None:
                continue
            found += _transform_findings(check, releases, directory)
        return found

    def run(self, config: FootprintConfig, sim: Simulation, path: Path | UPath) -> None:
        """
        Calculate a simulation's footprint from its particles and write it at *path*.

        The particles get the ``foot`` the settings ask for
        (:func:`footprint_particles`): the transport model's own, or one
        computed from the met files the run read, which is used and not
        stored. The near-field correction reads ``footprint_depth``. A
        relative file name in a transform's settings starts from the
        project directory (``sim.directory``). When no particle reaches the
        grid, the file written has no rows and is marked ``stilt:empty``: a
        finished result, which ``sim.footprint`` reads as ``None``.
        :meth:`stilt.Simulation.calc_footprint` makes footprints with other
        settings, without writing them.

        Parameters
        ----------
        config : FootprintConfig
            The variant's footprint settings, as :meth:`resolve` returned them.
        sim : runtraj.simulation.Simulation
            The simulation, as the worker gives it: the particle part, which
            reads the particles.
        path : Path or UPath
            Where to write the footprint file.
        """
        if config.grid is None:  # resolve gives every footprint its grid
            raise ValueError(f"The footprint of {sim} has no grid.")
        try:
            foot = calc_footprint(
                simulation_particles(sim, config),
                sim.release,
                config.grid,
                smooth_factor=config.smooth_factor,
                tres=config.tres,
                hnf_plume=config.hnf_plume,
                footprint_depth=config.footprint_depth,
                transforms=config.transforms,
                name=sim.variant.name,
                directory=sim.directory,
                geometry_hash=config.geometry_hash,
            )
        except EmptyFootprint:
            foot = None
        _write(path, config, sim.variant, sim.release, foot, sim.realization, self.name)


def footprint_particles(
    particles: pd.DataFrame,
    config: FootprintConfig,
    met_files: Iterable[str | os.PathLike[str]],
    *,
    z_top: float | None = None,
    variant: Variant | None = None,
) -> pd.DataFrame:
    """
    Return the particles a footprint under *config* is made from, with the ``foot`` it uses.

    The particles are returned as they are when they have ``foot`` and
    ``pbl_height`` is unset: the transport model's own ``foot``, such as
    HYSPLIT's. Otherwise PYSTILT computes ``foot`` with ``footprint_depth``,
    ``pbl_height`` and ``pbl_height_min`` (:func:`stilt.footprint.add_foot`),
    in place of the model's, and returns a copy with it. The particles
    passed in are not changed, and the computed ``foot`` is stored
    nowhere: each footprint computes it again.

    Parameters
    ----------
    particles : pandas.DataFrame
        A particle table, such as ``sim.particles``.
    config : FootprintConfig
        The footprint settings.
    met_files : iterable of path-like
        The met files the run read, which a particle file records
        (:func:`runtraj.particles.particles_metadata`). They are read only
        when ``foot`` is computed, and must still be at those paths then.
    z_top : float, optional
        The top of the run's domain, HYSPLIT's ``z_top``, which
        ``pbl_height: richardson`` reads.
    variant : runtraj.config.Variant, optional
        The variant that made the particles, such as ``sim.variant``. Its
        met opens its files in the format its transport model reads
        (:meth:`runtraj.models.MetConfig.open`), around the particles. Without it
        the files are read as ARL files, as HYSPLIT's are, so pass it for a
        transport model whose met is in another format.

    Returns
    -------
    pandas.DataFrame
        The particles, with ``foot``.

    Raises
    ------
    FileNotFoundError
        If ``foot`` is computed and the run named no met files, or one of
        them is no longer there.

    Examples
    --------
    >>> meta = runtraj.particles.particles_metadata(
    ...     sim.particles_path
    ... )  # doctest: +SKIP
    >>> config = stilt.FootprintConfig(
    ...     grid=grid, pbl_height="richardson"
    ... )  # doctest: +SKIP
    >>> particles = stilt.footprint.footprint_particles(
    ...     sim.particles, config, meta.met_files, variant=sim.variant
    ... )  # doctest: +SKIP
    """
    if not _computes_foot(particles, config):
        return particles
    files = [Path(f) for f in met_files]
    if not files:
        raise FileNotFoundError(
            "The footprint needs a foot computed by PYSTILT, from the met "
            "files the run read, and the run named none."
        )
    gone = [f for f in files if not f.exists()]
    if gone:
        raise FileNotFoundError(
            f"The footprint needs a foot computed by PYSTILT from the met files "
            f"the run read, and {gone[0]} is not there any more"
            f"{f' (nor {len(gone) - 1} more)' if len(gone) > 1 else ''}. Put the "
            "met files back where the run read them."
        )
    had_foot = "foot" in particles.columns
    out = particles.drop(columns="foot") if had_foot else particles
    met = (
        files
        if variant is None
        else variant.met_config.open(files, _particle_bounds(particles))
    )
    out = add_foot(
        out,
        met,
        config.footprint_depth,
        pbl_height=config.pbl_height,
        pbl_height_min=config.pbl_height_min,
        z_top=z_top,
    )
    logger.info(
        "foot computed by PYSTILT from %s and the air below it in the met files "
        "(stilt.footprint.add_foot)%s",
        _mixed_layer(particles, config),
        " in place of the model's foot" if had_foot else "",
    )
    return out


def _particle_bounds(particles: pd.DataFrame) -> Bounds | None:
    """
    Return the smallest box around the particles' longitudes and latitudes, or ``None`` when none is finite.

    Longitudes count modulo 360, so particles on both sides of 180 degrees
    get a box across it (``xmin > xmax``), not one around the globe. A box
    needs a width and a height: particles at one longitude or latitude get
    0.01 degrees on each side; the model adds its own margin.
    """
    lon = particles["lon"].to_numpy(dtype=float)
    lat = particles["lat"].to_numpy(dtype=float)
    keep = np.isfinite(lon) & np.isfinite(lat)
    if not keep.any():
        return None
    ymin, ymax = float(lat[keep].min()), float(lat[keep].max())
    if ymin == ymax:
        ymin, ymax = max(ymin - 0.01, -90.0), min(ymax + 0.01, 90.0)
    # The box starts after the widest gap between the sorted longitudes;
    # when that gap is the one across 180 degrees, it is the plain box.
    lon = np.unique(_to_180(lon[keep]))
    gaps = np.diff(lon, append=lon[0] + 360.0)
    widest = int(np.argmax(gaps))
    xmin, xmax = float(lon[(widest + 1) % lon.size]), float(lon[widest])
    if xmin == xmax:
        xmin, xmax = xmin - 0.01, xmax + 0.01
    return Bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _to_180(lon: np.ndarray) -> np.ndarray:
    """Return longitudes in degrees from -180 up to 180, those already there as they are."""
    inside = (lon >= -180.0) & (lon < 180.0)
    return np.where(inside, lon, (lon + 180.0) % 360.0 - 180.0)


def _mixed_layer(particles: pd.DataFrame, config: FootprintConfig) -> str:
    """Say where the mixed-layer height of a ``foot`` PYSTILT computes comes from."""
    pbl_height = config.pbl_height
    if pbl_height is None and "mlht" in particles.columns:
        return "the model's mlht"
    if pbl_height is None or pbl_height == "met":
        found = "the met's PBLH"
    elif pbl_height == "richardson":
        found = "the mixed layer found from the met's profile (HYSPLIT's kmixd 3)"
    else:
        found = f"a mixed layer of {pbl_height:g} m"
    if config.pbl_height_min > 0:
        found += f", at least {config.pbl_height_min:g} m,"
    return found


def stored_particles(
    particles: pd.DataFrame,
    config: FootprintConfig,
    path: Path | UPath | None,
    *,
    start: dt.datetime | pd.Timestamp | None = None,
    z_top: float | None = None,
    variant: Variant | None = None,
) -> pd.DataFrame:
    """
    Return particles read from the file at *path* with the ``foot`` a footprint under *config* uses (:func:`footprint_particles`).

    The met files are read from the particle file's metadata only when
    ``foot`` is computed. For that, particles without ``time`` (the run's
    own, in memory) get it from *start*, the release time, and lose it
    again. *variant*, the variant that made the particles, says how its
    transport model opens the met files (:func:`footprint_particles`).

    Raises
    ------
    ValueError
        If ``foot`` is computed, *variant* is not given, and the particle
        file's transport model reads met that is not in ARL files.
    """
    if not _computes_foot(particles, config):
        return particles
    met_files = []
    if path is not None:
        meta = particles_metadata(path)
        met_files = meta.met_files
        model = meta.settings.get("model", {}).get("name")
        if variant is None and model is not None:
            _refuse_own_met(model, path)
    added_time = "time" not in particles.columns and start is not None
    if added_time:
        particles = particles.assign(
            time=pd.Timestamp(start)
            + pd.to_timedelta(particles["age"].to_numpy(), unit="min")
        )
    out = footprint_particles(
        particles, config, met_files, z_top=z_top, variant=variant
    )
    return out.drop(columns="time") if added_time else out


def _refuse_own_met(model: str, path: Path | UPath) -> None:
    """Refuse to read the met of *model* as ARL files when its met opens its own (:meth:`runtraj.models.MetConfig.open`)."""
    try:
        found = get_model(model)
    except (ImportError, ValueError):  # not installed here: nothing to ask it
        return
    if found.met_config_class.open is not MetConfig.open:
        raise ValueError(
            f"The particles in {path} were made by the transport model "
            f"{model!r}, whose met files are not ARL files. Pass the variant "
            "that made them (variant=sim.variant), so the model opens its met."
        )


def simulation_particles(sim: Simulation, config: FootprintConfig) -> pd.DataFrame:
    """Return a simulation's particles with the ``foot`` a footprint under *config* uses (:func:`stored_particles`)."""
    return stored_particles(
        sim.particles,
        config,
        sim.particles_path,
        start=sim.release.time,
        z_top=getattr(sim.transport, "z_top", None),
        variant=sim.variant,
    )


def variant_footprint(variant: Variant) -> FootprintConfig:
    """
    Return *variant*'s footprint settings, or the defaults when it has no footprint.

    The defaults get the depth of the transport model's own ``foot``, as
    :meth:`FootprintStage.resolve` gives it (:func:`model_foot_depth`), so a
    footprint made from particles of a variant without one corrects the
    near field as its run would.
    """
    own = variant.stages.get("footprint")
    if own is not None:
        return own
    own = FootprintConfig.model_validate({})
    depth = model_foot_depth(own, variant.transport.settings())
    return own if depth is None else own.model_copy(update={"footprint_depth": depth})


def _computes_foot(particles: pd.DataFrame, config: FootprintConfig) -> bool:
    """Whether a footprint under *config* is made from a ``foot`` PYSTILT computes: ``pbl_height`` is set, or *particles* have none."""
    return config.pbl_height is not None or "foot" not in particles.columns


def model_foot_depth(
    config: FootprintConfig, model_settings: Mapping[str, Any]
) -> float | None:
    """
    Return the depth the transport model made its ``foot`` with when the footprint uses that ``foot``, else ``None``.

    The depth is HYSPLIT's ``veght``. A model that records none, or that is
    not asked for ``foot`` (HYSPLIT's ``varsiwant``), or ``pbl_height``
    set, gives ``None``: then PYSTILT computes ``foot`` with
    ``footprint_depth``.
    """
    depth = model_settings.get("veght")
    columns = model_settings.get("varsiwant")
    if config.pbl_height is not None or depth is None:
        return None
    if columns is not None and "foot" not in columns:
        return None
    return float(depth)


def store_footprint(
    output: Output, variant: Variant, foot: xr.DataArray, realization: int | None = None
) -> Path | UPath:
    """
    Write a footprint into *variant*'s footprint folder of an output directory, without calculating it.

    For a footprint made elsewhere, as one imported from STILT-R is; a run
    calculates its own (:meth:`FootprintStage.run`). The footprint must be
    on the variant's grid. The file records the footprint settings, the
    folder's hash, and the hash of the particles it was made from, so it
    reads alone.
    """
    release = foot.stilt.release
    name = FootprintStage.name
    path = output.make_path(name, variant, str(release.id), realization)
    return _write(path, variant.stages[name], variant, release, foot, realization)


def store_empty_footprint(
    output: Output, variant: Variant, release: Release, realization: int | None = None
) -> Path | UPath:
    """Record in *variant*'s footprint folder that a release's footprint is empty: no particle reached the grid."""
    name = FootprintStage.name
    path = output.make_path(name, variant, str(release.id), realization)
    return _write(path, variant.stages[name], variant, release, None, realization)


def _write(
    path: Path | UPath,
    config: FootprintConfig,
    variant: Variant,
    release: Release,
    foot: xr.DataArray | None,
    realization: int | None,
    stage: str = FootprintStage.name,
) -> Path | UPath:
    """
    Write a release's footprint, made with *config*, at *path*; the empty one when *foot* is ``None``.

    The file records what every result file does
    (:func:`runtraj.identity.file_stamp`, with the hash of *variant*'s folder
    of *stage*), and the hash of the particles it was made from.
    """
    stamp = file_stamp(variant.stage_hashes[stage], realization)
    stamp[b"stilt:particles_hash"] = variant.result_hash.encode()
    if foot is None:
        return write_empty_footprint(path, release, config, variant.name, stamp)
    return write_footprint(path, foot, config, stamp)


def _transform_findings(
    check: Callable[..., Any], releases: pd.DataFrame, directory: Path | None
) -> list[_Finding]:
    """Return what a transform's own check finds for the releases, as plain tuples."""
    ids = list(dict.fromkeys(str(r) for r in releases["release"]))
    return [
        (str(name), None if release is None else str(release), str(message))
        for name, release, message in check(ids, directory)
    ]


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
    """List the releases with a release point outside *grid*'s bounds."""
    outside = rows[~grid.contains(rows["longitude"], rows["latitude"])]
    box = (
        f"{grid.xmin:g} to {grid.xmax:g} longitude, "
        f"{grid.ymin:g} to {grid.ymax:g} latitude"
        if grid.is_longlat
        else f"x {grid.xmin:.10g} to {grid.xmax:.10g}, y {grid.ymin:.10g} to {grid.ymax:.10g} "
        f"in {grid.crs}"
    )
    return [
        (
            "grid",
            str(row.release),
            f"longitude {row.longitude:g} and latitude {row.latitude:g} are "
            f"outside the footprint grid ({box})",
        )
        for row in outside.drop_duplicates("release").itertuples()
    ]


__all__ = [
    "FootprintStage",
    "footprint_particles",
    "model_foot_depth",
    "simulation_particles",
    "stored_particles",
    "variant_footprint",
    "store_empty_footprint",
    "store_footprint",
]
