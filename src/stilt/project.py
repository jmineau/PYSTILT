"""
:class:`stilt.Project`: a project, with the reads of what STILT makes from its particles.

It is the particle part (:class:`runtraj.project.Project`) and only adds: the
footprints of a selection as one dataset, the Jacobian, the transport
error, and simulations that read their footprint (:class:`stilt.Simulation`).
It opens the same directory. The workers use the particle part, and nothing
on disk names this class, so a machine that only runs the particles opens
the project as well.

A new project made here (:meth:`Project.init`, ``stilt init``) gets the
values STILT-R runs HYSPLIT with (:data:`STILT_R_HYSPLIT`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import pandas as pd
import xarray as xr
import yaml
from runtraj import project
from runtraj.config import ProjectConfig, Variant
from runtraj.particles import read_particles
from runtraj.releases import Release

from stilt.footprint import Geometry, Jacobian
from stilt.footprint.aggregation import _jacobian
from stilt.footprint.io import _threads, open_footprints
from stilt.footprint.stage import stored_particles, variant_footprint
from stilt.simulation import Simulation
from stilt.transport_error import _transport_errors

#: STILT-R's HYSPLIT settings where they differ from what HYSPLIT does on its
#: own: ``stilt.Project.init`` and ``stilt init`` write them into a new project's
#: ``models: hysplit:`` block, so it runs HYSPLIT as STILT-R does. The values
#: are those of STILT-R's ``run_stilt.r``. ``kblt``, ``kmixd`` and ``vscales``
#: are what HYSPLIT chooses itself in STILT mode, and are written so that
#: ``SETUP.CFG`` matches STILT-R's line for line, as are ``conage``,
#: ``cpack``, ``delt``, ``kagl``, ``mhrs``, ``pinbc``, ``pinpf`` and
#: ``poutf``, which change nothing in the particles: HYSPLIT steps 1 minute
#: whatever ``delt`` is, and the others are read only for puffs, the
#: concentration grid, the trajectory model, or files a run has none of.
#: ``varsiwant`` is STILT-R's list with ``pres`` added, for pressure weighting.
STILT_R_HYSPLIT: dict[str, Any] = {
    "numpar": 1000,
    "capemin": -1.0,
    "kblt": 5,
    "kmixd": 3,
    "krand": 4,
    "vscales": -1.0,
    "wvert": False,
    "conage": 48,
    "cpack": 1,
    "delt": 1.0,
    "kagl": 1,
    "mhrs": 9999,
    "pinbc": "",
    "pinpf": "",
    "poutf": "",
    "varsiwant": [
        "time",
        "indx",
        "long",
        "lati",
        "zagl",
        "foot",
        "mlht",
        "pres",
        "dens",
        "samt",
        "sigw",
        "tlgr",
    ],
}

#: The commented ``config.yaml`` that ``stilt.Project.init(path, starter=True)`` and
#: ``stilt init`` write, to edit by hand.
STARTER_CONFIG = """# PYSTILT project configuration
# See docs for details: https://jmineau.github.io/PYSTILT


# Meteorology, by name: the folder of your ARL files. To download them from NOAA
# instead, write hrrr: {download: hrrr, directory: /path/to/keep/them}.
mets:
  hrrr: /path/to/arl/meteorology


# Transport models, by name, each with its settings: the defaults of every
# variant that runs it. Negative n_hours means backward in time.
models:
  hysplit:
    n_hours: -24
    # STILT-R's values where HYSPLIT's own defaults differ. A setting left
    # out runs with HYSPLIT's default.
    numpar: 1000
    capemin: -1     # no convection
    kblt: 5         # Hanna turbulence
    kmixd: 3        # mixed-layer height from the bulk Richardson number
    krand: 4        # random numbers seeded from the clock
    vscales: -1     # Hanna timescale in a stable boundary layer
    wvert: false
    # These change nothing in the particles; they make SETUP.CFG match STILT-R's.
    conage: 48
    cpack: 1
    delt: 1.0
    kagl: 1
    mhrs: 9999
    pinbc: ""
    pinpf: ""
    poutf: ""
    varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, dens, samt, sigw, tlgr]


# Stages, by name: what is made from each simulation's particles. The footprint
# stage grids them. Remove the stages block to make particles only. The grid
# below is a placeholder over northern Utah: replace it with the area your
# releases' footprints cover (stilt check reports releases outside it).
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid:
      xmin: -113.0
      xmax: -110.5
      ymin: 40.0
      ymax: 42.0
      xres: 0.01
      yres: 0.01
    hnf_plume: true  # correct foot for the thin plume near the release before gridding


# Variants. Every release runs once per variant, with the blocks above as
# the defaults. An entry with no overrides runs the defaults as they are; add
# others to run the same releases under other settings, e.g. a mixed-layer
# bracket or a wind-error ensemble (see the docs). A variant changes a
# footprint setting under its own stages: key. Only the variants listed here
# run.
variants:
  hrrr: {}
#  hrrr-zi08: {ziscale: 0.8}
#  hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}


# Results go to this directory, relative to the project unless absolute.
# Several projects can name the same directory and share runs.
output: ./output


# Execution is optional. Local execution is the default. Each backend's
# settings are in the block of its name, and only the backend's block is read.
# execution:
#   backend: local    # or "slurm"
#   local:
#     cpus: 1         # releases at once
#   slurm:
#     n_workers: 1    # array tasks at once
#     cpus: 1         # releases at once in each task
#     time: "02:00:00"
"""


def _realization(value: Any) -> int | None:
    """Return a ``realization`` cell as an int, or ``None`` for a single run."""
    return None if pd.isna(value) else int(value)


def _rows(frame: pd.DataFrame) -> list[tuple[str, str, int | None]]:
    """Return each row's ``(release, variant, realization)``."""
    return [
        (str(r), str(v), _realization(k))
        for r, v, k in zip(
            frame["release"], frame["variant"], frame["realization"], strict=True
        )
    ]


def _one_group(frame: pd.DataFrame, what: str) -> tuple[str, int | None]:
    """Return the one ``(variant, realization)`` of a selection, or raise."""
    groups = list(dict.fromkeys((v, k) for _, v, k in _rows(frame)))
    variants = list(dict.fromkeys(v for v, _ in groups))
    if len(variants) == 1 and len(groups) > 1:
        realizations = [k for _, k in groups]
        raise ValueError(
            f"{what} is made from one realization of an ensemble; this "
            f"selection holds realizations {realizations} of {variants[0]!r}. "
            "Pick one with the realization column, as in "
            f"ens[ens.realization == {realizations[0]}]."
        )
    if len(groups) != 1:
        raise ValueError(
            f"{what} is made from one variant (and one realization of an "
            f"ensemble); this selection has {groups}. Select one first, as in "
            "sims[sims.variant == 'hrrr']."
        )
    return groups[0]


def _with_stilt_r_values(config: ProjectConfig) -> ProjectConfig:
    """
    Return *config* with STILT-R's HYSPLIT values in its ``hysplit`` block, where it gives none.

    A config that does not run the built-in HYSPLIT is returned as it is.
    """
    data = yaml.safe_load(config.to_yaml())
    models = data.get("models") or {"hysplit": {}}
    if "hysplit" not in models or "class" in (models["hysplit"] or {}):
        return config
    block = models["hysplit"] or {}
    models["hysplit"] = block | {
        name: value for name, value in STILT_R_HYSPLIT.items() if name not in block
    }
    data["models"] = models
    return ProjectConfig.model_validate(data)


class Project(project.Project):
    """
    A STILT project: releases, settings, the simulations they define, and their footprints.

    ``Project(path)`` opens a project directory and reads it. It is every
    release under every variant, and a view of their results: list the
    simulations, see which have finished, load particles and footprints.
    The results are in the output directory ``config.yaml`` names
    (``./output`` by default), which several projects can share.

    It is :class:`runtraj.project.Project`, the particle part the workers use,
    and it adds the reads of what STILT makes from the particles:
    :meth:`footprints`, :meth:`jacobian`, :meth:`transport_error`, and
    simulations that read their footprint (:class:`stilt.Simulation`). Of
    what it inherits it changes only :meth:`simulation`, which returns that
    class, and :meth:`init`, which writes STILT-R's values; nothing on disk
    names it.

    Make a new project with :meth:`init` (or ``stilt init``). After that,
    ``config.yaml`` is yours to edit; PYSTILT never rewrites it.

    Parameters
    ----------
    path : str or Path
        Project directory.
    output : str or Path, optional
        Read and write results here instead of the output directory
        ``config.yaml`` names: a path, or a URL such as
        ``s3://bucket/output``. For a container whose output is somewhere
        else than the project's.

    Raises
    ------
    FileNotFoundError
        If the directory has no ``config.yaml``, so it is not a project: a
        mistyped path is not read as a project with no releases.

    Attributes
    ----------
    directory : Path
        The project directory, absolute.

    Examples
    --------
    Make a project, run it, and load a footprint:

    >>> import stilt  # doctest: +SKIP
    >>> release = stilt.PointRelease(  # doctest: +SKIP
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> project = stilt.Project.init(  # doctest: +SKIP
    ...     "./my_project",
    ...     releases=[release],
    ...     mets={"hrrr": "/data/hrrr"},
    ...     models={"hysplit": {"n_hours": -24}},
    ...     stages={
    ...         "footprint": {
    ...             "class": "stilt.footprint.FootprintStage",
    ...             "grid": {
    ...                 "xmin": -113,
    ...                 "xmax": -110.5,
    ...                 "ymin": 40,
    ...                 "ymax": 42,
    ...                 "xres": 0.01,
    ...                 "yres": 0.01,
    ...             },
    ...         }
    ...     },
    ...     variants={"hrrr": {}},
    ... )
    >>> project.run()  # doctest: +SKIP
    >>> foot = project.simulation(release.id, "hrrr").footprint  # doctest: +SKIP

    Open it again later and check a selection of simulations:

    >>> project = stilt.Project("./my_project")  # doctest: +SKIP
    >>> sims = project.simulations  # doctest: +SKIP
    >>> project.status(sims[sims.variant == "hrrr"])  # doctest: +SKIP
    """

    def simulation(
        self, release_id: str, variant: str, realization: int | None = None
    ) -> Simulation:
        """
        Return one simulation, with STILT's reads: a :class:`stilt.Simulation`.

        It is :meth:`runtraj.project.Project.simulation`, which finds the
        release and the variant, as the subclass that reads the footprint,
        the background, and the transport error. The two compare equal.

        Parameters
        ----------
        release_id : str
            The release's id, as in ``project.releases``.
        variant : str
            The variant's name.
        realization : int, optional
            Which realization of an ensemble variant (``realizations: N``),
            ``0`` to ``N - 1``. Leave it out for a variant that runs once.

        Raises
        ------
        KeyError
            If the project has no such release or variant.
        ValueError
            If *realization* is not one the variant runs as: ``0`` to
            ``N - 1`` for ``realizations: N``, ``None`` otherwise.

        Examples
        --------
        >>> sim = project.simulation(release_id, "hrrr")  # doctest: +SKIP
        >>> sim.footprint  # doctest: +SKIP
        """
        sim = super().simulation(release_id, variant, realization)
        return Simulation(
            sim.release, sim.variant, sim.output, sim.directory, sim.realization
        )

    @classmethod
    def init(
        cls,
        path: str | Path,
        releases: Release | Iterable[Release] | str | Path | None = None,
        *,
        config: ProjectConfig | None = None,
        starter: bool = False,
        **settings: Any,
    ) -> Self:
        """
        Make a new project and return it.

        Writes ``config.yaml``, with the settings given, and
        ``releases.csv`` when releases are given. With ``starter=True`` it
        writes the commented starter config instead, to edit by hand, as
        ``stilt init`` does.

        Either way, a project that runs HYSPLIT gets STILT-R's values where
        they differ from HYSPLIT's own defaults (:data:`STILT_R_HYSPLIT`)
        in its ``models: hysplit:`` block, unless the settings give them,
        so it runs as STILT-R does. Otherwise it is
        :meth:`runtraj.project.Project.init`.

        Parameters
        ----------
        path : str or Path
            Project directory. It is created if needed.
        releases : Release, iterable of Release, str or Path, optional
            Releases, or the path of a releases CSV to copy them from.
        config : ProjectConfig, optional
            The settings. Or give them as keywords instead.
        starter : bool, default False
            Write the commented starter ``config.yaml``
            (:data:`STARTER_CONFIG`) in place of a config. Without
            releases, ``releases.csv`` gets only its header.
        **settings
            The blocks of :class:`~stilt.ProjectConfig`: ``mets``, ``models``,
            ``stages`` and ``variants``, and ``execution`` and ``output``.

        Raises
        ------
        FileExistsError
            If the directory already has a ``config.yaml``. Open it with
            ``Project(path)`` instead, and edit the file to change settings.
        TypeError
            If both *config* and keyword settings are given, or either is
            given with ``starter=True``.
        ValueError
            If a variant's settings are invalid, or *releases* is given but
            holds none (an empty list, or a CSV file with only its header).
            Nothing is written.
        """
        if config is not None and settings:
            raise TypeError("Give a ProjectConfig or keyword settings, not both.")
        if not starter:
            config = _with_stilt_r_values(config or ProjectConfig(**settings))
            return super().init(path, releases, config=config)
        if config is not None or settings:
            raise TypeError(
                "A starter config takes no settings; edit its config.yaml instead."
            )
        starting = ProjectConfig.model_validate(yaml.safe_load(STARTER_CONFIG))
        new = super().init(path, releases, config=starting)
        # The same settings, with the comments that explain them.
        new.config_path.write_text(STARTER_CONFIG)
        if releases is None:
            # Header only: every other line of a releases file is a release.
            new.releases_path.write_text("time,longitude,latitude,altitude\n")
        return new

    def footprints(self, sel: Any = None) -> xr.Dataset:
        """
        Open the footprints of the selected simulations as one dataset.

        Footprints are stacked by their layer from the release time, so
        releases at any time share one layer axis, named for the unit
        ``tres`` is written in and counted in it: ``hour`` with the default
        ``tres`` of ``1h``, ``day`` for ``1D``, ``minute`` for ``15min``. The
        ``time`` coordinate says when each layer starts. Only the files' metadata is
        read here. The values load with dask when they are used, in blocks
        of releases from one date folder. The selection must hold one
        variant, as for :meth:`jacobian`.

        Parameters
        ----------
        sel : DataFrame or mask, optional
            The simulations, as for :meth:`status`, all of one variant (and
            one realization of an ensemble). All of them by default.

        Returns
        -------
        xarray.Dataset
            ``foot`` with dims ``(release, hour, lat, lon)`` (``y`` and
            ``x`` on a projected grid, and ``day`` or ``minute`` in place
            of ``hour`` when ``tres`` is written in them), in selection
            order. A release with
            an empty footprint has no row and is listed in
            ``attrs["empty"]``; one not run yet is listed in
            ``attrs["missing"]``.

        Raises
        ------
        ValueError
            If the selection holds more or fewer than one variant, or
            several realizations of an ensemble, the variant has no grid, or
            no selected simulation has a footprint file yet.

        Notes
        -----
        A footprint is about 35 MB of values on a 350,000-cell grid over 24
        hours, so a computation that loads every release at once suits a
        selection of hundreds. Sums over ``hour`` or the grid run block by
        block. :meth:`jacobian` sums any number of footprints onto a target.

        Examples
        --------
        >>> ds = project.footprints(sims[sims.variant == "hrrr"])  # doctest: +SKIP
        >>> ds.foot.sum("hour").mean("release").plot()  # doctest: +SKIP
        """
        frame = self._selected(sel)
        name, k = _one_group(frame, "A footprint dataset")
        variant = self.variants[name]
        if "footprint" not in variant.stages:
            raise ValueError(f"Variant {name!r} makes no footprints (no grid).")
        requested = list(dict.fromkeys(frame["release"]))
        found = self.output.present("footprint", variant, requested, k)
        paths = [
            path
            for r in requested
            if r in found
            and (path := self.output.path("footprint", variant, r, k)) is not None
        ]
        if not paths:
            raise ValueError(
                f"None of the {len(requested)} selected simulations of {name!r} "
                "has a footprint yet."
            )
        ds = open_footprints(paths)
        ds.attrs["missing"] = [r for r in requested if r not in found]
        return ds

    def jacobian(
        self,
        sel: Any,
        target: Geometry,
        time_bins: pd.IntervalIndex,
        *,
        workers: int | None = None,
    ) -> Jacobian:
        """
        Sum the selected footprints onto a target, per time bin, as one sparse matrix.

        The same operation as ``foot.stilt.aggregate``, for every selected
        footprint at once. Each footprint cell is split among the target
        cells it overlaps in proportion to area, time layers are summed
        within each of *time_bins*, and cells or layers outside the target
        or the bins are dropped. The selection must hold one variant, and
        one realization of an ensemble: ``ens[ens.realization == k]``.

        Footprints are read and summed one release at a time, in *workers*
        threads. Each thread holds one footprint, so memory stays at a few
        footprints whatever the selection's size; the result itself is
        sparse.

        Parameters
        ----------
        sel : DataFrame or mask
            The simulations, as for :meth:`status`, all of one variant (and
            one realization of an ensemble).
        target : Geometry
            Where the fluxes are: a grid, mesh, or set of zones.
        time_bins : pandas.IntervalIndex
            Flux time bins, such as a flux inventory's steps. They must be
            closed on the left (``closed="left"``): each bin holds the
            footprint layers that start in it. A layer goes whole into the
            bin where it starts, so the bins must be as long as the
            footprints' layers (``tres``) or longer. You build these
            yourself, from the inventory's steps read as
            ``foot.stilt.enhancement`` reads them (see
            :ref:`flux-step-labels`).
        workers : int, optional
            Threads that read and sum batches. Defaults to the CPUs this
            process may use (in a Slurm job, the job's), at most 8: more
            threads add memory and stop adding speed.

        Returns
        -------
        Jacobian
            Rows are the selected releases with a non-empty footprint, in
            selection order; columns are ``(time bin, target cell)``.
            Releases with an empty footprint are listed in ``empty``, and
            those not run yet in ``missing``.

        Raises
        ------
        TypeError
            If ``time_bins`` is not a :class:`pandas.IntervalIndex`.
        ValueError
            If the selection holds more or fewer than one variant, or
            several realizations of an ensemble, the variant has no grid or
            no footprints yet, ``time_bins`` is not closed on the left, or a
            bin is shorter than the footprints' layers (use a shorter
            ``tres``).
        """
        frame = self._selected(sel)
        name, k = _one_group(frame, "A Jacobian")
        variant = self.variants[name]
        if "footprint" not in variant.stages:
            raise ValueError(f"Variant {name!r} makes no footprints (no grid).")
        if self.output.folder("footprint", variant) is None:
            raise ValueError(f"Variant {name!r} has no footprints yet.")
        requested = list(dict.fromkeys(frame["release"]))
        found = self.output.present("footprint", variant, requested, k)
        return _jacobian(
            lambda r: self.output.read("footprint", variant, r, k),
            [r for r in requested if r in found],
            variant.stages["footprint"],
            target,
            time_bins,
            missing=[r for r in requested if r not in found],
            workers=workers,
        )

    def transport_error(
        self,
        sel: Any,
        error: str,
        flux: xr.DataArray,
        *,
        workers: int | None = None,
        **options: Any,
    ) -> pd.DataFrame:
        """
        Return the transport error of each selected simulation's modeled enhancement.

        The same estimate as ``sim.transport_error``, for every selected
        release at once (Lin and Gerbig, 2005; the *Transport Error* guide).
        *sel* is the unperturbed run, and *error* names the wind-error
        variant of the same releases. Every realization of *error* that has
        run is averaged, so a release whose ensemble is still running gets
        an estimate from the realizations it has.

        Releases are read and estimated one at a time, in *workers*
        threads. Each thread holds one release's particle tables: the
        unperturbed one and one per realization.

        Parameters
        ----------
        sel : DataFrame or mask
            The simulations, as for :meth:`status`, all of one variant that
            runs once (no ``realizations``).
        error : str
            The wind-error variant, such as ``"hrrr-err"``.
        flux : xarray.DataArray
            Surface flux field, in µmol m⁻² s⁻¹, the same for every release.
        workers : int, optional
            Threads that read and estimate releases. Defaults to the CPUs
            this process may use (in a Slurm job, the job's), at most 8.
        **options
            Passed to :func:`stilt.transport_error`: ``levels``,
            ``length_scale``, ``percentile``, and ``background``.

        Returns
        -------
        pandas.DataFrame
            The selection with six more columns, from
            :class:`~stilt.TransportError`: ``enhancement``,
            ``enhancement_perturbed``, ``background``, ``variance``,
            ``noise``, and ``realizations`` (how many realizations of
            *error* were averaged). A release without particles in the
            selected variant or in any realization of *error* gets NaN and
            ``realizations`` 0.

        Raises
        ------
        KeyError
            If the project has no variant named *error*.
        ValueError
            If the selection holds more or fewer than one variant, that
            variant is an ensemble, or it is *error* itself.

        Notes
        -----
        Each release's particles are read whole. A table of 1.4 million
        rows takes about a second to read, so a selection of hundreds of
        releases takes minutes. The per-level tables of
        :class:`~stilt.TransportError` are not kept; get them for
        one release with ``sim.transport_error``.

        Examples
        --------
        >>> sims = project.simulations  # doctest: +SKIP
        >>> tower = sims[
        ...     (sims.variant == "hrrr") & (sims.site == "tower_a")
        ... ]  # doctest: +SKIP
        >>> errors = project.transport_error(tower, "hrrr-err", flux)  # doctest: +SKIP
        >>> errors.groupby(errors.time.dt.month).variance.mean()  # doctest: +SKIP
        """
        frame = self._selected(sel)
        name, _ = _one_group(frame, "A transport error")
        variant = self.variants[name]
        if variant.realizations is not None:
            raise ValueError(
                f"Variant {name!r} is an ensemble (realizations: "
                f"{variant.realizations}); the selection is the unperturbed run, "
                "a variant that runs once."
            )
        if error not in self.variants:
            raise KeyError(
                f"No variant {error!r}; the variants are {list(self.variants)}."
            )
        if error == name:
            raise ValueError(
                f"The selection is of {error!r} itself; select the unperturbed variant."
            )
        perturbed = self.variants[error]
        requested = list(dict.fromkeys(frame["release"]))
        have = self.output.present("particles", variant, requested)
        ran = {
            k: self.output.present("particles", perturbed, requested, k)
            for k in perturbed.realization_numbers
        }
        # Each release with both sides, and the realizations of *error* it has.
        runs = {
            r: ks
            for r in requested
            if r in have and (ks := [k for k, found in ran.items() if r in found])
        }

        footprint = variant_footprint(variant)

        def particles(v: Variant, rid: str, k: int | None) -> pd.DataFrame:
            path = self.output.path("particles", v, rid, k)
            assert path is not None  # the listing found the file
            return stored_particles(
                read_particles(path),
                footprint,
                path,
                z_top=getattr(v.transport, "z_top", None),
                variant=v,
            )

        def read(rid: str) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
            return (
                particles(variant, rid, None),
                [particles(perturbed, rid, k) for k in runs[rid]],
            )

        found = _transport_errors(
            read,
            list(self._releases(runs).values()),
            flux,
            transforms=footprint.transforms,
            directory=self.directory,
            workers=_threads(workers),
            **{
                "hnf_plume": footprint.hnf_plume,
                "footprint_depth": footprint.footprint_depth,
                **options,
            },
        )
        result = found.reindex(frame["release"].to_numpy())
        result["realizations"] = result["realizations"].fillna(0).astype("int64")
        return frame.assign(**{c: result[c].to_numpy() for c in result.columns})


__all__ = ["STARTER_CONFIG", "STILT_R_HYSPLIT", "Project"]
