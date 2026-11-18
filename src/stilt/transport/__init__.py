"""
The transport model interface.

A transport model turns one receptor and one set of transport settings into
a particle table. HYSPLIT (:class:`stilt.transport.hysplit.HysplitModel`)
is the one built in; a model in another package, or PYSTILT's model of runs
imported from STILT-R (:class:`stilt.stilt_r.StiltRModel`), is named by its
import path. The worker reaches
a model through :func:`get_model` by the name a run's settings record, so
another model needs no change to the worker.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stilt._plugins import import_class
from stilt.exceptions import SimulationError
from stilt.meteorology import run_window
from stilt.particles import add_release_heights, check_particles

if TYPE_CHECKING:
    from stilt.receptors import Receptor


@dataclass(frozen=True)
class ModelRun:
    """
    What one model run produced.

    Attributes
    ----------
    particles : pandas.DataFrame
        The particle table, one row per particle per time step
        (:data:`stilt.particles.PARTICLE_SCHEMA`).
    log : str
        The model's log of the run, kept with the particles
        (:attr:`stilt.Simulation.log`). Empty for a model that writes none.
    met_files : list of Path
        The meteorology files the run read, recorded in the particle file
        (:func:`stilt.particles.particles_metadata`). Empty for a model that reads
        no files.
    """

    particles: pd.DataFrame
    log: str = ""
    met_files: list[Path] = field(default_factory=list)


class ModelInfo(BaseModel):
    """
    The transport model build a run was made with.

    Its name, the SHA-256 of its executable, and the SHA-256 of each data
    file it reads are part of the run's settings, so two builds
    never share a settings folder. The version the build reports is
    recorded beside the settings (``model_version`` in ``_settings.yaml``),
    not in them: two builds of one release report the same version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        description=(
            "The model as its ``models`` block in ``config.yaml`` names it: "
            "a built-in name such as ``hysplit``, or the import path of a "
            "model class (the block's ``class``)."
        ),
    )
    sha256: str | None = Field(
        None,
        description=(
            "SHA-256 of the executable the model runs, such as HYSPLIT's "
            "``hycs_std``. ``None`` for a model that runs none."
        ),
    )
    data_files: dict[str, str] | None = Field(
        None,
        description=(
            "SHA-256 of each data file the run read, by file name, such as "
            "HYSPLIT's tables. ``None`` for a model whose build fixes them."
        ),
    )
    version: str | None = Field(
        None,
        exclude=True,
        description=(
            "The version the build reports, such as ``hysplit.v5.1.0``. "
            "Recorded beside the run's settings and left out of them."
        ),
    )


class TransportConfig(BaseModel):
    """
    The base of every transport model's config.

    It holds the parameters PYSTILT's own code reads, whatever the model:
    the run length and the seed, which set the run window and the
    realizations. Everything else, the particle count included, is the
    model's own. A model's config subclasses it with its own
    parameters and is the model's ``config_class``. In ``config.yaml`` the
    parameters are the keys of the model's block under ``models:``, and a
    variant that runs the model may override any of them. A subclass overrides :meth:`settings` or
    :meth:`realizations` only where its model differs.
    """

    model_config = ConfigDict(extra="forbid")

    #: Fields that change no particle, such as where the build is, left out
    #: of a run's recorded settings.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset()

    n_hours: int = Field(
        description=(
            "Length of each simulation, in hours. Negative runs backward in "
            "time. It has no default: a config must say it."
        ),
    )

    @field_validator("n_hours")
    @classmethod
    def _n_hours_is_not_zero(cls, n_hours: int) -> int:
        """Refuse a run of no hours, which releases particles and follows none."""
        if n_hours == 0:
            raise ValueError(
                "n_hours must not be 0: use a negative number of hours for a "
                "backward run and a positive one for a forward run."
            )
        return n_hours

    seed: int | None = Field(
        None,
        description=(
            "Seed of the model's random numbers, for a reproducible run. "
            "Realization ``k`` of a variant runs with ``seed + k``."
        ),
    )

    def settings(self) -> dict[str, Any]:
        """Return what a run records of this config: every field but the :attr:`UNRECORDED` ones."""
        return self.model_dump(mode="json", exclude=set(self.UNRECORDED))

    def realizations(self, n: int) -> list[Self]:
        """
        Return *n* realizations of this config, realization ``k`` with ``seed + k``.

        Realization 0 uses the configured seed. Without a seed every
        realization is the same config, and the model must draw its own
        random numbers for them to differ.
        """
        seed = self.seed
        return [
            self.model_copy(update={"seed": None if seed is None else seed + k})
            for k in range(n)
        ]


class TransportModel(Protocol):
    """
    A transport model that follows particles from a receptor.

    Attributes
    ----------
    name : str
        The model's short name, such as ``"hysplit"``.
    needs_workdir : bool, optional
        Whether :meth:`run` is handed an empty directory for its files. True
        when the model does not say; a model that writes no files sets it
        False, and gets ``None``.
    batched : bool, optional
        Whether the model runs many receptors in one call
        (:meth:`run_many`). False when the model does not say. The worker
        then hands it every receptor of a variant that needs particles at
        once, where it otherwise calls :meth:`run` for one at a time.
    check_met : callable, optional
        ``check_met(met, receptors, n_hours)``, for a model that reads files,
        which ``stilt check`` calls to find what would stop its runs before
        any starts. *met* is the model's met config, *receptors* a table
        with one row per release point (``receptor``, ``time``,
        ``longitude``, ``latitude``), and *n_hours* the length of the runs.
        It returns a DataFrame with ``check`` (a short name, such as
        ``met_files``), ``receptor`` (the id, or ``NA`` for a problem of the
        met itself), and ``message`` columns, one row for each problem. It
        runs and downloads nothing. A model without it has no met files to
        check.
    fetch_met : callable, optional
        ``fetch_met(met, windows)``, for a model that reads files, which
        ``stilt fetch`` calls to get the files its runs need before any
        starts. *met* is the model's met config and *windows* the time each
        run covers (:func:`stilt.meteorology.run_window`). It gets the files
        a run of those windows would ask for, downloading what the met
        downloads, and returns a DataFrame with ``file`` and ``message``
        columns: a row for each file, then a row for each window whose files
        could not be had. A model without it has no met files to get.
    downloads_met : callable, optional
        ``downloads_met(met)``, for a model with ``fetch_met``: whether *met*
        downloads files, so that getting them once before the runs saves
        each run from downloading its own. ``stilt submit`` submits a fetch
        job before the job array only when it is true for a met the
        unfinished simulations use. A model without it downloads nothing.
    """

    name: str

    @property
    def config_class(self) -> type[TransportConfig]:
        """The model's config, a :class:`TransportConfig`, which validates its parameters."""
        ...

    @property
    def met_config_class(self) -> type[BaseModel]:
        """
        The model's met config, which validates an entry under ``mets:``.

        Its ``settings()`` returns what a run records of the met: which
        weather it is, and anything else that changes the particles, but not
        where its files are kept. HYSPLIT's is
        :class:`stilt.transport.hysplit.MetConfig`.
        """
        ...

    def sha256(self, config: Any) -> str | None:
        """
        Return the SHA-256 of the executable *config* would run, ``None`` for a model that runs none.

        Part of the run's identity: another build gives another settings
        folder. Raises ``FileNotFoundError`` when the executable cannot be
        found here; a project still opens, and its status compares stored
        folders without the build.
        """
        ...

    def version(self, config: Any) -> str | None:
        """Return the version the build *config* would run reports, recorded with the run but not part of its identity."""
        ...

    def data_files(self, config: Any) -> dict[str, str] | None:
        """
        Return the checksum of each data file a run with *config* reads, by file name.

        ``None`` for a model whose build fixes its data. These are part of
        the run's identity, since the data change the particles.
        """
        ...

    def run(
        self,
        receptor: Receptor,
        config: Any,
        met: Any,
        window: tuple[dt.datetime, dt.datetime],
        workdir: Path | None = None,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Run one receptor and return its particles.

        The particles are the model's own: PYSTILT checks them and adds the
        release heights, and ``foot`` when the model wrote none, afterwards
        (:func:`run_model`).

        Parameters
        ----------
        receptor : Receptor
            Where and when particles are released.
        config : TransportConfig
            The model's config, of its ``config_class``.
        met : BaseModel
            The meteorology, of the model's ``met_config_class``. In a
            project its directories are absolute.
        window : tuple of datetime
            The time the run covers, ``(start, end)`` in time order
            (:func:`stilt.meteorology.run_window`).
        workdir : Path, optional
            An empty directory for the run's files, kept when the run fails.
            ``None`` for a model that needs one to make its own.
        timeout : int, optional
            Time limit in seconds, for a model that runs a program.
            ``None`` waits indefinitely.

        Raises
        ------
        SimulationError
            When the run fails, with the model's log in ``log``.
        """
        ...


class BatchedTransportModel(TransportModel, Protocol):
    """
    A transport model that runs many receptors in one call, such as an emulator on a GPU.

    It sets ``batched = True`` and gives :meth:`run_many` beside
    :meth:`run`.
    """

    batched: bool

    def run_many(
        self,
        receptors: list[Receptor],
        config: Any,
        met: Any,
        windows: list[tuple[dt.datetime, dt.datetime]],
        workdir: Path | None = None,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Run many receptors with one config, and return their particles as one table.

        The table has a ``receptor`` column holding each row's receptor id,
        beside the particle columns :meth:`run` returns. ``windows[i]`` is
        the time ``receptors[i]`` covers. A receptor with no rows failed;
        the others are written. The log and met files are shared by all.

        Raises
        ------
        SimulationError
            When the whole call fails, with the model's log in ``log``.
        """
        ...


def _hysplit() -> TransportModel:
    """Return HYSPLIT, importing its package only when it is asked for."""
    from stilt.transport.hysplit import HysplitModel

    return HysplitModel()


#: The transport models built in, by the name of their block under
#: ``models:`` in ``config.yaml``: ``hysplit``. Any other model, such as
#: the one of runs imported from STILT-R (:class:`stilt.stilt_r.StiltRModel`),
#: is named by its import path instead, the ``class`` of its block, as a
#: transform's ``kind:`` is.
MODELS: dict[str, Callable[[], TransportModel]] = {
    "hysplit": _hysplit,
}


def get_model(name: str) -> TransportModel:
    """
    Return the transport model *name* names.

    *name* is a built-in model (:data:`MODELS`, such as ``hysplit``) or the
    import path of a model class, such as ``emulator.stilt.EmulatorModel``,
    which is imported and made with no arguments. A model in a package of
    its own needs no registration, so every process that opens the project
    finds it.

    Raises
    ------
    ValueError
        If *name* is not a built-in model and not an import path.
    MissingClass
        If *name* is an import path that cannot be imported on this machine
        (an :class:`ImportError`).
    """
    if "." in name:
        return import_class(name)()
    factory = MODELS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown transport model {name!r}. The built-in models are "
            f"{sorted(MODELS)}; another is named by its import path, such as "
            "mypkg.models.MyModel."
        )
    return factory()


def run_model(
    name: str,
    receptor: Receptor,
    config: TransportConfig,
    met: Any,
    workdir: Path | None = None,
    timeout: int | None = None,
) -> ModelRun:
    """
    Run the transport model called *name* for one receptor, and finish its particles.

    The particles are checked against the particle table
    (:func:`stilt.particles.check_particles`) and get their release heights
    (:func:`stilt.particles.add_release_heights`). These steps are
    PYSTILT's, the same for any model. Nothing else is added: ``foot`` is
    the model's own, or missing for a model that writes none, and the
    footprint stage computes one when it needs it
    (:func:`stilt.footprint.footprint_particles`). The near-field
    correction is not applied here either: :func:`stilt.calc_footprint`
    applies it when it makes a footprint (``hnf_plume``).

    Parameters
    ----------
    name : str
        The transport model (:data:`MODELS`).
    receptor : Receptor
        Where and when particles are released.
    config : TransportConfig
        The model's config.
    met : BaseModel
        The meteorology, of the model's ``met_config_class``.
    workdir : Path, optional
        An empty directory for the run's files. ``None`` lets the model make
        its own.
    timeout : int, optional
        Time limit in seconds.

    Raises
    ------
    SimulationError
        If the run fails, as when the meteorology does not cover it
        (``MET_COVERAGE``), or the model wrote no particles.
    """
    window = run_window(receptor.time, config.n_hours)
    run = get_model(name).run(receptor, config, met, window, workdir, timeout=timeout)
    return _finish(run, receptor, config)


def run_model_many(
    name: str,
    receptors: list[Receptor],
    config: TransportConfig,
    met: Any,
    workdir: Path | None = None,
    timeout: int | None = None,
) -> dict[str, ModelRun | SimulationError]:
    """
    Run a batched transport model once for many receptors, and finish each one's particles.

    The model's one table is split by its ``receptor`` column, and each
    receptor's particles get the steps :func:`run_model` applies. A receptor
    whose particles are missing or fail those steps gets its error, and the
    others their runs.

    Returns
    -------
    dict
        ``{receptor id: ModelRun}``, or the :class:`SimulationError` of a
        receptor that failed.

    Raises
    ------
    SimulationError
        If the whole call fails.
    """
    windows = [run_window(r.time, config.n_hours) for r in receptors]
    model = get_model(name)
    run_many = getattr(model, "run_many", None)
    if run_many is None:
        raise TypeError(f"Transport model {name!r} has no run_many.")
    run = run_many(receptors, config, met, windows, workdir, timeout=timeout)
    if "receptor" not in run.particles.columns:
        raise SimulationError(
            "A batched transport model's particles need a receptor column.",
            reason="NO_PARTICLE_DATA",
            log=run.log,
        )
    by_receptor = dict(tuple(run.particles.groupby("receptor", sort=False)))
    results: dict[str, ModelRun | SimulationError] = {}
    for receptor in receptors:
        rows = by_receptor.get(str(receptor.id))
        own = replace(
            run,
            particles=(
                pd.DataFrame()
                if rows is None
                else rows.drop(columns="receptor").reset_index(drop=True)
            ),
        )
        try:
            results[str(receptor.id)] = _finish(own, receptor, config)
        except SimulationError as error:
            error.log = run.log
            results[str(receptor.id)] = error
    return results


def _finish(run: ModelRun, receptor: Receptor, config: TransportConfig) -> ModelRun:
    """
    Return *run* with its particles checked and their release heights.

    Particles that stop before the end of the run are kept: the transport
    model fails a run cut short by its meteorology itself, and particles
    that all left the met's domain are a complete run. The particles keep
    the columns the model wrote: ``foot`` is the model's own, or missing.

    Raises
    ------
    SimulationError
        With the run's log, if the model wrote no particles.
    """
    if run.particles.empty:
        raise SimulationError(
            "The transport model wrote no particles.",
            reason="NO_PARTICLE_DATA",
            log=run.log,
        )
    check_particles(run.particles)
    return replace(run, particles=add_release_heights(run.particles, receptor))


def run_trajectories(
    receptor: Receptor,
    met: BaseModel | Mapping[str, Any],
    *,
    model: str = "hysplit",
    workdir: str | Path | None = None,
    timeout: int | None = None,
    **params: Any,
) -> pd.DataFrame:
    """
    Run the transport model for one receptor and return its particles.

    This is what a project runs for each receptor, without a project: the
    particles are returned, not stored. STILT-R calls it
    ``calc_trajectory``.

    Parameters
    ----------
    receptor : Receptor
        Where and when particles are released.
    met : str, dict or BaseModel
        The meteorology, as one entry under ``mets:`` in ``config.yaml``,
        checked by the model's met config
        (:class:`stilt.transport.hysplit.MetConfig` for HYSPLIT). A string is
        the folder of its files, and a relative folder starts from the
        working directory.
    model : str, default "hysplit"
        The transport model.
    workdir : str or Path, optional
        Directory the model runs in, kept afterwards for a look at its
        files. It must be empty or not exist yet. Without it the model runs
        in a temporary directory that is removed.
    timeout : int, optional
        Time limit in seconds. ``None`` waits indefinitely.
    **params
        The transport model's parameters, such as ``n_hours`` and
        ``numpar`` (:class:`stilt.transport.hysplit.HysplitConfig`).
        ``n_hours`` is required; the others default to the model's own
        values, which for HYSPLIT are not STILT-R's (see
        :data:`stilt.stilt_project.STILT_R_HYSPLIT`).

    Returns
    -------
    pandas.DataFrame
        The particle table, one row per particle per output step, with
        ``time`` (UTC), the receptor time plus ``age``, as a table read from
        a particle file has. ``foot`` is the model's own (HYSPLIT's, made
        with ``veght``), with no near-field correction:
        :func:`stilt.calc_footprint` applies it (``hnf_plume``), given
        ``veght`` as its ``footprint_depth``. STILT-R's ``calc_trajectory``
        applies it here. A model that writes no ``foot`` returns none:
        :func:`stilt.footprint.add_foot` computes one.

    Raises
    ------
    ValueError
        If *workdir* is not empty, or a parameter is not one of the
        model's.
    SimulationError
        If the run fails; ``reason`` says why.
    MeteorologyError
        If the met files the run needs are missing.

    Examples
    --------
    >>> receptor = stilt.PointReceptor(  # doctest: +SKIP
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> particles = stilt.run_trajectories(
    ...     receptor, "/data/hrrr", n_hours=-24, numpar=200
    ... )  # doctest: +SKIP
    """
    transport_model = get_model(model)
    config = transport_model.config_class(**params)
    met_config = transport_model.met_config_class.model_validate(met)
    if workdir is not None:
        workdir = Path(workdir)
        if workdir.exists() and any(workdir.iterdir()):
            raise ValueError(
                f"{workdir} is not empty. The model would read files left from "
                "another run; give an empty or new directory."
            )
        workdir.mkdir(parents=True, exist_ok=True)
    particles = run_model(
        model, receptor, config, met_config, workdir, timeout
    ).particles
    particles["time"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        particles["age"].to_numpy(), unit="min"
    )
    return particles


__all__ = [
    "MODELS",
    "BatchedTransportModel",
    "ModelInfo",
    "ModelRun",
    "TransportConfig",
    "TransportModel",
    "get_model",
    "run_model",
    "run_model_many",
    "run_trajectories",
]
