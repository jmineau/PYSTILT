"""
The transport model interface.

A transport model turns one receptor and one set of transport settings into
a particle table. Each model is a subpackage here. HYSPLIT
(:class:`stilt.transport.hysplit.HysplitModel`) is the only one today. The
worker reaches it through :func:`get_model` by the name a run's settings
record, so a second model needs no change to the worker.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stilt.exceptions import SimulationError
from stilt.meteorology import run_window
from stilt.particles import (
    HNF_PLUME_COLUMNS,
    add_release_heights,
    check_particles,
    correct_near_field,
)

if TYPE_CHECKING:
    from stilt.receptors import Receptor

logger = logging.getLogger(__name__)


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
    """The transport model build a run was made with, recorded in the run's settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        description=(
            "The model as ``model:`` names it: a built-in name such as "
            "``hysplit``, or the import path of a model class."
        ),
    )
    version: str = Field(
        ..., description="Version string of the build, such as ``v5.1.0``."
    )
    data_files: dict[str, str] | None = Field(
        None,
        description=(
            "SHA-256 of each data file the run used in place of the model's "
            "own, by file name. ``None`` when it used the model's own."
        ),
    )


class TransportConfig(BaseModel):
    """
    The base of every transport model's config.

    It holds the parameters PYSTILT's own code reads, whatever the model:
    the run length and the seed, which set the run window and the
    realizations, and the near-field settings PYSTILT applies to any
    model's particles. Everything else, the particle count included, is the
    model's own. A model's config subclasses it with its own
    parameters and is the model's ``config_class``. In ``config.yaml`` the
    parameters are flat, top-level keys for the project's model, and a
    variant that names another model inherits the shared ones and gives
    that model's own. A subclass overrides :meth:`settings` or
    :meth:`realizations` only where its model differs.
    """

    model_config = ConfigDict(extra="forbid")

    #: Fields that change no particle, such as where the build is, left out
    #: of a run's recorded settings.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset()

    n_hours: int = Field(
        -24,
        description="Length of each simulation, in hours. Negative runs backward in time.",
    )
    hnf_plume: bool = Field(
        True,
        description=(
            "Apply a vertical Gaussian plume model to particles in the hyper "
            "near-field. This shrinks their effective dilution depth and raises "
            "the influence of fluxes close to the receptor. Needs the particle "
            "columns ``dens``, ``tlgr``, ``sigw``, ``foot``, ``mlht``, and "
            "``samt``."
        ),
    )
    veght: float = Field(
        0.5,
        description=(
            "Height below which a particle's time counts toward the footprint. "
            "A value of 1 or less is a fraction of the mixed-layer height; a "
            "larger value is meters above ground."
        ),
    )
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

    def version(self, config: Any) -> str:
        """Return the version of the build *config* would run, recorded with the run."""
        ...

    def data_files(self, config: Any) -> dict[str, str] | None:
        """
        Return the checksum of each data file *config* replaces, by file name.

        ``None`` when the run uses the model's own data files. These are part
        of the run's identity, since the data change the particles.
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
        release heights and the near-field correction afterwards
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


#: The transport models built into PYSTILT, by the name ``model:`` takes in
#: ``config.yaml``. A model in its own package is named by its import path
#: instead, as a transform's ``kind:`` is.
MODELS: dict[str, Callable[[], TransportModel]] = {"hysplit": _hysplit}


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
    ImportError
        If *name* is an import path that cannot be imported on this machine.
    """
    if "." in name:
        module_name, _, attr = name.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, attr)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"Transport model {name!r} could not be imported ({error}). "
                "Install the package that defines it on this machine."
            ) from None
        return cls()
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
    (:func:`stilt.particles.check_particles`), get their release heights
    (:func:`stilt.particles.add_release_heights`), and, when
    ``config.hnf_plume`` is set, the near-field correction
    (:func:`stilt.particles.correct_near_field`). These steps are PYSTILT's,
    the same for any model. A model whose particles lack the columns the
    correction reads gets none, and the run's log says so.

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
        If the run fails, the model wrote no particles, or they stop before
        the end of the run (``MET_COVERAGE``, :func:`check_reach`).
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


def reach_minutes(particles: pd.DataFrame) -> float:
    """Return how far the particles got from the release, in minutes: the largest ``|time|``."""
    return float(np.abs(particles["time"].to_numpy(dtype=float)).max())


def check_reach(particles: pd.DataFrame, n_hours: int) -> None:
    """
    Raise if no particle reaches the end of the run.

    A run is complete when its furthest particle gets to ``n_hours`` from
    the release, within one output step (the smallest gap between the
    particle table's times). Particles stop early when the
    meteorology ends, or when every one has left the meteorology's domain
    (or its crop). Either way the footprint would hold only part of the
    run, so the simulation fails, as ``MET_COVERAGE``, whatever the model.

    Raises
    ------
    SimulationError
        With ``reason`` ``MET_COVERAGE``.
    """
    times = np.unique(np.abs(particles["time"].to_numpy(dtype=float)))
    end = abs(n_hours) * 60
    step = float(np.diff(times).min()) if len(times) > 1 else 0.0
    reach = float(times[-1])
    if reach >= end - step:
        return
    raise SimulationError(
        f"The particles stop {reach / 60:.3g} h into a {end / 60:g} h run: the "
        "meteorology does not cover the whole run, or every particle left its "
        "domain.",
        reason="MET_COVERAGE",
    )


def _finish(run: ModelRun, receptor: Receptor, config: TransportConfig) -> ModelRun:
    """
    Return *run* with its particles checked, their release heights, and the near-field correction.

    Raises
    ------
    SimulationError
        With the run's log, if the model wrote no particles, or they stop
        before the end of the run (:func:`check_reach`).
    """
    if run.particles.empty:
        raise SimulationError(
            "The transport model wrote no particles.",
            reason="NO_PARTICLE_DATA",
            log=run.log,
        )
    check_particles(run.particles)
    try:
        check_reach(run.particles, config.n_hours)
    except SimulationError as error:
        error.log = f"{run.log}{error}\n"
        raise
    particles = add_release_heights(run.particles, receptor)
    if config.hnf_plume:
        missing = sorted(set(HNF_PLUME_COLUMNS) - set(particles.columns))
        if missing:
            note = f"near-field correction skipped: no {', '.join(missing)} columns"
            logger.warning("%s: %s", receptor.id, note)
            return replace(run, particles=particles, log=f"{run.log}{note}\n")
        particles = correct_near_field(particles, receptor, config.veght)
    return replace(run, particles=particles)


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
    met : dict or BaseModel
        The meteorology, as one entry under ``mets:`` in ``config.yaml``,
        checked by the model's met config
        (:class:`stilt.transport.hysplit.MetConfig` for HYSPLIT). A relative
        ``directory`` starts from the working directory.
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

    Returns
    -------
    pandas.DataFrame
        The particle table, one row per particle per output step.

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
    >>> receptor = stilt.PointReceptor(
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
    >>> particles = stilt.run_trajectories(receptor, met, n_hours=-24, numpar=200)
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
    return run_model(model, receptor, config, met_config, workdir, timeout).particles


__all__ = [
    "MODELS",
    "BatchedTransportModel",
    "ModelInfo",
    "ModelRun",
    "TransportConfig",
    "TransportModel",
    "check_reach",
    "get_model",
    "run_model",
    "run_model_many",
    "run_trajectories",
]
