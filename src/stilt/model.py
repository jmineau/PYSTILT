"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the R-STILT model framework.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.collections import (
    FootprintCollection,
    ReceptorCollection,
    SimulationCollection,
    TrajectoryCollection,
)
from stilt.config import (
    ModelConfig,
    RuntimeSettings,
    STILTParams,
)
from stilt.config.model import _config_or_kwargs
from stilt.errors import ConfigValidationError
from stilt.execution import (
    Executor,
    JobHandle,
    LocalHandle,
    SlurmExecutor,
    get_executor,
    sigterm_as_interrupt,
)
from stilt.meteorology import MetStream
from stilt.project import Project
from stilt.receptors import Receptor
from stilt.service import PostgresQueue, resolve_queue
from stilt.simulation import SimID, Simulation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.visualization import ModelPlotAccessor


@dataclass(frozen=True, slots=True)
class StatusCounts:
    """Completion counts for one project."""

    total: int = 0
    completed: int = 0
    pending: int = 0


class Model:
    """
    Science-facing STILT project interface.

    ``Model`` is the primary Python entry point for configuring a STILT project,
    running simulations, and loading results.

    A project is one root — a local directory or object-store URI — holding
    ``config.yaml``, ``receptors.csv``, and ``simulations/by-id/``. The
    simulations a model defines are its receptors crossed with its met
    streams; whether each is complete is read from the outputs by key.

    Parameters
    ----------
    project : str or Path or None, optional
        Project root. A temporary directory when omitted.
    receptors : Receptor or iterable or str or Path or None, optional
        In-memory receptors or a path to a receptor CSV. Defaults to the
        project's ``receptors.csv``.
    config : ModelConfig or None, optional
        In-memory project config. Defaults to the project's ``config.yaml``.
    compute_root : str or Path or None, optional
        Local parent directory under which worker simulation directories are
        created. Defaults to the project's ``simulations/by-id`` for local
        projects and a temp directory for cloud projects.
    runtime : RuntimeSettings or None, optional
        Runtime-only deployment settings (cache root, DB URL, compute root).
        Read from ``PYSTILT_*`` environment variables when omitted.
    **kwargs
        Forwarded to :class:`~stilt.config.ModelConfig` when *config* is not
        provided. Mutually exclusive with *config*.

    Attributes
    ----------
    project : Project
        Project root and store.
    config : ModelConfig
    receptors : ReceptorCollection
    mets : dict[str, MetStream]
    simulations : SimulationCollection
    trajectories : TrajectoryCollection
    footprints : FootprintCollection
    plot : ModelPlotAccessor
    queue : PostgresQueue or None
        Work queue for pull/serve workers; ``None`` unless ``PYSTILT_DB_URL`` is set.
    """

    def __init__(
        self,
        project: str | Path | None = None,
        receptors: Receptor | Iterable | str | Path | None = None,
        config: ModelConfig | None = None,
        compute_root: str | Path | None = None,
        runtime: RuntimeSettings | None = None,
        **kwargs,
    ):
        self.runtime = runtime if runtime is not None else RuntimeSettings()
        self.project = Project(project, cache_dir=self.runtime.cache_dir)
        self.compute_root = self._resolve_compute_root(compute_root)
        self._config = _config_or_kwargs(config, kwargs, ModelConfig)

        self._mets: dict[str, MetStream] | None = None
        self._params: STILTParams | None = None
        self._receptors: ReceptorCollection | None = None
        self._receptors_input = receptors
        self._queue: PostgresQueue | None = None
        self._simulations: SimulationCollection | None = None
        self._trajectories: TrajectoryCollection | None = None
        self._footprints: FootprintCollection | None = None
        self._plot: ModelPlotAccessor | None = None

    def __repr__(self) -> str:
        return f"Model(project={self.project.root!r})"

    @property
    def name(self) -> str:
        """Human-readable project name."""
        return self.project.name

    def _resolve_compute_root(self, compute_root: str | Path | None) -> Path:
        """Return the parent directory under which worker sim dirs are created."""
        if compute_root is not None:
            raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
            return Path(raw).resolve()
        if self.runtime.compute_root is not None:
            return self.runtime.compute_root.expanduser().resolve()
        if not self.project.is_cloud:
            return self.project.simulations_dir
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        return Path(tmp_root) / "pystilt" / self.project.name

    # -- Inputs ----------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """Project config, loaded from ``config.yaml`` if not provided at construction."""
        if self._config is None:
            self._config = self.project.load_config()
        return self._config

    @property
    def receptors(self) -> ReceptorCollection:
        """Receptors, by position (``receptors[0]``) or id (``receptors[sim_id.receptor]``)."""
        if self._receptors is None:
            self._receptors = ReceptorCollection(
                self._receptors_input, project=self.project
            )
        return self._receptors

    @property
    def params(self) -> STILTParams:
        """Transport parameters shared by every simulation (from config)."""
        if self._params is None:
            self._params = self.config.to_stilt_params()
        return self._params

    @property
    def mets(self) -> dict[str, MetStream]:
        """Named met streams, resolved from config."""
        if self._mets is None:
            self._mets = {
                name: MetStream(
                    name,
                    directory=cfg.directory,
                    file_format=cfg.file_format,
                    file_tres=cfg.file_tres,
                    n_min=cfg.n_min,
                    source_type=cfg.source,
                    source_kwargs=cfg.source_kwargs,
                    backend=cfg.backend,
                    subgrid_enable=cfg.subgrid_enable,
                    subgrid_bounds=cfg.subgrid_bounds,
                    subgrid_buffer=cfg.subgrid_buffer,
                    subgrid_levels=cfg.subgrid_levels,
                    subgrid_dir=cfg.subgrid_dir,
                )
                for name, cfg in self.config.mets.items()
            }
        return self._mets

    @property
    def queue(self) -> PostgresQueue | None:
        """Postgres work queue, present only when ``PYSTILT_DB_URL`` is configured."""
        if self._queue is None and self.runtime.db_url:
            self._queue = resolve_queue(self.runtime)
        return self._queue

    def register(self, receptors: Iterable[Receptor] | None = None) -> list[str]:
        """
        Persist the model's inputs to the project and return its simulation ids.

        Writes ``config.yaml`` and ``receptors.csv`` into the project store so
        that workers (local processes, Slurm tasks, Kubernetes pods) can
        rebuild this model from the root alone. When a work queue is
        configured, the simulations are enqueued as pending.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to add. They are merged into the project's existing
            receptors (deduplicated by id) and the merged set is written.
            When omitted, the model's own receptors are persisted — copying
            the source CSV byte-for-byte when they came from a file.

        Returns
        -------
        list[str]
            Simulation ids for the receptors registered by this call.
        """
        self.project.save_config(self.config)

        if receptors is None:
            batch = list(self.receptors)
            source = self.receptors.source_path
            if source is not None:
                self.project.copy_receptors(source)
            elif not self.project.has_receptors:
                self.project.save_receptors(batch)
        else:
            batch = list(receptors)
            existing = self.project.load_receptors() or []
            merged = {r.id: r for r in existing}
            merged.update({r.id: r for r in batch})
            self.project.save_receptors(list(merged.values()))
            # The registered set changed: rebuild receptors from the project.
            self._receptors_input = None
            self._receptors = None
            self._simulations = None

        sim_ids = [
            str(SimID.from_parts(met, receptor))
            for met in self.mets
            for receptor in batch
        ]
        if self.queue is not None:
            self.queue.register(sim_ids)
        return sim_ids

    # -- Simulations -----------------------------------------------------------

    def simulation(self, sim_id: str) -> Simulation:
        """Build a handle for one simulation id (no side effects on disk)."""
        sid = SimID(sim_id)
        return Simulation(
            directory=self.compute_root / sid,
            receptor=self.receptors[sid.receptor],
            params=self.params,
            meteorology=self.mets[sid.met],
            store=self.project.store,
        )

    @property
    def simulations(self) -> SimulationCollection:
        """Simulations this model defines (receptors × mets)."""
        if self._simulations is None:
            self._simulations = SimulationCollection(self)
        return self._simulations

    @property
    def trajectories(self) -> TrajectoryCollection:
        """Cross-simulation trajectory accessor."""
        if self._trajectories is None:
            self._trajectories = TrajectoryCollection(self)
        return self._trajectories

    @property
    def footprints(self) -> FootprintCollection:
        """Cross-simulation footprint accessor namespace."""
        if self._footprints is None:
            self._footprints = FootprintCollection(self)
        return self._footprints

    @property
    def plot(self) -> ModelPlotAccessor:
        """Plotting namespace (e.g. ``model.plot.availability()``)."""
        if self._plot is None:
            from stilt.visualization import ModelPlotAccessor

            self._plot = ModelPlotAccessor(self)
        return self._plot

    def status(self) -> StatusCounts:
        """Return completion counts (total / completed / pending), read from the outputs."""
        sim_ids = self.simulations.keys()
        if not sim_ids:
            return StatusCounts()
        incomplete = len(self.simulations.incomplete())
        return StatusCounts(
            total=len(sim_ids), completed=len(sim_ids) - incomplete, pending=incomplete
        )

    # -- Execution -------------------------------------------------------------

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool | None = None,
        wait: bool = True,
    ) -> JobHandle:
        """
        Persist inputs, then start workers for every incomplete simulation.

        When ``config.footprints`` is non-empty, workers run HYSPLIT as needed
        and compute every footprint in one pass; otherwise only trajectories
        are produced.

        Parameters
        ----------
        executor : Executor, optional
            Override the executor resolved from ``config.execution``.
        skip_existing : bool or None, optional
            Skip simulations whose outputs are all present. ``None`` (default)
            reads ``config.skip_existing``.
        wait : bool, optional
            Block until workers finish (default). ``False`` returns the
            :class:`JobHandle` immediately — fire-and-forget for Slurm.

        Returns
        -------
        JobHandle
        """
        self._simulations = None
        resolved_skip = (
            skip_existing if skip_existing is not None else self.config.skip_existing
        )
        resolved_executor = executor or get_executor(self.config.execution or {})
        if isinstance(resolved_executor, SlurmExecutor) and self.project.is_cloud:
            raise ConfigValidationError(
                "Slurm execution currently requires a local project root."
            )

        sim_ids = self.register()
        if not sim_ids:
            logger.info("run: no receptors configured — nothing to do")
            return LocalHandle()

        pending = self.simulations.incomplete() if resolved_skip else list(sim_ids)
        if not pending:
            logger.info("run: all simulations already complete — nothing to do")
            return LocalHandle()

        names = list(self.config.footprints)
        logger.info(
            "run(%s): starting %s workers for %d simulations",
            ", ".join(names) if names else "trajectories",
            resolved_executor.dispatch,
            len(pending),
        )

        handle = resolved_executor.start(
            pending,
            project=self.project.root,
            compute_root=str(self.compute_root),
            skip_existing=resolved_skip,
        )
        if wait:
            logger.info("run: waiting for workers to finish...")
            try:
                with sigterm_as_interrupt():
                    handle.wait()
            except KeyboardInterrupt:
                logger.warning("run interrupted")
                raise

        return handle


__all__ = ["Model", "StatusCounts"]
