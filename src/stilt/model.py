"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the R-STILT model framework.
"""

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.collections import (
    FootprintCollection,
    ReceptorCollection,
    SimulationCollection,
    TrajectoryCollection,
)
from stilt.completion import StatusCounts, expected_for_config, is_complete
from stilt.config import (
    ModelConfig,
    RuntimeSettings,
    foot_names,
    resolve_runtime_settings,
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
from stilt.manifest import Manifest
from stilt.meteorology import MetStream
from stilt.receptors import Receptor
from stilt.service import PostgresQueue, resolve_queue
from stilt.simulation import SimID
from stilt.storage import (
    ProjectFiles,
    ProjectLayout,
    Storage,
    make_store,
    project_slug,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.visualization import ModelPlotAccessor


class Model:
    """
    Science-facing STILT project interface.

    ``Model`` is the primary Python entry point for configuring a STILT project,
    running one-off simulations, and loading simulation results.

    Completion is read from the outputs by key; the registry is the manifest.
    A Postgres work queue (``model.queue``) is used only for pull/serve workers,
    exposed through the CLI and :func:`stilt.execution.pull_simulations`.

    Parameters
    ----------
    project : str or Path or None, optional
        Local project root or cloud URI used to identify the model.
    receptors : Receptor or iterable or str or Path or None, optional
        In-memory receptors or a path to a receptor CSV.
    config : ModelConfig or None, optional
        In-memory project config. When omitted, config is loaded lazily from storage.
    output_dir : str or Path or None, optional
        Output root. Defaults to ``project``.
    compute_root : str or Path or None, optional
        Local parent directory where worker simulation directories are created.
    runtime : RuntimeSettings or None, optional
        Runtime-only deployment settings such as cache roots and DB URLs.
    **kwargs
        Forwarded to :class:`~stilt.config.ModelConfig` when *config* is not
        provided.  Any valid ``ModelConfig`` field name is accepted (e.g.
        ``n_hours=-48``, ``numpar=500``).  Mutually exclusive with *config* —
        passing both raises ``TypeError``.

    Attributes
    ----------
    config : ModelConfig
        Lazily loaded project configuration.
    receptors : ReceptorCollection
        Science-facing receptor accessor.
    simulations : SimulationCollection
        Registered simulation handles backed by the manifest registry.
    mets : dict[str, MetStream]
        Named meteorology streams.
    trajectories : TrajectoryCollection
        Cross-simulation trajectory accessor.
    footprints : FootprintCollection
        Cross-simulation footprint accessor namespace.
    plot : ModelPlotAccessor
        Plotting namespace for model summaries and outputs.
    queue : PostgresQueue or None
        Postgres work queue for pull/serve workers (``None`` locally).
    storage : Storage
        Output bootstrap and output storage facade.
    runtime : RuntimeSettings
        Runtime-only deployment settings.
    layout : ProjectLayout
        Resolved project/output refs (``project_dir``, ``output_dir``,
        ``project_root``, ``output_root``, ``is_cloud_project``, ``is_cloud_output``).
    """

    def __init__(
        self,
        project: str | Path | None = None,
        receptors: Receptor | Iterable | str | Path | None = None,
        config: ModelConfig | None = None,
        output_dir: str | Path | None = None,
        compute_root: str | Path | None = None,
        runtime: RuntimeSettings | None = None,
        **kwargs,
    ):
        # Runtime settings are not persisted in the project config, but they may be
        # needed to resolve the work queue backend, so resolve them first.
        self.runtime = resolve_runtime_settings(runtime)

        # The project layout resolver also handles output_dir defaulting and cloud URI parsing.
        self.layout = ProjectLayout.resolve(project, output_dir)
        self.project = self._project_name(self.layout.project_ref)
        self.storage = Storage(
            project_dir=self.layout.project_dir,
            output_dir=self.layout.output_dir,
            store=make_store(self.layout.output_root, cache_dir=self.runtime.cache_dir),
            is_cloud_project=self.layout.is_cloud_project,
        )

        # Compute root is not persisted in the project config, but it may be needed to
        # resolve the met archive and worker sim dirs, so resolve it early.
        self.compute_root = self._resolve_compute_root(compute_root)

        self._config = _config_or_kwargs(config, kwargs, ModelConfig)

        # Lazy caches for expensive properties
        self._mets: dict[str, MetStream] | None = None
        self._receptors = receptors  # may be None, a Receptor, or an iterable of Receptors; resolved lazily in the accessor
        self._queue: PostgresQueue | None = None
        self._manifest: Manifest | None = None
        self._simulations: SimulationCollection | None = None
        self._trajectories: TrajectoryCollection | None = None
        self._footprints: FootprintCollection | None = None
        self._plot: ModelPlotAccessor | None = None

    def __repr__(self) -> str:
        """Compact developer-facing model representation."""
        return (
            f"Model(project={self.project!r}, output_root={self.layout.output_root!r})"
        )

    @property
    def manifest(self) -> Manifest:
        """Registry of registered simulations (``.stilt/manifest.parquet``)."""
        if self._manifest is None:
            self._manifest = Manifest(self.storage.store)
        return self._manifest

    @property
    def queue(self) -> PostgresQueue | None:
        """
        Postgres work queue, present only when a DB URL is configured.

        Local projects have no queue: the registry is the manifest and
        completion is computed by key. The queue exists only to distribute work
        to pull/serve workers.
        """
        if self._queue is None and self.runtime.db_url:
            self._queue = resolve_queue(self.runtime)
        return self._queue

    def _resolve_compute_root(self, compute_root: str | Path | None) -> Path:
        """Return the parent directory under which worker sim dirs are created."""
        if compute_root is not None:
            raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
            return Path(raw).resolve()
        if self.runtime.compute_root is not None:
            return self.runtime.compute_root.expanduser().resolve()
        if not self.layout.is_cloud_output:
            return ProjectFiles(self.layout.output_dir).by_id_dir
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        return Path(tmp_root) / "pystilt"

    def _project_name(self, project_str: str) -> str:
        """Return a human-readable project name for local paths and cloud URIs."""
        if self.layout.is_cloud_project and project_str:
            return project_slug(project_str)
        return self.layout.project_dir.name

    @property
    def config(self) -> ModelConfig:
        """
        Project :class:`ModelConfig`, loaded from ``config.yaml`` if not provided at construction.

        Returns
        -------
        ModelConfig
        """
        if self._config is None:
            self._config = self.storage.load_config()
        return self._config

    @property
    def receptors(self) -> ReceptorCollection:
        """
        Sequence-like receptor accessor for this project.

        Access by position (``model.receptors[0]``) or by receptor identifier
        (``model.receptors[sim_id.receptor_id]``).

        Returns
        -------
        ReceptorCollection
        """
        if not isinstance(self._receptors, ReceptorCollection):
            self._receptors = ReceptorCollection(
                self._receptors,
                storage=self.storage,
            )
        return self._receptors

    def register_pending(
        self,
        receptors: Iterable[Receptor] | None = None,
        *,
        scene_id: str | None = None,
    ) -> list[str]:
        """
        Persist model inputs and register one batch of pending work.

        This is the output registration boundary shared by local runs,
        queue-backed workers, and the CLI ``stilt register`` command.

        Registration is idempotent: calling this multiple times with the same
        receptors does not create duplicate registry entries.  The manifest
        uses an upsert so existing
        rows are updated in place rather than duplicated.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to register. When omitted, the model's current receptor
            inputs are used.
        scene_id : str, optional
            Optional grouping label stored on all registered simulations.

        Returns
        -------
        list[str]
            Registered simulation identifiers.
        """
        recs = tuple(receptors) if receptors is not None else tuple(self.receptors)
        source_path = self.receptors.source_path if receptors is None else None

        self.storage.publish_config(self.config)
        self.storage.publish_receptors(
            None if source_path is not None else list(recs),
            source_path=source_path,
        )

        pairs = tuple(
            (str(SimID.from_parts(met_name, receptor)), receptor)
            for met_name in self.mets
            for receptor in recs
        )
        foot = foot_names(dict(self.config.footprints))
        self.manifest.register(list(pairs), footprint_names=foot, scene_id=scene_id)
        # When a queue is configured, also seed it so pull/serve workers can claim.
        if self.queue is not None:
            self.queue.register(list(pairs), scene_id=scene_id)

        # Registration updates the registry, so drop
        # cached accessors — next access rebuilds from that output surface.
        self._receptors = None
        self._simulations = None

        return [sim_id for sim_id, _ in pairs]

    @property
    def simulations(self) -> SimulationCollection:
        """
        Lazy simulation collection for registered simulations.

        Returns
        -------
        SimulationCollection
        """
        if self._simulations is None:
            params = self.config.to_stilt_params()
            self._simulations = SimulationCollection(
                self.layout.output_dir,
                params,
                self.mets,
                self.receptors,
                list(self.config.footprints),
                self.manifest,
                self.storage.store,
            )
        return self._simulations

    @property
    def plot(self) -> "ModelPlotAccessor":
        """
        Plotting namespace for this model (e.g. ``model.plot.availability()``).

        Returns
        -------
        ModelPlotAccessor
        """
        if self._plot is None:
            from stilt.visualization import ModelPlotAccessor

            self._plot = ModelPlotAccessor(self)
        return self._plot

    @property
    def trajectories(self) -> TrajectoryCollection:
        """Trajectory accessor for querying and loading simulation outputs."""
        if self._trajectories is None:
            self._trajectories = TrajectoryCollection(self)
        return self._trajectories

    @property
    def footprints(self) -> FootprintCollection:
        """Footprint accessor namespace for named footprint outputs."""
        if self._footprints is None:
            self._footprints = FootprintCollection(self)
        return self._footprints

    @property
    def mets(self) -> dict[str, MetStream]:
        """Named met streams, resolved from config."""
        if self._mets is None:
            self._mets = {}
            for name, config in self.config.mets.items():
                self._mets[name] = MetStream(
                    name,
                    directory=config.directory,
                    file_format=config.file_format,
                    file_tres=config.file_tres,
                    n_min=config.n_min,
                    source_type=config.source,
                    source_kwargs=config.source_kwargs,
                    backend=config.backend,
                    subgrid_enable=config.subgrid_enable,
                    subgrid_bounds=config.subgrid_bounds,
                    subgrid_buffer=config.subgrid_buffer,
                    subgrid_levels=config.subgrid_levels,
                    subgrid_dir=config.subgrid_dir,
                )
        return self._mets

    # -- Queries --------------------------------------------------------------

    def _counts(self, sim_ids: list[str]) -> StatusCounts:
        """Aggregate completion (by key) over a set of registered sim ids."""
        if not sim_ids:
            return StatusCounts()
        expected = expected_for_config(self.config)
        total = len(sim_ids)
        completed = sum(
            1 for sim_id in sim_ids if is_complete(sim_id, expected, self.storage)
        )
        return StatusCounts(total=total, completed=completed, pending=total - completed)

    def status(self, scene_id: str | None = None) -> StatusCounts:
        """
        Return completion counts (total/completed/pending) from the registry.

        Counts are derived from the manifest registry plus by-key completion, so
        they reflect the outputs on disk. ``running``/``failed`` are execution
        states tracked only by the queue and are left zero here.
        """
        if scene_id is not None:
            sim_ids = self.manifest.sim_ids_by_scene().get(scene_id, [])
        else:
            sim_ids = self.manifest.sim_ids()
        return self._counts(sim_ids)

    def scene_counts(self) -> dict[str, StatusCounts]:
        """Return completion counts grouped by registered scene."""
        return {
            scene: self._counts(sim_ids)
            for scene, sim_ids in self.manifest.sim_ids_by_scene().items()
        }

    # -- Execution ------------------------------------------------------------

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool | None = None,
        rebuild: bool | None = None,
        wait: bool = True,
    ) -> JobHandle:
        """
        Register pending work and start workers to drain it.

        When ``config.footprints`` contains one or more footprint
        configurations, workers auto-run HYSPLIT trajectories as needed and
        then compute footprints in a single pass.  When no footprint configs
        are defined, only HYSPLIT trajectories are dispatched.

        Workers either drain the Postgres work queue via
        :func:`~stilt.execution.pull_simulations` or consume immutable chunk
        shards via :func:`~stilt.execution.push_simulations`; the coordinator
        registers simulations and starts workers, then optionally blocks.

        Parameters
        ----------
        executor : Executor, optional
            Override the executor resolved from ``config.execution``.
        skip_existing : bool or None, optional
            Skip simulations that already have output.  ``None`` (default)
            reads the value from ``config.skip_existing``.
        rebuild : bool or None, optional
            Deprecated no-op. Completion is always read from the outputs by key,
            so there is nothing to rebuild before planning.
        wait : bool, optional
            If ``True`` (default), block until all workers finish. If ``False``,
            return the :class:`JobHandle` immediately — suitable for
            fire-and-forget Slurm runs.

        Returns
        -------
        JobHandle
            A handle you can call ``.wait()`` on later, or ignore.

        Notes
        -----
        ``Model.run()`` is the main science-facing execution path for local or
        notebook use. For claim-based service workflows, use
        :func:`stilt.execution.pull_simulations` against a model configured with
        a PostgreSQL-backed queue, or use the CLI ``stilt pull-worker`` and
        ``stilt serve`` commands.
        """
        # Execution may produce new outputs, so drop any cached simulation view.
        self._simulations = None
        resolved_skip = (
            skip_existing if skip_existing is not None else self.config.skip_existing
        )
        resolved_executor = executor or get_executor(self.config.execution or {})
        if isinstance(resolved_executor, SlurmExecutor) and (
            self.layout.is_cloud_project or self.layout.is_cloud_output
        ):
            raise ConfigValidationError(
                "Slurm execution currently requires both project and output roots "
                "to be local paths."
            )
        dispatch = resolved_executor.dispatch

        foot_configs = dict(self.config.footprints)

        sim_ids = self.register_pending()
        if not sim_ids:
            logger.info("run: no receptors configured — nothing to do")
            return LocalHandle()

        # Completion is decided by the outputs themselves (by key):
        # a sim is pending unless every artifact it must produce already exists.
        # When error params are set, that set includes the error trajectory, so
        # sims that pre-date error mode are correctly re-dispatched to backfill it.
        if resolved_skip:
            expected = expected_for_config(self.config)
            pending = [
                sim_id
                for sim_id in sim_ids
                if not is_complete(sim_id, expected, self.storage)
            ]
        else:
            pending = list(sim_ids)
        if not pending:
            logger.info("run: all simulations already complete — nothing to do")
            return LocalHandle()

        names = list(foot_configs.keys())
        logger.info(
            "run(%s): starting %s workers for %d simulations",
            ", ".join(names) if names else "trajectories",
            dispatch,
            len(pending),
        )

        handle = resolved_executor.start(
            pending,
            project=self.layout.output_root,
            output_dir=str(self.layout.output_dir),
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
