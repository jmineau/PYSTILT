"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the R-STILT model framework.
"""

import logging
import os
import tempfile
from collections.abc import Iterable
from functools import wraps
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
from stilt.index import IndexCounts, SimulationIndex
from stilt.index.factory import resolve_index
from stilt.meteorology import MetSource
from stilt.receptor import Receptor
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


def _wrap_wait_with_rebuild(handle: JobHandle, index: SimulationIndex) -> JobHandle:
    """Wrap one handle's ``wait()`` so push completion rebuilds durable state once."""
    original_wait = handle.wait
    completed = False

    @wraps(original_wait)
    def wait() -> None:
        nonlocal completed
        if completed:
            return
        original_wait()
        index.rebuild()
        completed = True

    handle.wait = wait
    return handle


class Model:
    """Science-facing STILT project interface.

    ``Model`` is the primary Python entry point for configuring a STILT project,
    running one-off simulations, and loading simulation results. It also owns
    the durable project index used by batch, HPC, and cloud execution.

    Pull-mode workers and streaming consumers operate against the model's
    durable index directly when the configured index supports claims.
    Claim-worker control is exposed primarily through the CLI and
    :func:`stilt.execution.pull_simulations` for advanced Python use.

    Parameters
    ----------
    project : str or Path or None, optional
        Local project root or cloud URI used to identify the model.
    receptors : Receptor or iterable or str or Path or None, optional
        In-memory receptors or a path to a receptor CSV.
    config : ModelConfig or None, optional
        In-memory project config. When omitted, config is loaded lazily from storage.
    output_dir : str or Path or None, optional
        Durable output root. Defaults to ``project``.
    compute_root : str or Path or None, optional
        Local parent directory where worker simulation directories are created.
    runtime : RuntimeSettings or None, optional
        Runtime-only deployment settings such as cache roots and DB URLs.

    Attributes
    ----------
    config : ModelConfig
        Lazily loaded project configuration.
    receptors : ReceptorCollection
        Science-facing receptor accessor.
    simulations : SimulationCollection
        Registered simulation handles backed by the durable index.
    mets : dict[str, MetSource]
        Named meteorology streams.
    trajectories : TrajectoryCollection
        Cross-simulation trajectory accessor.
    footprints : FootprintCollection
        Cross-simulation footprint accessor namespace.
    plot : ModelPlotAccessor
        Plotting namespace for model summaries and outputs.
    index : SimulationIndex
        Durable simulation registry used by the coordinator and output queries.
    storage : Storage
        Durable bootstrap and output storage facade.
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
        # needed to resolve the durable index backend, so resolve them first.
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
        self._mets: dict[str, MetSource] | None = None
        self._receptors = receptors  # may be None, a Receptor, or an iterable of Receptors; resolved lazily in the accessor
        self._index: SimulationIndex | None = None
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
    def index(self) -> SimulationIndex:
        """Durable simulation registry for this model."""
        if self._index is None:
            self._index = resolve_index(
                None,
                output_root=self.layout.output_root,
                runtime=self.runtime,
                builtin_backend="postgres" if self.layout.is_cloud_output else "sqlite",
            )
        return self._index

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
        """Project :class:`ModelConfig`, loaded from ``config.yaml`` if not provided at construction.

        Returns
        -------
        ModelConfig
        """
        if self._config is None:
            self._config = self.storage.load_config()
        return self._config

    @property
    def receptors(self) -> ReceptorCollection:
        """Sequence-like receptor accessor for this project.

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
        """Persist model inputs and register one batch of pending work.

        This is the durable registration boundary shared by local runs,
        queue-backed workers, and the CLI ``stilt register`` command.

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
        self.index.register(
            list(pairs),
            footprint_names=foot_names(dict(self.config.footprints)),
            scene_id=scene_id,
        )

        # Registration updates the durable receptor/index truth, so drop
        # cached accessors — next access rebuilds from that durable surface.
        self._receptors = None
        self._simulations = None

        return [sim_id for sim_id, _ in pairs]

    @property
    def simulations(self) -> SimulationCollection:
        """Lazy simulation collection for registered simulations.

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
                self.index,
                self.storage.store,
            )
        return self._simulations

    @property
    def plot(self) -> "ModelPlotAccessor":
        """Plotting namespace for this model (e.g. ``model.plot.availability()``).

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
    def mets(self) -> dict[str, MetSource]:
        """Named met streams, with subgrid cache paths resolved under compute_root."""
        if self._mets is None:
            self._mets = {}
            for name, config in self.config.mets.items():
                self._mets[name] = MetSource(
                    name,
                    directory=config.directory,
                    file_format=config.file_format,
                    file_tres=config.file_tres,
                    n_min=config.n_min,
                    subgrid_enable=config.subgrid_enable,
                    subgrid_bounds=config.subgrid_bounds,
                    subgrid_buffer=config.subgrid_buffer,
                    subgrid_levels=config.subgrid_levels,
                )
        return self._mets

    # -- Queries --------------------------------------------------------------

    def status(self, scene_id: str | None = None) -> IndexCounts:
        """Return cheap aggregate counts for the current project registry."""
        return self.index.counts(scene_id=scene_id)

    def scene_counts(self) -> dict[str, IndexCounts]:
        """Return grouped aggregate counts for each registered scene."""
        return self.index.scene_counts()

    # -- Execution ------------------------------------------------------------

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool | None = None,
        rebuild: bool | None = None,
        wait: bool = True,
    ) -> JobHandle:
        """Register pending work and start workers to drain it.

        When ``config.footprints`` contains one or more footprint
        configurations, workers auto-run HYSPLIT trajectories as needed and
        then compute footprints in a single pass.  When no footprint configs
        are defined, only HYSPLIT trajectories are dispatched.

        Workers either drain a claim-capable project index via
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
            Rebuild the durable index from outputs before planning work.
            ``None`` (default) uses auto mode: rebuild when skip-existing
            is enabled, otherwise skip the pre-run rebuild.
        wait : bool, optional
            If ``True`` (default), block until all workers finish and rebuild
            durable index from outputs. If ``False``, return the :class:`JobHandle`
            immediately — suitable for fire-and-forget Slurm runs.

        Returns
        -------
        JobHandle
            A handle you can call ``.wait()`` on later, or ignore.

        Notes
        -----
        ``Model.run()`` is the main science-facing execution path for local or
        notebook use. For claim-based service workflows, use
        :func:`stilt.execution.pull_simulations` against a model configured with
        a PostgreSQL-backed index, or use the CLI ``stilt pull-worker`` and
        ``stilt serve`` commands.
        """
        index = self.index

        # Execution mutates durable index state, so any cached simulation view must
        # be rebuilt after each coordinator run.
        self._simulations = None
        resolved_skip = (
            skip_existing if skip_existing is not None else self.config.skip_existing
        )
        resolved_rebuild = resolved_skip if rebuild is None else rebuild
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

        if resolved_rebuild:
            logger.info("run: rebuilding durable index before planning")
            index.rebuild()

        index.reset_to_pending(sim_ids, clear_outputs=not resolved_skip)
        pending = index.pending_trajectories()
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
        if dispatch == "push":
            handle = _wrap_wait_with_rebuild(handle, index)

        if wait:
            logger.info("run: waiting for workers to finish...")
            try:
                with sigterm_as_interrupt():
                    handle.wait()
            except KeyboardInterrupt:
                logger.warning("run interrupted")
                raise

        return handle
