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

import pandas as pd

from stilt.artifacts import (
    ArtifactStore,
    FsspecArtifactStore,
    error_trajectory_path,
    footprint_path,
    is_cloud_project,
    project_config_key,
    project_config_path,
    project_receptors_key,
    project_receptors_path,
    project_slug,
    resolve_directory,
    simulation_artifact_key,
    simulation_dir_path,
    simulation_index_path,
    trajectory_path,
)
from stilt.config import FootprintConfig, ModelConfig, STILTParams, _config_or_kwargs
from stilt.executors import (
    Executor,
    JobHandle,
    LocalHandle,
    _sigterm_as_interrupt,
    get_executor,
)
from stilt.footprint import Footprint
from stilt.meteorology import MetArchive, MetStream
from stilt.receptor import Receptor, read_receptors
from stilt.repositories import (
    ArtifactSummary,
    PostgreSQLRepository,
    SimulationRepository,
    SQLiteRepository,
)
from stilt.runtime import RuntimeSettings, resolve_runtime_settings
from stilt.simulation import SimID, Simulation
from stilt.trajectory import Trajectories
from stilt.workers import SimulationTask

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.visualization import ModelPlotAccessor


class Model:
    """STILT project interface for managing simulations."""

    def __init__(
        self,
        project: str | Path | None = None,
        receptors: Receptor | Iterable | str | Path | None = None,
        repository: SimulationRepository | None = None,
        config: ModelConfig | None = None,
        output_dir: str | Path | None = None,
        compute_root: str | Path | None = None,
        artifact_store: ArtifactStore | None = None,
        runtime: RuntimeSettings | None = None,
        **kwargs,
    ):
        project_str = str(project or "")
        output_str = str(output_dir) if output_dir is not None else project_str
        if not project_str and output_dir is not None:
            project_str = output_str
        project_dir: str | Path | None = None
        output_local_dir: str | Path | None = None
        self.runtime = resolve_runtime_settings(runtime)
        self._is_cloud_project = is_cloud_project(project_str)
        self._is_cloud_output = is_cloud_project(output_str)

        if not self._is_cloud_project:
            project_dir = project or (
                output_dir
                if output_dir is not None and not self._is_cloud_output
                else None
            )
        if not self._is_cloud_output:
            output_local_dir = output_dir or project or None

        if self._is_cloud_output and repository is None:
            db_url = self.runtime.db_url
            if not db_url:
                raise ValueError(
                    "PYSTILT_DB_URL env var or RuntimeSettings.db_url is required "
                    "for cloud output projects"
                )
            repository = PostgreSQLRepository(db_url)

        self.directory = resolve_directory(project_dir)
        self.output_dir = resolve_directory(output_local_dir)
        self.project_ref = (
            project_str if self._is_cloud_project else str(self.directory)
        )
        self.output_ref = output_str if self._is_cloud_output else str(self.output_dir)
        self.project = self._project_name(project_str)
        self.artifact_store: ArtifactStore = artifact_store or FsspecArtifactStore(
            self.output_ref, cache_dir=self.runtime.cache_dir
        )
        self.compute_root = self._resolve_compute_root(compute_root)

        self._config = _config_or_kwargs(config, kwargs, ModelConfig)

        self._receptors: list[Receptor] | None = None
        self._receptors_path: Path | None = None
        if receptors is None:
            pass  # lazy: load from repository in .receptors property
        elif isinstance(receptors, (str, Path)):
            self._receptors_path = Path(receptors)
        elif isinstance(receptors, Receptor):
            self._receptors = [receptors]
        elif isinstance(receptors, Iterable):
            items = list(receptors)
            if items and (
                isinstance(items[0], Receptor)
                or (isinstance(items[0], Iterable) and not isinstance(items[0], str))
            ):
                # collection of specs or Receptors
                self._receptors = [
                    r if isinstance(r, Receptor) else Receptor(*r) for r in items
                ]
            else:
                # single receptor spec: (time, longitude, latitude, altitude)
                self._receptors = [Receptor(*items)]

        self.repository: SimulationRepository = repository or SQLiteRepository(
            self.output_dir
        )
        self.met_archive = MetArchive(self.runtime.met_archive)

        # Lazy caches for expensive properties
        self._simulations: SimulationMapping | None = None
        self._mets: dict[str, MetStream] | None = None
        self._plot: ModelPlotAccessor | None = None

    def _resolve_compute_root(self, compute_root: str | Path | None) -> Path:
        """Return the parent directory under which worker sim dirs are created."""
        if compute_root is not None:
            raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
            return Path(raw).resolve()
        if self.runtime.compute_root is not None:
            return self.runtime.compute_root.expanduser().resolve()
        if not self._is_cloud_output:
            return simulation_index_path(self.output_dir)
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        return Path(tmp_root) / "pystilt"

    def _project_name(self, project_str: str) -> str:
        """Return a human-readable project name for local paths and cloud URIs."""
        if self._is_cloud_project and project_str:
            return project_slug(project_str)
        return self.directory.name

    def _config_bytes(self) -> bytes:
        """Serialize the current in-memory config for storage bootstrap."""
        if self._config is None:
            raise ValueError("No in-memory config available to serialize.")
        fd, tmp_name = tempfile.mkstemp(prefix="pystilt_config_", suffix=".yaml")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            self._config.to_yaml(tmp_path)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _receptors_csv_bytes(self, receptors: list[Receptor]) -> bytes:
        """Serialize in-memory receptors to a CSV compatible with read_receptors()."""
        rows: list[dict[str, object]] = []
        for idx, receptor in enumerate(receptors):
            for lon, lat, altitude in zip(
                receptor.longitudes,
                receptor.latitudes,
                receptor.altitudes,
                strict=False,
            ):
                rows.append(
                    {
                        "group": idx,
                        "time": receptor.time.isoformat(sep=" "),
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "altitude": float(altitude),
                        "altitude_ref": receptor.altitude_ref,
                    }
                )
        return pd.DataFrame(rows).to_csv(index=False).encode()

    def _receptors_source_path(self) -> Path:
        """Resolve a local receptors source path for cloud project bootstrap."""
        if self._receptors_path is None:
            raise ValueError("No receptors path was provided.")
        if self._receptors_path.is_absolute():
            return self._receptors_path
        if self._is_cloud_project:
            return self._receptors_path.resolve()
        return self.directory / self._receptors_path

    def _publish_output_inputs(self, receptors: list[Receptor] | None = None) -> None:
        """Persist config/receptors needed to reconstruct from the durable root.

        Workers reopen the model from ``output_dir`` / durable storage, so any
        in-memory config or receptors supplied at construction time must be
        materialized there before execution starts. This applies even when
        ``project`` and ``output_dir`` are the same local directory.
        """
        if self._config is not None:
            self.artifact_store.write_bytes(project_config_key(), self._config_bytes())
        if self._receptors_path is not None:
            self.artifact_store.write_bytes(
                project_receptors_key(), self._receptors_source_path().read_bytes()
            )
        elif receptors is not None:
            self.artifact_store.write_bytes(
                project_receptors_key(), self._receptors_csv_bytes(receptors)
            )
        elif self._receptors is not None:
            self.artifact_store.write_bytes(
                project_receptors_key(), self._receptors_csv_bytes(self._receptors)
            )

    @property
    def config(self) -> ModelConfig:
        """Project :class:`ModelConfig`, loaded from ``config.yaml`` if not provided at construction.

        Returns
        -------
        ModelConfig
        """
        if self._config is None:
            config_path = project_config_path(self.directory)
            if config_path.exists():
                self._config = ModelConfig.from_yaml(config_path)
                return self._config
            if not self.artifact_store.exists(project_config_key()):
                raise FileNotFoundError(
                    f"No config.yaml found in {self.project_ref} or {self.output_ref}. "
                    "Create one with ModelConfig.to_yaml()."
                )
            config_path = self.artifact_store.local_path(project_config_key())
            self._config = ModelConfig.from_yaml(config_path)
        return self._config

    @property
    def receptors(self) -> list[Receptor]:
        """List of receptors for this project, loaded on first access.

        Sources are checked in order: constructor argument, CSV path, or
        the project repository.

        Returns
        -------
        list[Receptor]
        """
        if self._receptors is not None:
            return self._receptors

        if self._receptors_path is not None:
            self._receptors = read_receptors(self._receptors_source_path())
        elif project_receptors_path(self.directory).exists():
            self._receptors = read_receptors(project_receptors_path(self.directory))
        elif self.artifact_store.exists(project_receptors_key()):
            self._receptors = read_receptors(
                self.artifact_store.local_path(project_receptors_key())
            )
        else:
            sim_ids = self.repository.all_sim_ids()
            self._receptors = [self.repository.get_receptor(sid) for sid in sim_ids]

        return self._receptors

    @property
    def simulations(self) -> "SimulationMapping":
        """Lazy dict-like view of all registered simulations.

        Returns
        -------
        SimulationMapping
        """
        if self._simulations is None:
            params = self._make_params()
            self._simulations = SimulationMapping(
                self.output_dir,
                params,
                self.mets,
                self.repository,
                self.artifact_store,
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
    def mets(self) -> dict[str, MetStream]:
        """Named met streams, with subgrid cache paths resolved under compute_root."""
        if self._mets is None:
            self._mets = {}
            for name, config in self.config.mets.items():
                if config.subgrid_enable is True:
                    config = config.model_copy(
                        update={
                            "subgrid_enable": self.compute_root
                            / "_subgrid_cache"
                            / name
                        }
                    )
                self._mets[name] = MetStream(
                    name,
                    directory=config.directory,
                    file_format=config.file_format,
                    file_tres=config.file_tres,
                    n_min=config.n_min,
                    subgrid_enable=config.subgrid_enable,
                    subgrid_bounds=config.subgrid_bounds,
                    subgrid_buffer=config.subgrid_buffer,
                    subgrid_levels=config.subgrid_levels,
                    archive=self.met_archive,
                )
        return self._mets

    def _resolve_mets(self, mets: str | list[str] | None) -> dict[str, MetStream]:
        """Resolve a met filter to a subset of self.mets (None = all)."""
        all_mets = self.mets
        if mets is None:
            return all_mets
        if isinstance(mets, str):
            mets = [mets]
        missing = [m for m in mets if m not in all_mets]
        if missing:
            raise KeyError(f"Unknown met name(s): {missing}")
        return {m: all_mets[m] for m in mets}

    def _make_params(self) -> STILTParams:
        """Build STILTParams from config (excluding met/footprint/grid sections)."""
        dump = self.config.model_dump(exclude={"footprints", "grids", "mets"})
        return STILTParams(**dump)

    @staticmethod
    def _footprint_output_names(
        foot_configs: dict[str, FootprintConfig],
    ) -> list[str]:
        """Return all requested footprint artifact names, including error outputs."""
        names: list[str] = []
        for name, cfg in foot_configs.items():
            names.append(name)
            if cfg.error:
                names.append(f"{name}_error")
        return names

    # -- Queries --------------------------------------------------------------

    def _filter_simulation_ids(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return simulation IDs matching the given filters."""
        df = self.repository.to_dataframe()
        if df.empty:
            return []

        if mets is not None:
            resolved = self._resolve_mets(mets)
            df = df.loc[df.index.map(lambda s: SimID(s).met).isin(resolved.keys())]

        if time_range:
            times = pd.to_datetime(df.index.map(lambda s: SimID(s).time))
            mask = (times >= time_range[0]) & (times <= time_range[1])
            df = df.loc[mask]

        if location_ids is not None:
            loc = df.index.map(lambda s: SimID(s).location_id)
            df = df.loc[loc.isin(location_ids)]

        sim_ids = df.index.tolist()
        if footprint is not None:
            sim_ids = [
                sid
                for sid in sim_ids
                if self.repository.footprint_completed(sid, footprint)
            ]
        return sim_ids

    def _resolve_artifact_path(self, sim_id: str, artifact_path: Path) -> Path | None:
        """Return a local path to a durable artifact when it exists."""
        if artifact_path.exists():
            return artifact_path
        key = simulation_artifact_key(sim_id, artifact_path.name)
        if self.artifact_store.exists(key):
            return self.artifact_store.local_path(key)
        return None

    def get_simulation_ids(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return simulation IDs matching the given filters.

        Parameters
        ----------
        mets : str, list of str, or None
            Restrict to specific met configuration keys. ``None`` returns all.
        footprint : str or None
            If provided, restrict to simulations that have a completed footprint
            with this name.
        time_range : tuple or None
            ``(start, end)`` timestamps to filter receptor times.
        location_ids : set of str or None
            Restrict to specific location IDs.

        Returns
        -------
        list[str]
            Canonical simulation identifiers.
        """
        return self._filter_simulation_ids(
            mets=mets,
            footprint=footprint,
            time_range=time_range,
            location_ids=location_ids,
        )

    def get_simulations(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Simulation]:
        """Return simulation handles matching the given filters.

        Parameters
        ----------
        mets : str, list of str, or None
            Restrict to specific met configuration keys. ``None`` returns all.
        footprint : str or None
            If provided, restrict to simulations whose footprint task reached
            terminal success for this name.
        time_range : tuple or None
            ``(start, end)`` timestamps to filter receptor times.
        location_ids : set of str or None
            Restrict to specific location IDs.

        Returns
        -------
        list[Simulation]
            Matching simulation handles.
        """
        return [
            self.simulations[sid]
            for sid in self.get_simulation_ids(
                mets=mets,
                footprint=footprint,
                time_range=time_range,
                location_ids=location_ids,
            )
        ]

    def get_trajectory_paths(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Path]:
        """Return local-accessible paths for matching trajectory artifacts."""
        paths: list[Path] = []
        for sim_id in self.get_simulation_ids(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        ):
            if self.repository.traj_status(sim_id) != "complete":
                continue
            sim_dir = simulation_dir_path(self.output_dir, sim_id)
            artifact_path = (
                error_trajectory_path(sim_dir, sim_id)
                if error
                else trajectory_path(sim_dir, sim_id)
            )
            resolved = self._resolve_artifact_path(sim_id, artifact_path)
            if resolved is not None:
                paths.append(resolved)
        return paths

    def get_trajectories(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Trajectories]:
        """Load trajectories across matching simulations."""
        trajectories: list[Trajectories] = []
        for path in self.get_trajectory_paths(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
            error=error,
        ):
            trajectories.append(Trajectories.from_parquet(path))
        return trajectories

    def get_footprint_paths(
        self,
        name: str,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Path]:
        """Return local-accessible paths for matching footprint artifacts."""
        paths: list[Path] = []
        for sim_id in self.get_simulation_ids(
            mets=mets,
            footprint=name,
            time_range=time_range,
            location_ids=location_ids,
        ):
            if self.repository.footprint_status(sim_id, name) != "complete":
                continue
            sim_dir = simulation_dir_path(self.output_dir, sim_id)
            artifact_path = footprint_path(sim_dir, name, sim_id=sim_id)
            resolved = self._resolve_artifact_path(sim_id, artifact_path)
            if resolved is not None:
                paths.append(resolved)
        return paths

    def get_footprints(
        self,
        name: str,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Footprint]:
        """Load footprints by name, with optional filters.

        Parameters
        ----------
        name : str
            Footprint name (passed to ``generate_footprints`` as ``name``).
        mets : str, list[str], or None
            Restrict to specific met configurations. None = all.
        time_range : tuple or None
            ``(start, end)`` timestamps to filter receptor times.
        location_ids : set[str] or None
            Restrict to specific location IDs.

        Returns
        -------
        list[Footprint]
            Loaded footprint objects, one per matching simulation.
        """
        footprints: list[Footprint] = []
        for path in self.get_footprint_paths(
            name,
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        ):
            try:
                footprints.append(Footprint.from_netcdf(path))
            except FileNotFoundError:
                logger.debug("Skipping missing footprint file: %s", path)
        return footprints

    def get_pending(
        self,
        mets: str | list[str] | None = None,
        include_failed: bool = False,
    ) -> list[str]:
        """Return sim_ids that haven't been run yet.

        Parameters
        ----------
        mets : str, list[str], or None
            Restrict to a subset of met configurations. None = all.
        include_failed : bool
            If True, count failed runs as pending (only completed are excluded).

        Returns
        -------
        list[str]
            Simulation IDs without a registered (or completed) trajectory.
        """
        if include_failed:
            existing = set(self.repository.completed_trajectories())
        else:
            existing = set(self.repository.all_sim_ids())
        resolved_mets = self._resolve_mets(mets)
        return [
            str(SimID.from_parts(met_name, r))
            for met_name in resolved_mets
            for r in self.receptors
            if str(SimID.from_parts(met_name, r)) not in existing
        ]

    # -- Execution ------------------------------------------------------------

    def submit(
        self,
        receptors: list[Receptor] | None = None,
        batch_id: str | None = None,
    ) -> list[str]:
        """Register receptors as pending simulations.

        Idempotent — already-registered simulations are left unchanged.
        Returns the list of sim_ids that were submitted.

        Parameters
        ----------
        receptors : list[Receptor] or None, optional
            Receptors to register.  Defaults to ``self.receptors``.
        batch_id : str or None, optional
            Label this group of simulations for progress tracking via
            :meth:`~stilt.repositories.SimulationRepository.batch_progress`.
            A timestamp-based name is used when ``None``.

        Returns
        -------
        list[str]
            Registered simulation IDs (met × receptor).
        """
        recs = receptors if receptors is not None else self.receptors
        self._publish_output_inputs(receptors=recs)
        pairs = [
            (str(SimID.from_parts(met_name, r)), r)
            for met_name in self.mets
            for r in recs
        ]
        self.repository.register_many(
            pairs,
            batch_id=batch_id,
            footprint_names=self._footprint_output_names(dict(self.config.footprints)),
        )
        return [sid for sid, _ in pairs]

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool | None = None,
        wait: bool = True,
        batch_id: str | None = None,
    ) -> JobHandle:
        """Register pending work and start workers to drain it.

        When ``config.footprints`` contains one or more footprint
        configurations, workers auto-run HYSPLIT trajectories as needed and
        then compute footprints in a single pass.  When no footprint configs
        are defined, only HYSPLIT trajectories are dispatched.

        Workers pull work from the project repository via
        :func:`~stilt.workers.worker_loop`; the coordinator registers
        simulations and starts workers, then optionally blocks.

        Parameters
        ----------
        executor : Executor, optional
            Override the executor resolved from ``config.execution``.
        skip_existing : bool or None, optional
            Skip simulations that already have output.  ``None`` (default)
            reads the value from ``config.skip_existing``.
        wait : bool, optional
            If ``True`` (default), block until all workers finish and sync the
            repository.  If ``False``, return the :class:`JobHandle`
            immediately — suitable for fire-and-forget Slurm runs.
        batch_id : str or None, optional
            Forwarded to :meth:`submit` for batch progress tracking.

        Returns
        -------
        JobHandle
            A handle you can call ``.wait()`` on later, or ignore.
        """
        self._simulations = None
        resolved_skip = (
            skip_existing if skip_existing is not None else self.config.skip_existing
        )
        foot_configs = dict(self.config.footprints)
        requested_footprints = self._footprint_output_names(foot_configs)

        # 1. Register all (met × receptor) combinations.
        all_sim_ids = self.submit(batch_id=batch_id)

        if not all_sim_ids:
            logger.info("run: no receptors configured — nothing to do")
            return LocalHandle()

        # 2. Adjust pending status based on skip_existing.
        if not resolved_skip:
            # Force-rerun: reset all sims to pending regardless of prior status.
            if foot_configs:
                self.repository.clear_footprints(
                    all_sim_ids, names=requested_footprints
                )
            for sim_id in all_sim_ids:
                self.repository.record_artifacts(sim_id, ArtifactSummary())
            self.repository.reset_to_pending(all_sim_ids)
        elif foot_configs:
            # Footprint mode with skip_existing: re-queue sims whose trajectory
            # is done but at least one footprint is still missing.
            sims_needing_foot = [
                sid
                for sid in all_sim_ids
                if any(
                    not self.repository.footprint_completed(sid, name)
                    for name in requested_footprints
                )
            ]
            if sims_needing_foot:
                self.repository.reset_to_pending(sims_needing_foot)

        # 3. Early exit when nothing is pending.
        if not self.repository.pending_trajectories():
            logger.info("run: all simulations already complete — nothing to do")
            return LocalHandle()

        # 4. Start workers.
        execution = self.config.execution or {}
        n_workers = execution.get("n_workers", 1)
        exe = executor or get_executor(execution)

        names = list(foot_configs.keys())
        logger.info(
            "run(%s): starting workers for %d simulations",
            ", ".join(names) if names else "trajectories",
            len(self.repository.pending_trajectories()),
        )

        handle = exe.start(
            self.output_ref,
            n_workers=n_workers,
            follow=False,
            output_dir=None,
            compute_root=str(self.compute_root),
        )

        if wait:
            try:
                with _sigterm_as_interrupt():
                    handle.wait()
            except KeyboardInterrupt:
                logger.warning("run interrupted")
                raise

        return handle

    def _build_run_args(self, sim_id: str) -> SimulationTask | None:
        """Build a :class:`~stilt.workers.SimulationTask` for a single simulation.

        Returns ``None`` when ``skip_existing`` is satisfied (trajectory or all
        requested footprints already complete).

        Called only by ``stilt worker``, which processes a single sim_id at a
        time and can afford one DB lookup per call.  :meth:`run` builds
        :class:`~stilt.workers.SimulationTask` inline from ``self.receptors`` to
        avoid N per-simulation DB queries.

        Parameters
        ----------
        sim_id : str
            Canonical simulation identifier.

        Returns
        -------
        SimulationTask or None
        """
        sid = SimID(sim_id)
        met_source = self.mets[sid.met]
        receptor = self.repository.get_receptor(sim_id)

        resolved_skip = self.config.skip_existing
        all_foot_configs = dict(self.config.footprints)

        if all_foot_configs:
            if resolved_skip:
                foot_configs: dict | None = {
                    name: cfg
                    for name, cfg in all_foot_configs.items()
                    if any(
                        not self.repository.footprint_completed(sim_id, output_name)
                        for output_name in self._footprint_output_names({name: cfg})
                    )
                } or None
                if foot_configs is None:
                    return None  # all footprints for this sim are done
            else:
                foot_configs = all_foot_configs
        else:
            if resolved_skip and sim_id in set(
                self.repository.completed_trajectories()
            ):
                return None
            foot_configs = None

        return SimulationTask(
            compute_root=self.compute_root,
            sim_id=sid,
            meteorology=met_source,
            receptor=receptor,
            params=self._make_params(),
            foot_configs=foot_configs,
            artifact_store=self.artifact_store,
            repository=self.repository,
        )


class SimulationMapping:
    """Persistent lazy dict: sim_id → Simulation.

    Simulation objects are cheap immutable handles so the cache never needs
    invalidation - new sims can be added any time without clearing.
    """

    def __init__(
        self,
        project_dir: Path,
        params: STILTParams,
        mets: dict[str, MetStream],
        repository: SimulationRepository,
        artifact_store: ArtifactStore,
    ):
        self._project_dir = project_dir
        self._params = params
        self._mets = mets
        self._repository = repository
        self._artifact_store = artifact_store
        self._cache: dict[str, Simulation] = {}

    def __getitem__(self, sim_id: str) -> Simulation:
        if sim_id not in self._cache:
            self._cache[sim_id] = self._build(SimID(sim_id))
        return self._cache[sim_id]

    def __contains__(self, sim_id: str) -> bool:
        return self._repository.has(sim_id)

    def __iter__(self):
        return iter(self._repository.all_sim_ids())

    def __len__(self) -> int:
        return self._repository.count()

    def items(self):
        """Yield ``(sim_id, Simulation)`` pairs for all registered simulations."""
        return ((sid, self[sid]) for sid in self)

    def values(self):
        """Yield :class:`Simulation` objects for all registered simulations."""
        return (self[sid] for sid in self)

    def _build(self, sim_id: SimID) -> Simulation:
        receptor = self._repository.get_receptor(sim_id)
        sim_dir = self._project_dir / "simulations" / "by-id" / sim_id
        return Simulation(
            directory=sim_dir,
            receptor=receptor,
            params=self._params,
            meteorology=self._mets[sim_id.met],
            artifact_store=self._artifact_store,
        )
