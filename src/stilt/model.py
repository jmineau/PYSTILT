"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the STILT-R model framework.
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
    OutputCollection,
    ReceptorCollection,
    SimulationCollection,
)
from stilt.config import (
    MetConfig,
    ModelConfig,
    RuntimeSettings,
    STILTParams,
    VariantConfig,
)
from stilt.config.model import _config_or_kwargs
from stilt.errors import ConfigChangedError, ConfigValidationError
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

#: Met fields that change no output: where the files are, not what they hold.
_UNRECORDED_MET_FIELDS = frozenset({"directory", "subgrid_dir"})


def _met_differences(met: MetConfig, recorded: dict) -> list[str]:
    """Names of the result-affecting met fields on which *met* differs from the record."""
    mine = met.model_dump(mode="json")
    return sorted(
        k
        for k in mine
        if k not in _UNRECORDED_MET_FIELDS and mine[k] != recorded.get(k)
    )


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
    simulations a model defines are its receptors crossed with its variants;
    whether each is complete is read from the outputs by key.

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
    variants : dict[str, VariantConfig]
    mets : dict[str, MetStream]
    simulations : SimulationCollection
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
        # A config given here is the user's latest word and is written to the
        # project; one loaded from the project is never rewritten.
        self._config_given = self._config is not None

        self._mets: dict[str, MetStream] | None = None
        self._variants: dict[str, VariantConfig] | None = None
        self._receptors: ReceptorCollection | None = None
        self._receptors_input = receptors
        self._queue: PostgresQueue | None = None
        self._simulations: SimulationCollection | None = None
        self._handles: dict[SimID, Simulation] = {}
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
        """The default transport parameters (from config)."""
        return self.config.to_stilt_params()

    @property
    def variants(self) -> dict[str, VariantConfig]:
        """Resolved variants by simulation-level name, in config order."""
        if self._variants is None:
            self._variants = self.config.resolve_variants()
        return self._variants

    @property
    def mets(self) -> dict[str, MetStream]:
        """Named met streams, resolved from config."""
        if self._mets is None:
            self._mets = {
                name: MetStream.from_config(name, cfg)
                for name, cfg in self.config.mets.items()
            }
        return self._mets

    @property
    def queue(self) -> PostgresQueue | None:
        """Postgres work queue, present only when ``PYSTILT_DB_URL`` is configured."""
        if self._queue is None and self.runtime.db_url:
            self._queue = resolve_queue(self.runtime)
        return self._queue

    def check_config(self) -> None:
        """
        Refuse to change the settings of a variant that has already run.

        The project's record (:meth:`stilt.Project.load_record`) holds the
        resolved settings every registered variant ran with. If a variant now
        resolves differently, its outputs would no longer match their name,
        so this raises :class:`~stilt.errors.ConfigChangedError` naming the
        fields. Declare a new variant for new settings, or :meth:`remove` the
        old one to rerun it.
        """
        record = self.project.load_record()
        changed = {}
        for name, variant in self.variants.items():
            if name in record["variants"]:
                diff = variant.differences(record["variants"][name])
                if diff:
                    changed[name] = diff
        for name, met in self.config.mets.items():
            if name in record["mets"]:
                diff = _met_differences(met, record["mets"][name])
                if diff:
                    changed[f"met {name}"] = diff
        if changed:
            detail = "; ".join(f"{n}: {', '.join(f)}" for n, f in changed.items())
            raise ConfigChangedError(
                f"These settings already ran under their name in {self.project.root} "
                f"({detail}). Declare a new variant for the new settings, or remove "
                "the old outputs first (Model.remove / stilt rm --variant)."
            )

    def orphans(self) -> list[str]:
        """Variants in the project's record that ``config.yaml`` no longer declares."""
        return [
            name
            for name in self.project.load_record()["variants"]
            if name not in self.variants
        ]

    def register(self, receptors: Iterable[Receptor] | None = None) -> list[str]:
        """
        Persist the model's inputs to the project and return its receptor ids.

        Makes sure the project holds ``config.yaml`` and ``receptors.csv``, so
        that workers (local processes, Slurm tasks, Kubernetes pods) can
        rebuild this model from the root alone, and records the resolved
        settings of every variant. A ``config.yaml`` loaded from the project
        is never rewritten; one given to :class:`Model` in Python is written
        out (without defaults). Receptors not yet in ``receptors.csv`` are
        appended to it in the file's own columns. When a work queue is
        configured, the receptors are enqueued as pending work.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to add to the project. When omitted, the model's own
            receptors are added (the source CSV is copied byte for byte into
            a project that has none).

        Returns
        -------
        list[str]
            Receptor ids registered by this call.

        Raises
        ------
        ConfigChangedError
            If a variant that already ran now resolves differently
            (:meth:`check_config`).
        """
        self.check_config()
        if self._config_given or not self.project.has_config:
            self.project.save_config(self.config)

        if receptors is None and self._receptors_input is None:
            batch = list(self.receptors)  # the project's own file; nothing to add
        else:
            batch = list(self.receptors) if receptors is None else list(receptors)
            source = self.receptors.source_path if receptors is None else None
            if self.project.add_receptors(batch, source=source):
                # The registered set changed: rebuild receptors from the project.
                self._receptors_input = None
                self._receptors = None
                self._simulations = None
                self._handles = {}

        record = self.project.load_record()
        record["mets"].update(
            {
                name: met.model_dump(mode="json")
                for name, met in self.config.mets.items()
            }
        )
        record["variants"].update(
            {name: v.record() for name, v in self.variants.items()}
        )
        self.project.save_record(record)

        receptor_ids = [str(r.id) for r in batch]
        if self.queue is not None:
            self.queue.register(receptor_ids)
        return receptor_ids

    def remove(self, variant: str) -> list[SimID]:
        """
        Delete every simulation of *variant* and drop it from the record.

        *variant* may be a simulation-level name, a realization group (every
        realization goes), or a name that ``config.yaml`` no longer declares
        (:meth:`orphans`). Variants derived from it with ``from:`` go with it,
        since their footprints came from its particles. Afterwards the variant
        is new again and reruns on the next :meth:`run`.

        Returns
        -------
        list[SimID]
            The simulations that were deleted.
        """
        record = self.project.load_record()
        recorded = record["variants"]
        names = [
            n for n, v in recorded.items() if n == variant or v.get("group") == variant
        ]
        if not names:
            raise KeyError(
                f"No variant {variant!r} in the record of {self.project.root}"
            )
        names += [
            n
            for n, v in recorded.items()
            if v.get("derived_from") in names and n not in names
        ]
        deleted = []
        for receptor in self.receptors:
            for name in names:
                sid = SimID(receptor.id, name)
                self._handle(sid, VariantConfig.model_validate(recorded[name])).delete()
                self._handles.pop(sid, None)
                deleted.append(sid)
        for name in names:
            del recorded[name]
        self.project.save_record(record)
        self._simulations = None
        return deleted

    # -- Simulations -----------------------------------------------------------

    def simulation(self, key: str | SimID | tuple[str, str]) -> Simulation:
        """Build (and cache) the handle for one simulation id; no side effects on disk."""
        sid = SimID.parse(key)
        if sid not in self._handles:
            self._handles[sid] = self._handle(sid, self.variants[sid.variant])
        return self._handles[sid]

    def _handle(self, sid: SimID, variant: VariantConfig) -> Simulation:
        """Build the handle for *sid* under *variant* (declared or recorded)."""
        parent = (
            self.simulation((sid.receptor, variant.derived_from))
            if variant.derived_from is not None
            else None
        )
        return Simulation(
            receptor=self.receptors[sid.receptor],
            meteorology=None if parent is not None else self.mets[variant.met],
            params=variant.stilt_params(),
            footprint=variant.footprint,
            variant=sid.variant,
            parent=parent,
            directory=self.compute_root / sid,
            store=self.project.store,
        )

    @property
    def simulations(self) -> SimulationCollection:
        """Simulations this model defines (receptors × variants)."""
        if self._simulations is None:
            self._simulations = SimulationCollection(self)
        return self._simulations

    @property
    def trajectories(self) -> OutputCollection:
        """Every simulation's trajectories (``simulations.trajectories``)."""
        return self.simulations.trajectories

    @property
    def footprint(self) -> OutputCollection:
        """Every simulation's footprint (``simulations.footprint``)."""
        return self.simulations.footprint

    @property
    def plot(self) -> ModelPlotAccessor:
        """Plotting namespace (e.g. ``model.plot.availability()``)."""
        if self._plot is None:
            from stilt.visualization import ModelPlotAccessor

            self._plot = ModelPlotAccessor(self)
        return self._plot

    def status(self) -> StatusCounts:
        """Return completion counts (total / completed / pending), read from the outputs."""
        total = len(self.simulations)
        if not total:
            return StatusCounts()
        pending = len(self.simulations.incomplete())
        return StatusCounts(total=total, completed=total - pending, pending=pending)

    # -- Execution -------------------------------------------------------------

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool = True,
        wait: bool = True,
    ) -> JobHandle:
        """
        Persist inputs, then start workers for every receptor with incomplete work.

        Workers run each of the receptor's simulations in turn: HYSPLIT where a
        trajectory is missing, then the footprint where one is configured.

        Parameters
        ----------
        executor : Executor, optional
            Override the executor resolved from ``config.execution``.
        skip_existing : bool, optional
            Skip simulations whose outputs are all present (default). ``False``
            reruns every simulation.
        wait : bool, optional
            Block until workers finish (default). ``False`` returns the
            :class:`JobHandle` immediately — fire-and-forget for Slurm.

        Returns
        -------
        JobHandle
        """
        self._simulations = None
        resolved_skip = skip_existing
        resolved_executor = executor or get_executor(self.config.execution or {})
        if isinstance(resolved_executor, SlurmExecutor) and self.project.is_cloud:
            raise ConfigValidationError(
                "Slurm execution currently requires a local project root."
            )

        receptor_ids = self.register()
        if not receptor_ids:
            logger.info("run: no receptors configured — nothing to do")
            return LocalHandle()

        pending = (
            self.simulations.incomplete().receptors if resolved_skip else receptor_ids
        )
        if not pending:
            logger.info("run: all simulations already complete — nothing to do")
            return LocalHandle()

        logger.info(
            "run(%s): starting %s workers for %d receptors",
            ", ".join(self.variants),
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
