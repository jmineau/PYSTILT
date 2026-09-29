"""The :class:`Model`, which sets up, runs, and loads a STILT project."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from stilt.collections import ReceptorCollection, SimulationCollection
from stilt.config import (
    MetConfig,
    ModelConfig,
    RuntimeSettings,
    VariantConfig,
)
from stilt.config.meteorology import UNRECORDED_MET_FIELDS
from stilt.config.variant import UNRECORDED_FIELDS
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


def _changed_fields(
    current: dict[str, Any], recorded: dict[str, Any], ignore: frozenset[str]
) -> list[str]:
    """Return the fields of ``current`` that differ from ``recorded``, skipping ``ignore``."""
    return sorted(
        k for k in current if k not in ignore and current[k] != recorded.get(k)
    )


class Model:
    """
    A STILT project: receptors, settings, and the simulations they define.

    A model runs every receptor once per variant and loads the resulting
    trajectories and footprints. Its inputs and outputs live in a project
    directory or object-store URI. Settings and receptors given here are
    saved to the project when the model runs, so ``Model(project)`` opens it
    again later.

    Parameters
    ----------
    project : str or Path, optional
        Project directory or object-store URI. A temporary directory is used
        when omitted.
    receptors : Receptor, iterable of Receptor, str or Path, optional
        Receptors to run, or the path of a receptors CSV (relative to the
        project directory). Defaults to the project's ``receptors.csv``.
    config : ModelConfig, optional
        Model settings. Defaults to the project's ``config.yaml``.
    compute_root : str or Path, optional
        Directory under which simulations run. Defaults to
        ``PYSTILT_COMPUTE_ROOT``, then to the project's ``simulations/by-id``
        for a local project or a temporary directory for a cloud project.
    runtime : RuntimeSettings, optional
        Settings for this machine (download cache, work-queue URL, and
        compute root). Read from ``PYSTILT_*`` environment variables when
        omitted.
    **kwargs
        Settings for :class:`~stilt.ModelConfig`, such as ``n_hours``,
        ``numpar``, ``mets``, and ``grid``. Cannot be combined with *config*.

    Attributes
    ----------
    project : Project
        The project's files.
    config : ModelConfig
        Model settings.
    receptors : ReceptorCollection
        Receptors, by position or by id.
    variants : dict of str to VariantConfig
        Settings of each variant, by name.
    mets : dict of str to MetStream
        Meteorology sources, by name.
    simulations : SimulationCollection
        Every receptor under every variant.
    plot : ModelPlotAccessor
        Plotting methods.
    queue : PostgresQueue or None
        Work queue for ``stilt pull-worker``. ``None`` unless
        ``PYSTILT_DB_URL`` is set.

    Examples
    --------
    Run one receptor for 24 hours back in time and load its footprint:

    >>> import stilt
    >>> receptor = stilt.PointReceptor("2023-07-15 18:00", -111.848, 40.766, 10)
    >>> met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
    >>> grid = stilt.Grid(
    ...     xmin=-113, xmax=-110.5, ymin=40, ymax=42, xres=0.01, yres=0.01
    ... )
    >>> model = stilt.Model(
    ...     project="./my_project",
    ...     receptors=[receptor],
    ...     mets={"hrrr": met},
    ...     n_hours=-24,
    ...     numpar=200,
    ...     grid=grid,
    ... )
    >>> model.run()
    >>> foot = model.simulations[receptor.id, "hrrr"].footprint

    Open the same project later:

    >>> model = stilt.Model("./my_project")
    >>> model.status()
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
        if config is not None and kwargs:
            raise TypeError("Cannot pass both a ModelConfig and keyword settings.")
        self._config = ModelConfig(**kwargs) if kwargs else config
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

    def _resolve_compute_root(self, compute_root: str | Path | None) -> Path:
        """Return the directory under which simulations run."""
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
        """Model settings, from ``config.yaml`` unless given to the constructor."""
        if self._config is None:
            self._config = self.project.load_config()
        return self._config

    @property
    def receptors(self) -> ReceptorCollection:
        """Receptors, by position (``receptors[0]``) or by id (``receptors[receptor_id]``)."""
        if self._receptors is None:
            self._receptors = ReceptorCollection(
                self._receptors_input, project=self.project
            )
        return self._receptors

    @property
    def variants(self) -> dict[str, VariantConfig]:
        """
        Settings of each variant, by name, in config order.

        A realization group appears once per realization (``hrrr-err-0``,
        ``hrrr-err-1``, ...).
        """
        if self._variants is None:
            self._variants = self.config.resolve_variants()
        return self._variants

    @property
    def mets(self) -> dict[str, MetStream]:
        """Meteorology sources declared in the config, by name."""
        if self._mets is None:
            self._mets = {
                name: MetStream(name, cfg) for name, cfg in self.config.mets.items()
            }
        return self._mets

    @property
    def queue(self) -> PostgresQueue | None:
        """Postgres work queue, or ``None`` when ``PYSTILT_DB_URL`` is not set."""
        if self._queue is None and self.runtime.db_url:
            self._queue = resolve_queue(self.runtime)
        return self._queue

    def check_config(self) -> None:
        """
        Check that no registered variant or met has changed its settings.

        The project records the settings every registered variant and met
        ran with (:meth:`stilt.project.Project.load_record`). Declare a new
        variant for new settings, or :meth:`remove` the old one to run it
        again.

        Raises
        ------
        ConfigChangedError
            If a variant or met now has different settings. The message
            names the fields that changed.
        """
        record = self.project.load_record()
        changed = {}
        for name, variant in self.variants.items():
            if name in record["variants"]:
                diff = _changed_fields(
                    variant.record(), record["variants"][name], UNRECORDED_FIELDS
                )
                if diff:
                    changed[name] = diff
        for name, met in self.config.mets.items():
            if name in record["mets"]:
                diff = _changed_fields(
                    met.model_dump(mode="json"),
                    record["mets"][name],
                    UNRECORDED_MET_FIELDS,
                )
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
        """Return the registered variants that ``config.yaml`` no longer declares."""
        return [
            name
            for name in self.project.load_record()["variants"]
            if name not in self.variants
        ]

    def register(self, receptors: Iterable[Receptor] | None = None) -> list[str]:
        """
        Save the model's settings and receptors to the project.

        Workers rebuild the model from the project alone, so :meth:`run`
        calls this first. A config given in Python is written to
        ``config.yaml`` with only the settings that were set. A
        ``config.yaml`` loaded from the project is left as it is. Receptors
        not yet in ``receptors.csv`` are appended to it, and the settings of
        every variant are recorded. When a work queue is configured, the
        receptors are added to it.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to add to the project. Defaults to the model's own
            receptors. When those came from a CSV and the project has no
            ``receptors.csv`` yet, the CSV is copied unchanged.

        Returns
        -------
        list of str
            Ids of the receptors registered, including any the project
            already had.

        Raises
        ------
        ConfigChangedError
            If a variant or met that already ran now has different settings
            (:meth:`check_config`).
        """
        self.check_config()
        orphans = self.orphans()
        if orphans:
            logger.warning(
                "config.yaml in %s no longer declares %s, which have outputs; "
                "they stay until removed (stilt rm --variant)",
                self.project.root,
                ", ".join(orphans),
            )
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
        Delete every simulation of a variant and forget its settings.

        Variants that take their particles from it with ``from:`` are deleted
        too. Afterwards the variant runs again as new on the next
        :meth:`run`, with whatever settings ``config.yaml`` now gives it.

        Parameters
        ----------
        variant : str
            Variant name, realization group (every realization is deleted),
            or a name ``config.yaml`` no longer declares (:meth:`orphans`).

        Returns
        -------
        list of SimID
            The simulations that were deleted.

        Raises
        ------
        KeyError
            If the project has no registered variant or group by that name.
        """
        record = self.project.load_record()
        recorded = {
            n: VariantConfig.model_validate(v) for n, v in record["variants"].items()
        }
        names = [n for n, v in recorded.items() if variant in (n, v.group)]
        if not names:
            raise KeyError(
                f"No variant {variant!r} in the record of {self.project.root}"
            )
        names += [
            n for n, v in recorded.items() if v.derived_from in names and n not in names
        ]
        # Build everything from the record, since config.yaml may no longer
        # declare the variant, its parent, or its met.
        mets = {
            n: MetStream(n, MetConfig.model_validate(m))
            for n, m in record["mets"].items()
        }

        def handle(receptor: Receptor, name: str) -> Simulation:
            config = recorded[name]
            parent = (
                handle(receptor, config.derived_from)
                if config.derived_from is not None
                else None
            )
            return self._handle(SimID(receptor.id, name), config, parent, mets)

        deleted = []
        for receptor in self.receptors:
            for name in names:
                handle(receptor, name).delete()
                self._handles.pop(SimID(receptor.id, name), None)
                deleted.append(SimID(receptor.id, name))
        for name in names:
            del record["variants"][name]
        self.project.save_record(record)
        self._simulations = None
        return deleted

    # -- Simulations -----------------------------------------------------------

    def simulation(self, key: str | SimID | tuple[str, str]) -> Simulation:
        """
        Return one simulation by id.

        Nothing is written to disk. The same object is returned on later
        calls.

        Parameters
        ----------
        key : str, SimID or tuple of (str, str)
            ``"<receptor_id>/<variant>"`` or a ``(receptor_id, variant)`` pair.
        """
        sid = SimID.parse(key)
        if sid not in self._handles:
            variant = self.variants[sid.variant]
            parent = (
                self.simulation((sid.receptor, variant.derived_from))
                if variant.derived_from is not None
                else None
            )
            self._handles[sid] = self._handle(sid, variant, parent, self.mets)
        return self._handles[sid]

    def _handle(
        self,
        sid: SimID,
        variant: VariantConfig,
        parent: Simulation | None,
        mets: dict[str, MetStream],
    ) -> Simulation:
        """Build the simulation *sid* with the given variant settings and met streams."""
        return Simulation(
            self.receptors[sid.receptor],
            variant,
            met=None if parent is not None else mets[variant.met],
            parent=parent,
            directory=self.compute_root / sid,
            store=self.project.store,
        )

    @property
    def simulations(self) -> SimulationCollection:
        """Every receptor under every variant, indexed by ``(receptor_id, variant)``."""
        if self._simulations is None:
            self._simulations = SimulationCollection(self)
        return self._simulations

    @property
    def plot(self) -> ModelPlotAccessor:
        """Plotting methods, such as ``model.plot.availability()``."""
        if self._plot is None:
            from stilt.visualization import ModelPlotAccessor

            self._plot = ModelPlotAccessor(self)
        return self._plot

    def status(self) -> pd.DataFrame:
        """
        Return one row per simulation saying which outputs exist.

        See :meth:`~stilt.collections.SimulationCollection.status`.
        """
        return self.simulations.status()

    # -- Execution -------------------------------------------------------------

    def run(
        self,
        executor: Executor | None = None,
        skip_existing: bool = True,
        wait: bool = True,
    ) -> JobHandle:
        """
        Run every simulation that has not finished.

        Saves the settings and receptors to the project (:meth:`register`),
        then starts workers for each receptor with missing outputs. A worker
        runs HYSPLIT for each of the receptor's variants that lacks a
        trajectory, then calculates the footprint where the variant has a
        grid.

        Parameters
        ----------
        executor : Executor, optional
            Where to run the workers. Defaults to the one set by
            ``config.execution`` (local processes unless configured).
        skip_existing : bool, default True
            Skip simulations whose outputs all exist. ``False`` runs every
            simulation again.
        wait : bool, default True
            Block until the workers finish. With ``False`` the handle is
            returned right away, which suits a Slurm submission.

        Returns
        -------
        JobHandle
            Handle to the started workers.

        Raises
        ------
        ConfigChangedError
            If a variant that already ran now has different settings.
        ConfigValidationError
            If Slurm execution is requested for a cloud project.
        """
        self._simulations = None
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
            self.simulations.incomplete().receptors if skip_existing else receptor_ids
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
            skip_existing=skip_existing,
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


__all__ = ["Model"]
