"""The :class:`Model`, which sets up, runs, and loads a STILT project."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from stilt.collections import ReceptorCollection, SimulationCollection
from stilt.config import (
    ModelConfig,
    RuntimeSettings,
    VariantConfig,
)
from stilt.execution import (
    Executor,
    JobHandle,
    LocalHandle,
    get_executor,
)
from stilt.meteorology import MetStream
from stilt.output import Footprints, Output
from stilt.project import Project
from stilt.receptors import Receptor
from stilt.service import PostgresQueue, resolve_queue
from stilt.simulation import SimID, Simulation
from stilt.transforms import TransformContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.visualization import ModelPlotAccessor


class Model:
    """
    A STILT project: receptors, settings, and the simulations they define.

    A model runs every receptor once per variant and loads the resulting
    trajectories and footprints. Its inputs live in the project directory;
    its results in the output directory ``config.yaml`` names (``./output``
    by default), which several projects can share. Settings and receptors
    given here are saved to the project when the model runs, so
    ``Model(project)`` opens it again later.

    Parameters
    ----------
    project : str or Path, optional
        Project directory. A temporary directory is used when omitted.
    receptors : Receptor, iterable of Receptor, str or Path, optional
        Receptors to run, or the path of a receptors CSV (relative to the
        project directory). Defaults to the project's ``receptors.csv``.
    config : ModelConfig, optional
        Model settings. Defaults to the project's ``config.yaml``.
    compute_root : str or Path, optional
        Scratch directory under which HYSPLIT runs. Defaults to
        ``PYSTILT_COMPUTE_ROOT``, then to ``$TMPDIR/pystilt/<project name>``.
    runtime : RuntimeSettings, optional
        Settings for this machine (work-queue URL and compute root). Read
        from ``PYSTILT_*`` environment variables when omitted.
    **kwargs
        Settings for :class:`~stilt.ModelConfig`, such as ``n_hours``,
        ``numpar``, ``mets``, and ``grid``. Cannot be combined with *config*.

    Attributes
    ----------
    project : Project
        The project's input files.
    output : Output
        The output directory.
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
        self.project = Project(project)
        self.compute_root = self._resolve_compute_root(compute_root)
        if config is not None and kwargs:
            raise TypeError("Cannot pass both a ModelConfig and keyword settings.")
        self._config = ModelConfig(**kwargs) if kwargs else config
        # A config given here is the user's latest word and is written to the
        # project; one loaded from the project is never rewritten.
        self._config_given = self._config is not None

        self._receptors_input = receptors

    def __repr__(self) -> str:
        return f"Model(project={self.project.root!r})"

    def _resolve_compute_root(self, compute_root: str | Path | None) -> Path:
        """Return the scratch directory under which HYSPLIT runs."""
        if compute_root is not None:
            raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
            return Path(raw).resolve()
        if self.runtime.compute_root is not None:
            return self.runtime.compute_root.expanduser().resolve()
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        # Resolved like the explicit forms, so a worker handed this path gets
        # the same one (macOS keeps TMPDIR under the /var -> /private/var link).
        return (Path(tmp_root) / "pystilt" / self.project.name).resolve()

    # -- Inputs ----------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """Model settings, from ``config.yaml`` unless given to the constructor."""
        if self._config is None:
            self._config = self.project.load_config()
        return self._config

    @cached_property
    def output(self) -> Output:
        """The output directory, from ``config.output`` (``./output`` by default)."""
        return Output(self.project.output_path(self.config))

    @cached_property
    def receptors(self) -> ReceptorCollection:
        """Receptors, by position (``receptors[0]``) or by id (``receptors[receptor_id]``)."""
        return ReceptorCollection(self._receptors_input, project=self.project)

    @cached_property
    def variants(self) -> dict[str, VariantConfig]:
        """
        Settings of each variant, by name, in config order.

        A realization group appears once per realization (``hrrr-err-0``,
        ``hrrr-err-1``, ...).
        """
        return self.config.resolve_variants()

    @cached_property
    def mets(self) -> dict[str, MetStream]:
        """Meteorology sources declared in the config, by name."""
        return {name: MetStream(name, cfg) for name, cfg in self.config.mets.items()}

    @cached_property
    def queue(self) -> PostgresQueue | None:
        """Postgres work queue, or ``None`` when ``PYSTILT_DB_URL`` is not set."""
        return resolve_queue(self.runtime)

    def _forget_simulations(self) -> None:
        """Drop the cached receptors and simulations, so they are rebuilt from the project."""
        self.__dict__.pop("receptors", None)
        self.__dict__.pop("simulations", None)

    def unreferenced(self) -> dict[str, list[str]]:
        """
        Return the output folders no variant of this config points at.

        Returns
        -------
        dict
            ``{"particles": [keys], "footprints": [keys]}``, the ``settings=``
            values of runs and footprint folders in the output directory that
            no current variant produces or reads. They come from settings
            that were changed or variants that were dropped, or from another
            project sharing the directory. PYSTILT never deletes them.
        """
        runs = {v.transport.hash for v in self.variants.values()}
        feet = {
            Footprints.hash_for(v.transport.hash, v.footprint)
            for v in self.variants.values()
            if v.footprint is not None
        }
        return {
            "particles": [r.key for r in self.output.runs() if r.hash not in runs],
            "footprints": [
                f.key for f in self.output.footprint_sets() if f.hash not in feet
            ],
        }

    def register(self, receptors: Iterable[Receptor] | None = None) -> list[str]:
        """
        Save the model's settings and receptors to the project.

        Workers rebuild the model from the project alone, so :meth:`run`
        calls this first. A config given in Python is written to
        ``config.yaml`` with only the settings that were set. A
        ``config.yaml`` loaded from the project is left as it is. Receptors
        not yet in ``receptors.csv`` are appended to it. When a work queue is
        configured, the receptors are added to it.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to add to the project. Defaults to the model's own
            receptors.

        Returns
        -------
        list of str
            Ids of the receptors registered, including any the project
            already had.
        """
        if self._config_given or not self.project.has_config:
            self.project.save_config(self.config)

        if receptors is None and self._receptors_input is None:
            batch = list(self.receptors)  # the project's own file; nothing to add
        else:
            batch = list(self.receptors) if receptors is None else list(receptors)
            if self.project.add_receptors(batch):
                # The registered set changed: rebuild receptors from the project.
                self._receptors_input = None
                self._forget_simulations()

        receptor_ids = [str(r.id) for r in batch]
        if self.queue is not None:
            self.queue.register(receptor_ids)
        return receptor_ids

    # -- Simulations -----------------------------------------------------------

    def simulation(self, key: str | SimID | tuple[str, str]) -> Simulation:
        """
        Return one simulation by id.

        A simulation is a value built from its receptor, its variant, and the
        output directory, so equal keys give equal simulations. Nothing is
        written to disk.

        Parameters
        ----------
        key : str, SimID or tuple of (str, str)
            ``"<receptor_id>/<variant>"`` or a ``(receptor_id, variant)`` pair.
        """
        sid = SimID.parse(key)
        return Simulation(
            self.receptors[sid.receptor], self.variants[sid.variant], self.output
        )

    def transform_context(self, sim: Simulation) -> TransformContext:
        """Return the context a simulation's transforms run with: its receptor, variant, and this project's directory."""
        return TransformContext(
            receptor=sim.receptor,
            variant=sim.variant.name,
            directory=self.project.directory,
        )

    @cached_property
    def simulations(self) -> SimulationCollection:
        """Every receptor under every variant, indexed by ``(receptor_id, variant)``."""
        return SimulationCollection(self)

    @cached_property
    def plot(self) -> ModelPlotAccessor:
        """Plotting methods, such as ``model.plot.availability()``."""
        from stilt.visualization import ModelPlotAccessor

        return ModelPlotAccessor(self)

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
        then starts workers for each receptor with missing results. A worker
        runs HYSPLIT once for each distinct set of transport settings whose
        particles are missing, then calculates the footprint of every
        variant that has a grid.

        Parameters
        ----------
        executor : Executor, optional
            Where to run the workers. Defaults to the one set by
            ``config.execution`` (local processes unless configured).
        skip_existing : bool, default True
            Skip simulations whose outputs all exist. ``False`` runs every
            simulation again.
        wait : bool, default True
            Block until the workers finish. With ``False`` a Slurm or
            Kubernetes run returns once it is submitted. A local run always
            finishes before this returns.

        Returns
        -------
        JobHandle
            Handle to the started workers.
        """
        self._forget_simulations()
        resolved_executor = executor or get_executor(self.config.execution or {})

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
            handle.wait()

        return handle


__all__ = ["Model"]
