"""The :class:`Model`, which sets up, runs, and loads a STILT project."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from stilt.collections import ReceptorCollection, SimulationCollection
from stilt.config import ExecutionConfig, ModelConfig, VariantConfig
from stilt.execution import JobHandle
from stilt.meteorology import Met
from stilt.output import Footprints, Output
from stilt.project import Project
from stilt.receptors import Receptor
from stilt.simulation import SimID, Simulation
from stilt.transforms import TransformContext

if TYPE_CHECKING:
    from stilt.visualization import ModelPlotAccessor


class Model:
    """
    A STILT project: receptors, settings, and the simulations they define.

    A model is every receptor under every variant, and a view of their
    results: select simulations, load trajectories and footprints, see what
    has finished. Its inputs live in the project directory; its results in
    the output directory ``config.yaml`` names (``./output`` by default),
    which several projects can share. The model itself writes nothing.
    :meth:`run` hands it to :func:`stilt.execution.run`, which saves the
    settings and receptors given here to the project, so ``Model(project)``
    opens it again later, and starts the workers.

    Parameters
    ----------
    project : str or Path, optional
        Project directory. A temporary directory is used when omitted.
    receptors : Receptor, iterable of Receptor, str or Path, optional
        Receptors to run, or the path of a receptors CSV (relative to the
        project directory). Defaults to the project's ``receptors.csv``.
    config : ModelConfig, optional
        Model settings. Defaults to the project's ``config.yaml``.
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
    mets : dict of str to Met
        The mets, by name.
    simulations : SimulationCollection
        Every receptor under every variant.
    plot : ModelPlotAccessor
        Plotting methods.

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
        **kwargs,
    ):
        self.project = Project(project)
        if config is not None and kwargs:
            raise TypeError("Cannot pass both a ModelConfig and keyword settings.")
        self._config = ModelConfig(**kwargs) if kwargs else config
        self._receptors_input = receptors

    def __repr__(self) -> str:
        return f"Model(project={self.project.root!r})"

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
    def mets(self) -> dict[str, Met]:
        """The mets declared in the config, by name."""
        return {name: Met(name, cfg) for name, cfg in self.config.mets.items()}

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

        Shorthand for :func:`stilt.execution.register`. A model that reads
        the project's ``receptors.csv`` reads it again afterwards, so
        receptors added here show up in ``model.receptors``. A model built
        with its own receptors keeps those.

        Parameters
        ----------
        receptors : iterable of Receptor, optional
            Receptors to add to the project. Defaults to the model's own.

        Returns
        -------
        list of str
            Ids of the receptors registered, including any the project
            already had.
        """
        from stilt.execution import register

        ids = register(self, receptors)
        if self._receptors_input is None:
            # The views below were read from the file that just changed.
            vars(self).pop("receptors", None)
            vars(self).pop("simulations", None)
        return ids

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
        skip_existing: bool = True,
        wait: bool = True,
        compute_root: str | Path | None = None,
        execution: ExecutionConfig | None = None,
    ) -> JobHandle:
        """
        Run every simulation that has not finished.

        Shorthand for :func:`stilt.execution.run`, which saves the settings
        and receptors to the project and runs the receptors with missing
        results, here or on Slurm.

        Parameters
        ----------
        skip_existing : bool, default True
            Skip simulations whose outputs all exist. ``False`` runs every
            simulation again.
        wait : bool, default True
            Block until the work finishes. With ``False`` a Slurm run
            returns once it is submitted. A local run always finishes before
            this returns.
        compute_root : str or Path, optional
            Scratch directory under which HYSPLIT runs. Defaults to
            ``PYSTILT_COMPUTE_ROOT``, then to
            ``$TMPDIR/pystilt/<project name>``.
        execution : ExecutionConfig, optional
            Where to run and with what resources, in place of the config's
            ``execution`` settings.

        Returns
        -------
        JobHandle
            Handle to the work.
        """
        from stilt.execution import run

        return run(
            self,
            execution=execution,
            skip_existing=skip_existing,
            wait=wait,
            compute_root=compute_root,
        )


__all__ = ["Model"]
