"""
The :class:`Project`: a directory of receptors and settings, and a view of their results.

A project is a directory holding what the user wrote::

    config.yaml       the settings (written once, by Project.init or stilt init)
    receptors.csv     the receptors (only appended to)

Results go to the output directory ``config.yaml`` names
(:class:`stilt.output.Output`), ``./output`` by default. The simulations a
project defines are its receptors crossed with the variants in
``config.yaml``. A project reads; the workers in :mod:`stilt.execution` are
the only code that writes results.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from stilt.config import ExecutionConfig, ProjectConfig, VariantConfig
from stilt.footprint import Footprint
from stilt.geometry import Geometry
from stilt.meteorology import Met
from stilt.output import Footprints, Jacobian, Output
from stilt.particles import Trajectories
from stilt.receptors import (
    COLUMNS,
    ROW_COLUMNS,
    Receptor,
    append_receptors_csv,
    check_distinct_ids,
    read_receptor_frame,
    read_receptors,
    receptor_from_rows,
    receptor_rows,
    receptors_to_csv,
)
from stilt.simulation import SimID, Simulation
from stilt.transforms import TransformContext

if TYPE_CHECKING:
    import submitit

    from stilt.execution import ReceptorResult
    from stilt.visualization import ProjectPlotAccessor

CONFIG_KEY = "config.yaml"
RECEPTORS_KEY = "receptors.csv"

#: The columns of :attr:`Project.receptors` before the label columns.
RECEPTOR_COLUMNS = ("receptor", "time", "kind", "location")


def _absolute(path: str | Path) -> Path:
    """Return *path* absolute, with ``~`` and variables expanded, so a worker started elsewhere finds the same place."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def project_slug(directory: str | Path) -> str:
    """Return a lowercase, hyphenated name for a project directory, safe for file and job names."""
    name = Path(str(directory).rstrip("/")).name or "project"
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower().replace("_", "-"))
    return re.sub(r"-{2,}", "-", slug).strip("-") or "project"


def _as_receptors(
    receptors: Receptor | Iterable[Receptor] | str | Path,
) -> list[Receptor]:
    """Return receptors given as one, several, or the path of a receptors CSV."""
    if isinstance(receptors, (str, Path)):
        return read_receptors(_absolute(receptors))
    if isinstance(receptors, Receptor):
        return [receptors]
    items = list(receptors)
    if not all(isinstance(item, Receptor) for item in items):
        raise TypeError(
            "Receptors must be a Receptor, an iterable of Receptors, or the path "
            "of a receptors CSV."
        )
    return items


class Project:
    """
    A STILT project: receptors, settings, and the simulations they define.

    ``Project(path)`` opens a project directory and reads it. It is every
    receptor under every variant, and a view of their results: list the
    simulations, see which have finished, load particles and footprints.
    The results are in the output directory ``config.yaml`` names
    (``./output`` by default), which several projects can share.

    Make a new project with :meth:`init` (or ``stilt init``). After that,
    ``config.yaml`` is yours to edit; PYSTILT never rewrites it.

    Parameters
    ----------
    path : str or Path
        Project directory.

    Attributes
    ----------
    directory : Path
        The project directory, absolute.

    Examples
    --------
    Make a project, run it, and load a footprint:

    >>> import stilt
    >>> receptor = stilt.PointReceptor(
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
    >>> project = stilt.Project.init(
    ...     "./my_project",
    ...     receptors=[receptor],
    ...     mets={"hrrr": met},
    ...     n_hours=-24,
    ...     grid={
    ...         "xmin": -113,
    ...         "xmax": -110.5,
    ...         "ymin": 40,
    ...         "ymax": 42,
    ...         "xres": 0.01,
    ...         "yres": 0.01,
    ...     },
    ... )
    >>> project.run()
    >>> foot = project.simulation(receptor.id, "hrrr").footprint

    Open it again later and select simulations with pandas:

    >>> project = stilt.Project("./my_project")
    >>> sims = project.simulations
    >>> project.status(sims[sims.variant == "hrrr"])
    """

    def __init__(self, path: str | Path) -> None:
        self.directory = _absolute(path)

    @classmethod
    def init(
        cls,
        path: str | Path,
        config: ProjectConfig | None = None,
        receptors: Receptor | Iterable[Receptor] | str | Path | None = None,
        **settings: Any,
    ) -> Project:
        """
        Make a new project and return it.

        Writes ``config.yaml``, with only the settings given, and
        ``receptors.csv`` when receptors are given. ``stilt init`` does the
        same from the command line, with a commented starter config.

        Parameters
        ----------
        path : str or Path
            Project directory. It is created if needed.
        config : ProjectConfig, optional
            The settings. Or give them as keywords instead.
        receptors : Receptor, iterable of Receptor, str or Path, optional
            Receptors, or the path of a receptors CSV to copy them from.
        **settings
            Settings for :class:`~stilt.ProjectConfig`, such as ``mets``,
            ``n_hours``, ``numpar``, and ``grid``.

        Raises
        ------
        FileExistsError
            If the directory already has a ``config.yaml``. Open it with
            ``Project(path)`` instead, and edit the file to change settings.
        TypeError
            If both *config* and keyword settings are given.
        """
        if config is not None and settings:
            raise TypeError("Give a ProjectConfig or keyword settings, not both.")
        if config is None:
            config = ProjectConfig(**settings)
        directory = _absolute(path)
        if (directory / CONFIG_KEY).exists():
            raise FileExistsError(
                f"{directory} already has a config.yaml. Open it with "
                "Project(path), and edit config.yaml to change its settings."
            )
        directory.mkdir(parents=True, exist_ok=True)
        config.to_yaml(directory / CONFIG_KEY)
        project = cls(directory)
        if receptors is not None:
            project.add_receptors(receptors)
        return project

    def __repr__(self) -> str:
        return f"Project({str(self.directory)!r})"

    # -- inputs ----------------------------------------------------------------

    @property
    def name(self) -> str:
        """The project's name: the directory name."""
        return self.directory.name

    @property
    def config_path(self) -> Path:
        """The project's ``config.yaml``."""
        return self.directory / CONFIG_KEY

    @property
    def receptors_path(self) -> Path:
        """The project's ``receptors.csv``."""
        return self.directory / RECEPTORS_KEY

    @cached_property
    def config(self) -> ProjectConfig:
        """
        The settings, read from ``config.yaml``.

        Raises
        ------
        FileNotFoundError
            If the project has no ``config.yaml``.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml in {self.directory}. Make a project with "
                "stilt.Project.init(...) or `stilt init`."
            )
        return ProjectConfig.from_yaml(self.config_path)

    @cached_property
    def output(self) -> Output:
        """The output directory, from ``config.output`` (``./output`` by default)."""
        raw = Path(os.path.expandvars(os.path.expanduser(self.config.output)))
        return Output(raw if raw.is_absolute() else (self.directory / raw).resolve())

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

    @cached_property
    def _rows(self) -> pd.DataFrame:
        """
        The rows of ``receptors.csv``, checked, with each row's receptor id.

        No receptor is built here; :meth:`receptor` builds one when asked.
        """
        if not self.receptors_path.exists():
            empty = pd.DataFrame(columns=["time", "longitude", "latitude", "altitude"])
            return receptor_rows(empty)
        return receptor_rows(read_receptor_frame(self.receptors_path))

    @cached_property
    def _positions(self) -> dict[str, np.ndarray]:
        """The positions of each receptor's rows in :attr:`_rows`, by id, in file order."""
        groups = self._rows.groupby("receptor", sort=False).indices
        return {str(rid): positions for rid, positions in groups.items()}

    @cached_property
    def _built(self) -> dict[str, Receptor]:
        """The receptors built so far, by id."""
        return {}

    @cached_property
    def receptors(self) -> pd.DataFrame:
        """
        The receptors, one row per receptor, in file order.

        The columns are ``receptor`` (the id), ``time``, ``kind``
        (``point``, ``column``, or ``multipoint``), ``location`` (the
        location id), and one column per label from ``receptors.csv``.
        :meth:`receptor` gives the receptor itself.
        """
        rows = self._rows
        labels = [c for c in rows.columns if c not in COLUMNS and c not in ROW_COLUMNS]
        first = rows.drop_duplicates("receptor")
        frame = first.loc[:, [*RECEPTOR_COLUMNS, *labels]].reset_index(drop=True)
        return frame.astype({"time": "datetime64[ns]"})

    def receptor(self, receptor_id: str) -> Receptor:
        """
        Return one receptor by id, built from its rows on first use.

        Raises
        ------
        KeyError
            If the project has no receptor with that id.
        """
        built = self._built.get(receptor_id)
        if built is None:
            positions = self._positions.get(receptor_id)
            if positions is None:
                raise KeyError(f"No receptor {receptor_id!r} in {self.directory}.")
            built = receptor_from_rows(self._rows.iloc[positions])
            self._built[receptor_id] = built
        return built

    def add_receptors(
        self, receptors: Receptor | Iterable[Receptor] | str | Path
    ) -> list[str]:
        """
        Add receptors to ``receptors.csv`` and return the ids of the new ones.

        The file is created if needed, and otherwise only appended to, in its
        own columns (:func:`stilt.receptors.append_receptors_csv`). Receptors
        it already holds are skipped.

        Parameters
        ----------
        receptors : Receptor, iterable of Receptor, str or Path
            Receptors, or the path of a receptors CSV to add them from.

        Raises
        ------
        ValueError
            If a new receptor has the id of a different receptor, since an
            id names a receptor's result files.
        """
        batch = _as_receptors(receptors)
        known = [self.receptor(r.id) for r in batch if r.id in self._positions]
        check_distinct_ids([*known, *batch])
        new = list({r.id: r for r in batch if r.id not in self._positions}.values())
        if new:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.receptors_path.exists():
                text = self.receptors_path.read_text()
                self.receptors_path.write_text(append_receptors_csv(text, new))
            else:
                self.receptors_path.write_text(receptors_to_csv(new))
            # The views below were read from the file that just changed.
            for name in ("_rows", "_positions", "_built", "receptors", "simulations"):
                vars(self).pop(name, None)
        return [r.id for r in new]

    # -- simulations -----------------------------------------------------------

    @cached_property
    def simulations(self) -> pd.DataFrame:
        """
        Every receptor under every variant, one row per simulation.

        Receptor by receptor, with variants in config order. The columns are
        ``receptor``, ``variant``, ``group`` (the variant's name in
        ``config.yaml``, shared by the realizations of one variant), then the
        other columns of :attr:`receptors`. Select rows with pandas, and pass
        them to :meth:`status`, :meth:`incomplete`, or the loaders.

        Examples
        --------
        >>> sims = project.simulations
        >>> sims[(sims.variant == "hrrr") & (sims.site == "WBB")]
        """
        variants = pd.DataFrame(
            [(name, v.group) for name, v in self.variants.items()],
            columns=["variant", "group"],
        )
        frame = self.receptors.merge(variants, how="cross")
        first = ["receptor", "variant", "group"]
        return frame.loc[:, first + [c for c in frame.columns if c not in first]]

    def simulation(self, receptor_id: str, variant: str) -> Simulation:
        """
        Return one simulation: a receptor under a variant.

        A simulation is a value built from its receptor, its variant, and the
        output directory, so it is cheap. It knows where its results are and
        loads them.

        Raises
        ------
        KeyError
            If the project has no such receptor or variant.
        """
        if variant not in self.variants:
            raise KeyError(
                f"No variant {variant!r}; the variants are {list(self.variants)}."
            )
        return Simulation(
            self.receptor(receptor_id), self.variants[variant], self.output
        )

    def _selected(self, simulations: pd.DataFrame | None) -> pd.DataFrame:
        """Return *simulations*, or every simulation of the project when ``None``."""
        return self.simulations if simulations is None else simulations

    def _present(
        self, simulations: pd.DataFrame
    ) -> dict[str, tuple[frozenset[str], frozenset[str] | None]]:
        """
        Return, per variant, the selected receptors with particles and with a footprint.

        This is :meth:`stilt.Simulation.is_complete`'s rule read from a
        listing of the date folders the selection falls in, in place of a
        file check per simulation: on a large project the checks are the
        slow part. The footprint set is ``None`` for a variant that makes no
        footprint.
        """
        present: dict[str, tuple[frozenset[str], frozenset[str] | None]] = {}
        for name, rows in simulations.groupby("variant", sort=False):
            variant = self.variants[str(name)]
            among = set(rows["receptor"])
            folder = self.output.find_particles(variant.transport)
            particles = frozenset(folder.receptors(among) if folder is not None else ())
            footprints: frozenset[str] | None = None
            if variant.footprint is not None:
                feet = (
                    None
                    if folder is None
                    else self.output.find_footprints(folder.hash, variant.footprint)
                )
                footprints = frozenset(
                    feet.receptors(among) if feet is not None else ()
                )
            present[str(name)] = (particles, footprints)
        return present

    def status(self, simulations: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Return the simulations with columns saying which results exist.

        Parameters
        ----------
        simulations : DataFrame, optional
            Rows of :attr:`simulations`. Defaults to all of them.

        Returns
        -------
        pandas.DataFrame
            *simulations* with four more columns. ``particles`` and
            ``footprint`` are ``True`` when the result exists, ``False`` when
            it is missing, and ``NA`` when the simulation does not make it.
            ``empty`` is ``True`` when the footprint is empty (no particle
            reached the grid), which needs each footprint file opened.
            ``complete`` is :meth:`~stilt.Simulation.is_complete`.
        """
        sims = self._selected(simulations)
        present = self._present(sims)
        pairs = list(zip(sims["receptor"], sims["variant"], strict=True))
        particles = [r in present[v][0] for r, v in pairs]
        feet = [
            None if (have := present[v][1]) is None else r in have for r, v in pairs
        ]
        empty = [
            None if f is None else f and self.simulation(r, v).empty_reason is not None
            for (r, v), f in zip(pairs, feet, strict=True)
        ]
        complete = [p and f is not False for p, f in zip(particles, feet, strict=True)]
        return sims.assign(
            particles=pd.array(particles, dtype="boolean"),
            footprint=pd.array(feet, dtype="boolean"),
            empty=pd.array(empty, dtype="boolean"),
            complete=pd.array(complete, dtype="bool"),
        )

    def incomplete(self, simulations: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Return the simulations that are missing an expected result.

        The same rows as ``status()`` marks not ``complete``, found without
        opening any footprint file, so it is quick on a large project.

        Parameters
        ----------
        simulations : DataFrame, optional
            Rows of :attr:`simulations`. Defaults to all of them.
        """
        sims = self._selected(simulations)
        present = self._present(sims)
        missing = [
            r not in present[v][0]
            or (present[v][1] is not None and r not in present[v][1])
            for r, v in zip(sims["receptor"], sims["variant"], strict=True)
        ]
        return sims.loc[np.asarray(missing, dtype=bool)]

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
            "particles": [
                r.key for r in self.output.particle_sets() if r.hash not in runs
            ],
            "footprints": [
                f.key for f in self.output.footprint_sets() if f.hash not in feet
            ],
        }

    def transform_context(self, sim: Simulation) -> TransformContext:
        """Return the context a simulation's transforms run with: its receptor, variant, and this project's directory."""
        return TransformContext(
            receptor=sim.receptor, variant=sim.variant.name, directory=self.directory
        )

    # -- results ---------------------------------------------------------------

    def _simulations_of(self, simulations: pd.DataFrame | None) -> list[Simulation]:
        """Return the :class:`Simulation` of each selected row."""
        sims = self._selected(simulations)
        return [
            self.simulation(r, v)
            for r, v in zip(sims["receptor"], sims["variant"], strict=True)
        ]

    def load_particles(
        self, simulations: pd.DataFrame | None = None
    ) -> dict[SimID, Trajectories]:
        """
        Load the particles of every selected simulation that has them.

        Parameters
        ----------
        simulations : DataFrame, optional
            Rows of :attr:`simulations`. Defaults to all of them.

        Returns
        -------
        dict
            :class:`~stilt.Trajectories` by :class:`~stilt.SimID`.
        """
        return {
            sim.id: sim.particles
            for sim in self._simulations_of(simulations)
            if sim.has_particles
        }

    def load_footprints(
        self, simulations: pd.DataFrame | None = None
    ) -> dict[SimID, Footprint]:
        """
        Load the footprint of every selected simulation that has one.

        An empty footprint is left out, as :attr:`stilt.Simulation.footprint`
        is ``None`` for it.

        Parameters
        ----------
        simulations : DataFrame, optional
            Rows of :attr:`simulations`. Defaults to all of them.

        Returns
        -------
        dict
            :class:`~stilt.Footprint` by :class:`~stilt.SimID`.
        """
        return {
            sim.id: foot
            for sim in self._simulations_of(simulations)
            if sim.makes_footprint
            and sim.has_footprint
            and (foot := sim.footprint) is not None
        }

    def jacobian(
        self,
        target: Geometry,
        time_bins: pd.IntervalIndex,
        variant: str,
        receptors: Iterable[str] | None = None,
    ) -> Jacobian:
        """
        Sum a variant's footprints onto a target, per time bin, as one sparse matrix.

        See :meth:`stilt.output.Footprints.jacobian`.

        Parameters
        ----------
        target : Geometry
            Where the fluxes are: a grid, mesh, or set of zones.
        time_bins : pandas.IntervalIndex
            Flux time bins, closed on the left.
        variant : str
            The variant whose footprints to use.
        receptors : iterable of str, optional
            Receptor ids, one row each. Defaults to every receptor.

        Raises
        ------
        ValueError
            If the variant has no grid or no footprints yet, or ``time_bins``
            is not closed on the left.
        """
        settings = self.variants[variant]
        if settings.footprint is None:
            raise ValueError(f"Variant {variant!r} makes no footprints (no grid).")
        folder = self.output.find_particles(settings.transport)
        feet = (
            None
            if folder is None
            else self.output.find_footprints(folder.hash, settings.footprint)
        )
        if feet is None:
            raise ValueError(f"Variant {variant!r} has no footprints yet.")
        ids = list(self._positions) if receptors is None else list(receptors)
        return feet.jacobian(target, time_bins, receptors=ids)

    @cached_property
    def plot(self) -> ProjectPlotAccessor:
        """Plotting methods, such as ``project.plot.availability()``."""
        from stilt.visualization import ProjectPlotAccessor

        return ProjectPlotAccessor(self)

    # -- running ---------------------------------------------------------------

    def run(
        self,
        skip_existing: bool = True,
        compute_root: str | Path | None = None,
        execution: ExecutionConfig | None = None,
    ) -> list[ReceptorResult]:
        """
        Run every simulation that has not finished, and wait for it.

        Shorthand for :func:`stilt.execution.run`. The receptors with missing
        results run here, or as a Slurm job array when ``execution`` says
        ``backend: slurm``. Use :meth:`submit` to submit to Slurm and return
        at once.

        Parameters
        ----------
        skip_existing : bool, default True
            Skip simulations whose results all exist. ``False`` runs every
            simulation again.
        compute_root : str or Path, optional
            Scratch directory under which HYSPLIT runs. Defaults to
            ``PYSTILT_COMPUTE_ROOT``, then to ``$TMPDIR/pystilt/<project name>``.
        execution : ExecutionConfig, optional
            Where to run and with what resources, in place of the config's
            ``execution`` settings.

        Returns
        -------
        list of ReceptorResult
            One per receptor that ran.
        """
        from stilt.execution import run

        return run(
            self,
            execution=execution,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )

    def submit(
        self,
        skip_existing: bool = True,
        compute_root: str | Path | None = None,
        execution: ExecutionConfig | None = None,
    ) -> list[submitit.Job[Any]]:
        """
        Submit every simulation that has not finished to Slurm, and return at once.

        Shorthand for :func:`stilt.execution.submit`. The parameters are those
        of :meth:`run`.

        Returns
        -------
        list of submitit.Job
            One per array task. ``job.wait()``, ``job.result()``,
            ``job.stdout()``, and ``job.cancel()`` follow and control them.

        Raises
        ------
        ValueError
            If the execution backend is not Slurm.
        """
        from stilt.execution import submit

        return submit(
            self,
            execution=execution,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )


__all__ = ["CONFIG_KEY", "RECEPTORS_KEY", "Project", "project_slug"]
