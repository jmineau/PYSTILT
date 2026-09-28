"""One simulation: a receptor run under one variant, its outputs and completion."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from stilt.config import FootprintConfig, STILTParams, VariantConfig
from stilt.errors import (
    EmptyTrajectoryError,
    identify_failure_reason,
)
from stilt.footprint import Footprint
from stilt.hysplit import HYSPLITDriver
from stilt.meteorology import MetStream
from stilt.project import (
    SIMULATION_LOG_FILENAME,
    SIMULATION_MET_DIRNAME,
    resolve_directory,
    simulation_prefix,
)
from stilt.receptors import Receptor, ReceptorID
from stilt.store import Store
from stilt.trajectory import Trajectories
from stilt.transforms import (
    ParticleTransform,
    TransformContext,
    apply_transforms,
)

if TYPE_CHECKING:
    from stilt.visualization import SimulationPlotAccessor

logger = logging.getLogger(__name__)


class SimID(NamedTuple):
    """
    Identity of one simulation: a receptor under a variant.

    Its string form is ``"{receptor_id}/{variant}"``, which is also the
    simulation's path below ``simulations/by-id/``.
    """

    receptor: ReceptorID
    variant: str

    def __str__(self) -> str:
        return f"{self.receptor}/{self.variant}"

    def __fspath__(self) -> str:
        """Allow ``root / sim_id`` to build the simulation directory."""
        return str(self)

    @classmethod
    def parse(cls, value: str | SimID | tuple[str, str]) -> SimID:
        """Build a :class:`SimID` from its string form or a ``(receptor, variant)`` pair."""
        if isinstance(value, SimID):
            return value
        if isinstance(value, tuple):
            receptor, variant = value
        else:
            receptor, sep, variant = str(value).partition("/")
            if not sep or not variant:
                raise ValueError(
                    f"Invalid sim id {value!r}; expected '{{receptor_id}}/{{variant}}'."
                )
        return cls(ReceptorID(receptor), variant)


class Simulation:
    """
    One receptor under one variant: a HYSPLIT run, or a footprint derived
    from another simulation's particles.

    A simulation owns its output filenames, their store keys, and the single
    definition of which outputs exist and whether it is complete.

    Parameters
    ----------
    receptor
        Where and when particles are released.
    config
        The resolved variant this receptor runs under: transport settings,
        footprint settings, and the variant name (``config.name`` is the
        simulation's variant, so ``id == receptor.id / config.name``).
    met
        The met stream to run HYSPLIT with. ``None`` only for a derived
        simulation, which uses its parent's particles.
    parent
        The simulation whose trajectory this one rasterizes (a ``from:``
        variant). Such a simulation never runs HYSPLIT.
    directory
        Compute-local working directory. A temporary one is created when
        omitted. Nothing is created on disk until an output is written.
    store
        Output store the outputs are published to and read back from when
        they are not on local disk. When the store's location for this
        simulation *is* ``directory``, publishing is a no-op.
    """

    def __init__(
        self,
        receptor: Receptor,
        config: VariantConfig,
        *,
        met: MetStream | None = None,
        parent: Simulation | None = None,
        directory: str | Path | None = None,
        store: Store | None = None,
    ):
        if parent is None and met is None:
            raise ValueError("A simulation that runs HYSPLIT needs a met stream.")
        self.id = SimID(receptor.id, config.name)
        self.receptor = receptor
        self.config = config
        self.met = met
        self.params: STILTParams = config.stilt_params()
        self.footprint_config: FootprintConfig | None = config.footprint
        self.parent = parent
        if directory is None:
            directory = resolve_directory(prefix="pystilt_") / self.id
        self.directory = resolve_directory(directory)
        self.key_prefix = simulation_prefix(self.id)
        self._store = store

        # Lazy state
        self._source_met_files: list[Path] | None = None
        self._met_files: list[Path] | None = None
        self._trajectories: Trajectories | None = None
        self._footprint: Footprint | None = None
        self._plot: SimulationPlotAccessor | None = None

    def __repr__(self) -> str:
        """Compact developer-facing simulation representation."""
        return f"Simulation(id={str(self.id)!r}, directory={str(self.directory)!r})"

    @property
    def variant(self) -> str:
        """Variant name."""
        return self.id.variant

    @property
    def is_derived(self) -> bool:
        """Whether this simulation rasterizes another simulation's trajectory."""
        return self.parent is not None

    # -- Paths and keys --------------------------------------------------------

    @property
    def met_dir(self) -> Path:
        """Compute-local meteorology staging directory."""
        return self.directory / SIMULATION_MET_DIRNAME

    @property
    def log_path(self) -> Path:
        """Compute-local HYSPLIT log path."""
        return self.directory / SIMULATION_LOG_FILENAME

    @property
    def trajectories_path(self) -> Path:
        """Compute-local trajectory parquet path (the parent's when derived)."""
        if self.parent is not None:
            return self.parent.trajectories_path
        return self.directory / f"{self.id.receptor}_traj.parquet"

    @property
    def footprint_path(self) -> Path:
        """Compute-local footprint netCDF path."""
        return self.directory / f"{self.id.receptor}_foot.nc"

    @property
    def empty_footprint_path(self) -> Path:
        """Compute-local marker recording that the footprint is legitimately empty."""
        return self.footprint_path.with_suffix(".empty")

    def key(self, path: str | Path) -> str:
        """Return the store key for one file under this simulation's directory."""
        return f"{self.key_prefix}/{Path(path).name}"

    def resolve(self, path: Path) -> Path | None:
        """Return a local path to one output from disk or the store, else ``None``."""
        if path.exists():
            return path
        if self._store is not None:
            key = self.key(path)
            if self._store.exists(key):
                return self._store.local_path(key)
        return None

    # -- Presence and completion -----------------------------------------------

    @property
    def has_trajectory(self) -> bool:
        """Whether the trajectory parquet exists on disk or in the store."""
        if self.parent is not None:
            return self.parent.has_trajectory
        return self.resolve(self.trajectories_path) is not None

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint is complete.

        The empty marker counts: a run that legitimately produced no footprint
        is a terminal outcome, not missing work.
        """
        return (
            self.resolve(self.footprint_path) is not None
            or self.resolve(self.empty_footprint_path) is not None
        )

    @property
    def runs_hysplit(self) -> bool:
        """Whether this simulation produces its own trajectory (it is not derived)."""
        return self.parent is None

    @property
    def makes_footprint(self) -> bool:
        """Whether this simulation produces a footprint (its variant has a grid)."""
        return self.footprint_config is not None

    def is_complete(self) -> bool:
        """Whether every expected output exists: the trajectory unless derived, the footprint if a grid is set."""
        return (not self.runs_hysplit or self.has_trajectory) and (
            not self.makes_footprint or self.has_footprint
        )

    def publish(self) -> None:
        """
        Copy this simulation's local outputs into the store.

        A no-op when there is no store or when the store's location for this
        simulation is already ``directory``.
        """
        if self._store is None:
            return
        paths = [self.log_path, self.footprint_path, self.empty_footprint_path]
        if not self.is_derived:
            paths.append(self.trajectories_path)
        for path in paths:
            self._store.publish_file(path, self.key(path))

    def delete(self) -> None:
        """
        Remove this simulation's outputs from the store and its working directory.

        Afterwards the simulation is incomplete and reruns on the next run. A
        derived simulation loses only its footprint; its parent's trajectory is
        the parent's to delete.
        """
        paths = [self.log_path, self.footprint_path, self.empty_footprint_path]
        if not self.is_derived:
            paths.append(self.trajectories_path)
        if self._store is not None:
            for path in paths:
                self._store.delete(self.key(path))
        shutil.rmtree(self.directory, ignore_errors=True)
        self._trajectories = None
        self._footprint = None

    def write_empty_footprint_marker(self) -> Path:
        """Create the empty-footprint marker."""
        marker = self.empty_footprint_path
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
        return marker

    def clear_empty_footprint_marker(self) -> None:
        """Remove the empty-footprint marker."""
        self.empty_footprint_path.unlink(missing_ok=True)

    @property
    def plot(self) -> SimulationPlotAccessor:
        """Plotting namespace (e.g. ``sim.plot.map()``)."""
        if self._plot is None:
            from stilt.visualization import SimulationPlotAccessor

            self._plot = SimulationPlotAccessor(self)
        return self._plot

    # -- Status ----------------------------------------------------------------

    @property
    def is_backward(self) -> bool:
        """Return True when ``n_hours < 0`` (backward Lagrangian run)."""
        return self.params.n_hours < 0

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Start and stop datetimes spanned by this simulation.

        Returns
        -------
        tuple[datetime, datetime]
            ``(start, stop)`` where *start* < *stop* regardless of run direction.
        """
        r_time = self.receptor.time
        if self.is_backward:
            start = r_time + dt.timedelta(hours=self.params.n_hours)
            stop = r_time
        else:
            start = r_time
            stop = r_time + dt.timedelta(hours=self.params.n_hours)
        return start, stop

    @property
    def outcome(self) -> str | None:
        """
        How this simulation ended, read from its outputs and log.

        Returns
        -------
        str or None
            ``'complete'`` if every expected output exists, a
            ``'failed:<reason>'`` string if HYSPLIT failed, or ``None`` if the
            simulation has not run.
        """
        if self.is_complete():
            return "complete"
        log_path = self.resolve(self.log_path)
        if log_path is None:
            return None
        return f"failed:{identify_failure_reason(log_path.parent)}"

    # -- Lazy accessors --------------------------------------------------------

    def _met_stream(self) -> MetStream:
        """The met stream, from the parent when derived."""
        if self.met is not None:
            return self.met
        if self.parent is not None:
            return self.parent._met_stream()
        raise ValueError(f"{self.id} has no met stream.")

    @property
    def source_met_files(self) -> list[Path]:
        """Archive/source met files required for this simulation's time window."""
        if not self._source_met_files:
            self._source_met_files = self._met_stream().required_files(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
            )
        return self._source_met_files

    @property
    def met_files(self) -> list[Path]:
        """
        Compute-local meteorology files staged for HYSPLIT execution.

        The output/source archive paths remain available via
        :attr:`source_met_files`.
        """
        if not self._met_files:
            self._met_files = self._met_stream().stage_files_for_simulation(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
                target_dir=self.met_dir,
            )
        return self._met_files

    @property
    def log(self) -> str:
        """
        Contents of the HYSPLIT stdout log file.

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self.resolve(self.log_path)
        if log_path is None:
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        return log_path.read_text()

    @property
    def trajectories(self) -> Trajectories | None:
        """
        Particle trajectories, loaded from parquet on first access.

        A derived simulation returns its parent's. ``None`` if no trajectory
        parquet exists and the simulation has not been run in this process.
        """
        if self.parent is not None:
            return self.parent.trajectories
        if self._trajectories is None:
            traj_path = self.resolve(self.trajectories_path)
            if traj_path is not None:
                self._trajectories = Trajectories.from_parquet(traj_path)
        return self._trajectories

    @property
    def footprint(self) -> Footprint | None:
        """The footprint, loaded from disk on first access; ``None`` if absent."""
        if self._footprint is None:
            path = self.resolve(self.footprint_path)
            if path is not None:
                self._footprint = Footprint.from_netcdf(path)
        return self._footprint

    # -- Execution -------------------------------------------------------------

    def run_trajectories(
        self,
        timeout: int | None = None,
        rm_dat: bool | None = None,
        write: bool = False,
    ) -> None:
        """
        Run HYSPLIT once, populating ``self.trajectories``.

        Parameters
        ----------
        timeout : int, optional
            Wall-clock cap in seconds for the hycs_std run. Defaults to
            ``params.timeout``.
        rm_dat : bool, optional
            Defaults to ``params.rm_dat``.
        write : bool
            If True, persist the trajectories to ``self.trajectories_path``.

        Raises
        ------
        HYSPLITTimeoutError, HYSPLITFailureError, NoParticleOutputError,
        EmptyTrajectoryError
        """
        if self.is_derived:
            raise ValueError(
                f"{self.id} is derived from {self.parent.id}; it has no HYSPLIT run."  # type: ignore[union-attr]
            )
        if rm_dat is None:
            rm_dat = self.params.rm_dat
        if timeout is None:
            timeout = self.params.timeout

        self.directory.mkdir(parents=True, exist_ok=True)
        runner = HYSPLITDriver(
            directory=self.directory,
            receptor=self.receptor,
            params=self.params,
            met_files=self.met_files,
        )
        runner.prepare()
        result = runner.execute(timeout=timeout, rm_dat=rm_dat)
        if result.log_path != self.log_path and result.log_path.exists():
            self.log_path.write_text(result.log_path.read_text())
        if result.particles.empty:
            raise EmptyTrajectoryError(f"No trajectory data for {self.id}")
        self._trajectories = Trajectories.from_particles(
            result.particles,
            receptor=self.receptor,
            params=self.params,
            met_files=self.source_met_files,
        )
        if write:
            self._trajectories.to_parquet(self.trajectories_path)

    def generate_footprint(
        self,
        config: FootprintConfig | None = None,
        write: bool = False,
        transforms: Sequence[ParticleTransform] | None = None,
        context: TransformContext | None = None,
    ) -> Footprint:
        """
        Rasterize the footprint from the trajectories.

        Runs HYSPLIT first when no trajectory exists yet. The result is kept on
        ``self.footprint``.

        Parameters
        ----------
        config : FootprintConfig, optional
            Footprint settings. Defaults to the variant's own; pass one to try
            other settings in memory (``sim.footprint_config.replace(...)``).
        write : bool
            If True, write the footprint netCDF to the simulation directory.
        transforms : sequence, optional
            Extra particle transforms applied after ``config.transforms`` and
            before rasterization (any object with ``apply(particles, context)``).
        context : TransformContext, optional
            Context handed to every transform. Defaults to one built from the
            receptor, variant, and project store.
        """
        if config is None:
            config = self.footprint_config
        if config is None:
            raise TypeError(
                f"{self.id} has no footprint settings; pass a FootprintConfig."
            )

        traj = self.trajectories
        if traj is None:
            self.run_trajectories(write=write)
            traj = self.trajectories
        assert traj is not None  # run_trajectories raises rather than leaving None

        particles = traj.data
        all_transforms = [*config.transforms, *(transforms or [])]
        if all_transforms:
            particles = apply_transforms(
                particles, all_transforms, context or self.transform_context()
            )
        foot = Footprint.calculate(
            particles, receptor=traj.receptor, config=config, name=self.variant
        )
        self._footprint = foot
        if write:
            foot.to_netcdf(self.footprint_path)
        return foot

    def transform_context(self) -> TransformContext:
        """
        The :class:`~stilt.TransformContext` this simulation hands its transforms.

        Carries the receptor, the variant name, and the project store (so a
        transform can read per-receptor inputs such as an averaging-kernel
        table). Use it to apply the footprint's transforms outside
        :meth:`generate_footprint`.
        """
        return TransformContext(
            receptor=self.receptor, variant=self.variant, store=self._store
        )


__all__ = ["SimID", "Simulation"]
