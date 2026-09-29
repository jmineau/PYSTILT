"""A simulation, one receptor run under one variant, and its outputs."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from stilt.config import FootprintConfig, STILTParams, VariantConfig
from stilt.errors import (
    EmptyFootprintError,
    EmptyTrajectoryError,
    identify_failure_reason,
)
from stilt.footprint import Footprint
from stilt.hysplit import HYSPLITDriver
from stilt.meteorology import MetStream
from stilt.project import resolve_directory, simulation_prefix
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
    Id of one simulation, a ``(receptor, variant)`` pair.

    Its string form is ``"<receptor_id>/<variant>"``, which is also the
    simulation's directory below ``simulations/by-id/``.

    Examples
    --------
    >>> sid = SimID.parse("202307151800_-111.848_40.766_10/hrrr")
    >>> sid.variant
    'hrrr'
    >>> str(sid)
    '202307151800_-111.848_40.766_10/hrrr'
    """

    receptor: ReceptorID
    variant: str

    def __str__(self) -> str:
        return f"{self.receptor}/{self.variant}"

    def __fspath__(self) -> str:
        """Return the string form, so ``root / sim_id`` gives the simulation directory."""
        return str(self)

    @classmethod
    def parse(cls, value: str | SimID | tuple[str, str]) -> SimID:
        """
        Build a :class:`SimID` from its string form or a ``(receptor, variant)`` pair.

        Raises
        ------
        ValueError
            If a string is not of the form ``"<receptor_id>/<variant>"``, or
            the receptor id is malformed.
        """
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
    One receptor run under one variant.

    A simulation runs HYSPLIT for its receptor and calculates a footprint
    from the particles. A derived simulation (a ``from:`` variant) runs no
    HYSPLIT and calculates its footprint from its parent's particles. The
    simulation knows where its output files are and whether they all exist
    (:meth:`is_complete`).

    You rarely build one yourself. Get it from a model instead, as in
    ``model.simulations[receptor_id, "hrrr"]``.

    Parameters
    ----------
    receptor : Receptor
        Where and when particles are released.
    config : VariantConfig
        Settings of the variant. ``config.name`` is the variant name.
    met : MetStream, optional
        Meteorology for the HYSPLIT run. Required unless *parent* is given.
    parent : Simulation, optional
        Simulation whose particles this one uses. Such a simulation never
        runs HYSPLIT.
    directory : str or Path, optional
        Working directory where HYSPLIT runs and outputs are written. A
        temporary directory is used when omitted. Nothing is created until
        an output is written.
    store : Store, optional
        Project store. Outputs are copied there by :meth:`publish` and read
        from there when they are not in *directory*.

    Attributes
    ----------
    id : SimID
        ``(receptor.id, config.name)``.
    params : STILTParams
        Transport settings of the variant.
    footprint_config : FootprintConfig or None
        Footprint settings, or ``None`` for a variant without a grid.
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
        return f"Simulation(id={str(self.id)!r}, directory={str(self.directory)!r})"

    @property
    def variant(self) -> str:
        """Variant name."""
        return self.id.variant

    @property
    def is_derived(self) -> bool:
        """Whether this simulation uses another simulation's particles."""
        return self.parent is not None

    # -- Paths and keys --------------------------------------------------------

    @property
    def met_dir(self) -> Path:
        """Directory where meteorology files are staged for HYSPLIT."""
        return self.directory / "met"

    @property
    def log_path(self) -> Path:
        """Path of the HYSPLIT log in the working directory."""
        return self.directory / "stilt.log"

    @property
    def trajectories_path(self) -> Path:
        """Path of the trajectory Parquet file (the parent's for a derived simulation)."""
        if self.parent is not None:
            return self.parent.trajectories_path
        return self.directory / f"{self.id.receptor}_traj.parquet"

    @property
    def footprint_path(self) -> Path:
        """Path of the footprint NetCDF file in the working directory."""
        return self.directory / f"{self.id.receptor}_foot.nc"

    @property
    def empty_footprint_path(self) -> Path:
        """
        Path of the marker written instead of the NetCDF when the footprint is empty.

        The file holds the reason (see :attr:`empty_reason`).
        """
        return self.footprint_path.with_suffix(".empty")

    def key(self, path: str | Path) -> str:
        """Return the store key of one of this simulation's files."""
        return f"{self.key_prefix}/{Path(path).name}"

    def resolve(self, path: Path) -> Path | None:
        """Return a local path to an output file, from the working directory or the store, or ``None``."""
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
        """Whether the trajectory file exists (the parent's for a derived simulation)."""
        if self.parent is not None:
            return self.parent.has_trajectory
        return self.resolve(self.trajectories_path) is not None

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint file or the empty-footprint marker exists.

        An empty footprint (no particles over the grid) is a finished result.
        """
        return (
            self.resolve(self.footprint_path) is not None
            or self.resolve(self.empty_footprint_path) is not None
        )

    @property
    def makes_footprint(self) -> bool:
        """Whether this simulation makes a footprint, which it does when its variant has a grid."""
        return self.footprint_config is not None

    def is_complete(self) -> bool:
        """
        Return whether every expected output exists.

        That is the trajectory (unless derived) and the footprint (when the
        variant has a grid).
        """
        return (self.is_derived or self.has_trajectory) and (
            not self.makes_footprint or self.has_footprint
        )

    @property
    def outputs(self) -> list[Path]:
        """Paths of the log, footprint, and empty-marker files, and the trajectory file unless derived."""
        paths = [self.log_path, self.footprint_path, self.empty_footprint_path]
        if not self.is_derived:
            paths.append(self.trajectories_path)
        return paths

    def publish(self) -> None:
        """
        Copy this simulation's outputs from the working directory to the store.

        Does nothing without a store or when the working directory is
        already the store's copy.
        """
        if self._store is None:
            return
        for path in self.outputs:
            self._store.publish_file(path, self.key(path))

    def delete(self) -> None:
        """
        Delete this simulation's outputs from the store and its working directory.

        The simulation then runs again on the next :meth:`stilt.Model.run`.
        A derived simulation deletes only its own files, and its parent's
        trajectory stays.
        """
        if self._store is not None:
            for path in self.outputs:
                self._store.delete(self.key(path))
        shutil.rmtree(self.directory, ignore_errors=True)
        self._trajectories = None
        self._footprint = None

    def write_empty_footprint_marker(self, reason: str) -> Path:
        """Write the empty-footprint marker holding *reason* and return its path."""
        marker = self.empty_footprint_path
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(reason + "\n", encoding="utf-8")
        return marker

    def clear_empty_footprint_marker(self) -> None:
        """Remove the empty-footprint marker."""
        self.empty_footprint_path.unlink(missing_ok=True)

    @property
    def plot(self) -> SimulationPlotAccessor:
        """Plotting methods, such as ``sim.plot.map()``."""
        if self._plot is None:
            from stilt.visualization import SimulationPlotAccessor

            self._plot = SimulationPlotAccessor(self)
        return self._plot

    # -- Status ----------------------------------------------------------------

    @property
    def is_backward(self) -> bool:
        """Whether particles run backward in time (``n_hours < 0``)."""
        return self.params.n_hours < 0

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Start and end of the period the particles cover.

        Returns
        -------
        tuple of datetime
            ``(start, stop)`` with ``start < stop`` for backward and forward
            runs alike.
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
    def empty_reason(self) -> str | None:
        """
        Why the footprint is empty, or ``None`` when it is not.

        Read from the ``.empty`` marker. ``"outside_domain"`` means no
        particle reached the grid, ``"no_particles"`` that there were none,
        and ``"unknown"`` that the marker was written before reasons were
        recorded.
        """
        marker = self.resolve(self.empty_footprint_path)
        if marker is None:
            return None
        return marker.read_text(encoding="utf-8").strip() or "unknown"

    @property
    def outcome(self) -> str | None:
        """
        How this simulation ended, read from its outputs and log.

        Returns
        -------
        str or None
            ``"complete"`` if every expected output exists,
            ``"failed:<reason>"`` if a log exists but outputs are missing
            (see :class:`~stilt.errors.FailureReason`), or ``None`` if the
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
        """Return the met stream, the parent's for a derived simulation."""
        if self.met is not None:
            return self.met
        if self.parent is not None:
            return self.parent._met_stream()
        raise ValueError(f"{self.id} has no met stream.")

    @property
    def source_met_files(self) -> list[Path]:
        """Meteorology files in the archive that cover this simulation's period."""
        if not self._source_met_files:
            self._source_met_files = self._met_stream().required_files(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
            )
        return self._source_met_files

    @property
    def met_files(self) -> list[Path]:
        """
        Meteorology files staged in :attr:`met_dir` for HYSPLIT.

        Accessing this stages the files. The archive paths are
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
        Text of the HYSPLIT log.

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
        Particle trajectories, or ``None`` if they do not exist yet.

        Loaded from the Parquet file on first access. A derived simulation
        returns its parent's.
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
        """
        The footprint, or ``None`` if it does not exist yet.

        Loaded from the NetCDF file on first access, or the one from the last
        :meth:`generate_footprint` call.
        """
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
        Run HYSPLIT and keep the particles as :attr:`trajectories`.

        Parameters
        ----------
        timeout : int, optional
            Time limit for the HYSPLIT run, in seconds. Defaults to
            ``params.timeout``.
        rm_dat : bool, optional
            Delete HYSPLIT's particle output files after reading them.
            Defaults to ``params.rm_dat``.
        write : bool, default False
            Also write the trajectories to :attr:`trajectories_path`.

        Raises
        ------
        ValueError
            If the simulation is derived and so has no HYSPLIT run.
        MeteorologyError
            If the meteorology files cannot be found or staged.
        HYSPLITTimeoutError
            If HYSPLIT runs past *timeout*.
        HYSPLITFailureError
            If HYSPLIT writes a known failure message to its log.
        NoParticleOutputError
            If HYSPLIT writes no particle file.
        EmptyTrajectoryError
            If the particle file holds no particles.
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
    ) -> Footprint | None:
        """
        Calculate the footprint from the particles.

        Runs HYSPLIT first when there are no trajectories yet. The result is
        also kept as :attr:`footprint`. When no particle reaches the grid
        there is no footprint: the method returns ``None`` and, with
        ``write=True``, writes the ``.empty`` marker instead of the NetCDF.

        Parameters
        ----------
        config : FootprintConfig, optional
            Footprint settings. Defaults to the variant's own. Pass other
            settings to try them without a new variant, for example
            ``sim.footprint_config.replace(smooth_factor=0.5)``.
        write : bool, default False
            Also write the footprint to :attr:`footprint_path`, and the
            trajectories when HYSPLIT had to run.
        transforms : sequence of ParticleTransform, optional
            Extra particle transforms, applied after ``config.transforms``.
            Any object with an ``apply(particles, context)`` method works.
        context : TransformContext, optional
            Passed to every transform. Defaults to
            :meth:`transform_context`.

        Returns
        -------
        Footprint or None
            The footprint, or ``None`` when no particle reaches the grid
            (see :attr:`empty_reason`).

        Raises
        ------
        TypeError
            If the variant has no grid and no *config* is given.
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
        try:
            foot = Footprint.calculate(
                particles, receptor=traj.receptor, config=config, name=self.variant
            )
        except EmptyFootprintError as error:
            self._footprint = None
            if write:
                self.write_empty_footprint_marker(error.reason)
                self.footprint_path.unlink(missing_ok=True)
            return None
        self._footprint = foot
        if write:
            foot.to_netcdf(self.footprint_path)
            self.clear_empty_footprint_marker()
        return foot

    def transform_context(self) -> TransformContext:
        """
        Return the :class:`~stilt.TransformContext` passed to this simulation's transforms.

        It holds the receptor, the variant name, and the project store, from
        which a transform can read per-receptor inputs such as an
        averaging-kernel table. Use it to apply the footprint's transforms
        outside :meth:`generate_footprint`.
        """
        return TransformContext(
            receptor=self.receptor, variant=self.variant, store=self._store
        )


__all__ = ["SimID", "Simulation"]
