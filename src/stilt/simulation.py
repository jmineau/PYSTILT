"""Simulation execution and artifact loading for STILT runs."""

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from stilt.artifacts import (
    ArtifactStore,
    error_trajectory_path,
    footprint_path,
    resolve_directory,
    simulation_artifact_key,
    simulation_log_path,
    simulation_met_path,
    trajectory_path,
)
from stilt.config import FootprintConfig, STILTParams, _config_or_kwargs
from stilt.errors import (
    EmptyTrajectoryError,
    identify_failure_reason,
)
from stilt.footprint import Footprint
from stilt.hysplit import HYSPLITRunner
from stilt.meteorology import MetStream
from stilt.receptor import Receptor
from stilt.trajectory import Trajectories
from stilt.transforms import (
    ParticleTransform,
    ParticleTransformContext,
    apply_particle_transforms,
    build_particle_transforms,
)

if TYPE_CHECKING:
    from stilt.visualization import SimulationPlotAccessor

logger = logging.getLogger(__name__)


class SimID(str):
    """
    Structured representation of a PYSTILT simulation ID.

    Format: ``{met}_{YYYYMMDDHHMM}_{location_id}``

    Behaves as a plain string — dict keys, path joins, and comparisons all
    work without ``str()`` conversion.  Attributes ``met``, ``time``,
    and ``location_id`` are parsed from the string on construction.

    Create via the canonical string::

        SimID("hrrr_202301011200_-111.85_40.77_5")

    Or from parts::

        SimID.from_parts(met="hrrr", receptor=r)
    """

    met: str
    time: pd.Timestamp
    location_id: str

    def __new__(cls, sim_id_str: str) -> "SimID":
        """Create from canonical ``'{met}_{YYYYMMDDHHMM}_{location_id}'`` string."""
        instance = super().__new__(cls, sim_id_str)
        try:
            i = sim_id_str.index("_")
        except ValueError:
            raise ValueError(f"Invalid sim_id format: {sim_id_str!r}") from None
        met = sim_id_str[:i]
        rest = sim_id_str[i + 1 :]
        if len(rest) < 13 or rest[12] != "_":
            raise ValueError(f"Invalid sim_id format: {sim_id_str!r}")
        time_str = rest[:12]
        location_id = rest[13:]
        try:
            time = pd.to_datetime(time_str, format="%Y%m%d%H%M")
        except ValueError:
            raise ValueError(
                f"Cannot parse timestamp from sim_id: {sim_id_str!r}"
            ) from None
        instance.met = met
        instance.time = time
        instance.location_id = location_id
        return instance

    @classmethod
    def from_parts(
        cls,
        met: str,
        receptor: "Receptor | None" = None,
        time: "pd.Timestamp | dt.datetime | str | None" = None,
        location_id: "str | None" = None,
    ) -> "SimID":
        """Build a :class:`SimID` from constituent parts.

        Parameters
        ----------
        met : str
            Name of the meteorology configuration (e.g. ``'hrrr'``).
        receptor : Receptor, optional
            Source receptor; provides the release time and location ID.
        time : pd.Timestamp or str, optional
            Release time; overrides ``receptor.time`` when provided.
        location_id : str, optional
            Location ID; overrides ``receptor.location_id`` when provided.

        Returns
        -------
        SimID
        """
        if receptor is not None:
            if time is None:
                time = receptor.time
            if location_id is None:
                location_id = receptor.location_id
        if not met:
            raise ValueError("'met' is required")
        if time is None or location_id is None:
            raise ValueError("Must provide 'receptor' or both 'time' and 'location_id'")
        ts = pd.Timestamp(time)
        return cls(f"{met}_{ts.strftime('%Y%m%d%H%M')}_{location_id}")

    def __fspath__(self) -> str:
        """Allow Path joins like ``base / sim_id`` without manual str() conversion."""
        return str(self)


class Simulation:
    """Container for running and reading a STILT simulation."""

    def __init__(
        self,
        meteorology: MetStream,
        receptor: Receptor,
        params: STILTParams,
        directory: str | Path | None = None,
        exe_dir: Path | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        if directory is None:
            scratch_dir = resolve_directory(prefix="pystilt_")
            directory = scratch_dir / SimID.from_parts(meteorology.name, receptor)
        self.directory = resolve_directory(directory)
        self.meteorology = meteorology
        self.receptor = receptor
        self.params = params
        self._exe_dir = exe_dir
        self._artifact_store = artifact_store

        # The sim ID is derived from the directory name
        # This dientangles the sim ID from the receptor and met,
        # allowing users to rename sim directories without breaking functionality.
        # The Model object in coordination with the service workers
        # then assigns the met_YYYYMMDDHHMM_location_id format to the sim directory name.
        self.id = SimID(self.directory.name)

        self.directory.mkdir(parents=True, exist_ok=True)

        # Lazy state
        self._source_met_files: list[Path] | None = None
        self._met_files: list[Path] | None = None
        self._trajectories = None
        self._error_trajectories = None
        self._footprints: dict[str, Footprint] = {}
        self._plot: SimulationPlotAccessor | None = None

    @property
    def met_dir(self) -> Path:
        """Compute-local meteorology staging directory."""
        return simulation_met_path(self.directory, self.id)

    @property
    def log_path(self) -> Path:
        """Compute-local HYSPLIT log path."""
        return simulation_log_path(self.directory, self.id)

    @property
    def trajectories_path(self) -> Path:
        """Compute-local trajectory parquet path."""
        return trajectory_path(self.directory, self.id)

    @property
    def error_trajectories_path(self) -> Path:
        """Compute-local error-trajectory parquet path."""
        return error_trajectory_path(self.directory, self.id)

    def footprint_path(self, name: str = "") -> Path:
        """Compute-local footprint path for one footprint name."""
        return footprint_path(self.directory, name, sim_id=self.id)

    def _artifact_key(self, path: Path) -> str:
        """Return the canonical storage key for a simulation artifact path."""
        return simulation_artifact_key(str(self.id), path.name)

    def _artifact_path(self, path: Path) -> Path | None:
        """Return a local path to an artifact from disk or the artifact store."""
        if path.exists():
            return path
        key = self._artifact_key(path)
        if self._artifact_store is not None and self._artifact_store.exists(key):
            return self._artifact_store.local_path(key)
        return None

    @property
    def plot(self) -> "SimulationPlotAccessor":
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
        """Start and stop datetimes spanned by this simulation.

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
            stop = r_time + dt.timedelta(hours=self.params.n_hours + 1)
        return start, stop

    @property
    def status(self) -> str | None:
        """Current status of this simulation directory.

        Returns
        -------
        str or None
            ``'complete'`` if the trajectory parquet exists, a
            ``'failed:<reason>'`` string if HYSPLIT failed, or ``None`` if the
            simulation directory does not yet exist.
        """
        if self._artifact_path(self.trajectories_path) is not None:
            return "complete"
        log_path = self._artifact_path(self.log_path)
        if log_path is None:
            return None
        return f"failed:{identify_failure_reason(log_path.parent)}"

    # -- Lazy accessors --------------------------------------------------------

    @property
    def source_met_files(self) -> list[Path]:
        """Archive/source met files required for this simulation's time window."""
        if not self._source_met_files:
            self._source_met_files = self.meteorology.required_files(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
            )
        return self._source_met_files

    @property
    def met_files(self) -> list[Path]:
        """Compute-local meteorology files staged for HYSPLIT execution.

        The durable/source archive paths remain available via
        :attr:`source_met_files`.

        Returns
        -------
        list[Path]
        """
        if not self._met_files:
            self._met_files = self.meteorology.stage_files_for_simulation(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
                target_dir=self.met_dir,
            )
        return self._met_files

    @property
    def log(self) -> str:
        """Contents of the HYSPLIT stdout log file.

        Returns
        -------
        str

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self._artifact_path(self.log_path)
        if log_path is None:
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        return log_path.read_text()

    # -- Footprints ------------------------------------------------------------

    def get_footprint(self, name: str) -> "Footprint | None":
        """Return a named footprint, loading from disk if not already cached.

        Parameters
        ----------
        name : str
            Footprint name (matches the key used in :attr:`foot_configs`).

        Returns
        -------
        Footprint or None
            The footprint if the file exists on disk, otherwise ``None``.
        """
        if name not in self._footprints:
            path = self._artifact_path(self.footprint_path(name))
            if path is None:
                return None
            self._footprints[name] = Footprint.from_netcdf(path)
        return self._footprints[name]

    # -- Execution -------------------------------------------------------------

    def run_trajectories(
        self,
        timeout: int | None = None,
        rm_dat: bool | None = None,
        write: bool = False,
    ) -> None:
        """Run HYSPLIT, populating ``self.trajectories`` and ``self.error_trajectories``.

        Parameters
        ----------
        write : bool
            If True, persist trajectories (and error trajectories if present) to
            ``self.traj_path`` / ``self.error_path``.

        Raises
        ------
        HYSPLITTimeoutError, HYSPLITFailureError, NoParticleOutputError,
        EmptyTrajectoryError
        """
        if rm_dat is None:
            rm_dat = self.params.rm_dat

        runner = HYSPLITRunner(
            directory=self.directory,
            receptor=self.receptor,
            params=self.params,
            met_files=self.met_files,
            exe_dir=self._exe_dir,
        )
        runner.prepare()
        result = runner.execute(timeout=timeout, rm_dat=rm_dat)

        self.log_path.write_text(result.stdout)

        if result.particles.empty:
            raise EmptyTrajectoryError(f"No trajectory data for {self.id}")

        self._trajectories = Trajectories.from_particles(
            result.particles,
            receptor=self.receptor,
            params=self.params,
            met_files=self.source_met_files,
        )

        if result.error_particles is not None and not result.error_particles.empty:
            self._error_trajectories = Trajectories.from_particles(
                result.error_particles,
                receptor=self.receptor,
                params=self.params,
                met_files=self.source_met_files,
                is_error=True,
            )

        if write:
            if self._trajectories is not None:
                self._trajectories.to_parquet(self.trajectories_path)
            if self._error_trajectories is not None:
                self._error_trajectories.to_parquet(self.error_trajectories_path)

    def generate_footprint(
        self,
        name: str,
        config: FootprintConfig | None = None,
        write: bool = False,
        error: bool = False,
        transforms: Sequence[ParticleTransform] | None = None,
        transform_context: ParticleTransformContext | None = None,
        **kwargs,
    ) -> "Footprint | None":
        """Compute a named footprint and store it in the footprint cache.

        Parameters
        ----------
        name : str
            Base label for this footprint (e.g. ``"slv"``).  When
            *error* is True, ``"_error"`` is appended automatically so
            the footprint is stored as ``"slv_error"``.
        config : FootprintConfig, optional
            Footprint configuration.  Mutually exclusive with ``**kwargs``.
        write : bool
            If True, write the footprint netCDF to the sim directory.
        error : bool
            If True, compute from the error trajectory instead of the main
            trajectory and store under ``"{name}_error"``.
        transforms : sequence, optional
            Additional typed particle transforms applied after any declarative
            ``config.transforms`` and before footprint rasterization.
        transform_context : ParticleTransformContext, optional
            Runtime context supplied to particle transforms. If omitted, a
            default context is built from the receptor, footprint name, and
            footprint config.
        **kwargs
            Forwarded to ``FootprintConfig`` when *config* is not given.
        """
        config = _config_or_kwargs(config, kwargs, FootprintConfig)
        if config is None:
            raise TypeError(
                "Must provide 'config' or keyword arguments for FootprintConfig."
            )

        traj = self.error_trajectories if error else self.trajectories

        if traj is None:
            # Auto-run, threading write so callers with write=False stay in-memory.
            self.run_trajectories(write=write)
            traj = self.error_trajectories if error else self.trajectories

        if traj is None:
            # Genuinely empty (e.g. no error particles produced).
            return None

        stored_name = f"{name}_error" if error else name
        particles = traj.data
        resolved_transforms = build_particle_transforms(config.transforms)
        if transforms is not None:
            resolved_transforms.extend(transforms)
        if resolved_transforms:
            context = transform_context or ParticleTransformContext(
                receptor=traj.receptor,
                footprint_name=stored_name,
                footprint_config=config,
                is_error=error,
            )
            particles = apply_particle_transforms(
                particles,
                resolved_transforms,
                context=context,
            )
        foot = Footprint.calculate(
            particles, receptor=traj.receptor, config=config, name=stored_name
        )
        if foot is not None:
            self._footprints[stored_name] = foot
            if write:
                foot.to_netcdf(self.footprint_path(stored_name))
        return foot

    # -- Lazy trajectory loading -----------------------------------------------

    @property
    def trajectories(self) -> Trajectories | None:
        """Main particle trajectories, loaded from parquet on first access.

        Returns ``None`` if no trajectory parquet exists and the simulation
        has not been run in this process.

        Returns
        -------
        Trajectories or None
        """
        if not self._trajectories:
            traj_path = self._artifact_path(self.trajectories_path)
            if traj_path is not None:
                self._trajectories = Trajectories.from_parquet(traj_path)
            else:
                logger.info(
                    f"No trajectories for {self.id} - has the simulation been run?"
                )
        return self._trajectories

    @property
    def error_trajectories(self) -> Trajectories | None:
        """Error-trajectory particles, loaded from parquet on first access.

        Returns ``None`` if no error parquet exists.

        Returns
        -------
        Trajectories or None
        """
        if not self._error_trajectories:
            error_path = self._artifact_path(self.error_trajectories_path)
            if error_path is not None:
                self._error_trajectories = Trajectories.from_parquet(error_path)
        return self._error_trajectories
