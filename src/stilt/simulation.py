"""Simulation execution and output loading for STILT runs."""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from stilt.config import ErrorParams, FootprintConfig, STILTParams
from stilt.config.model import _config_or_kwargs
from stilt.errors import (
    EmptyTrajectoryError,
    identify_failure_reason,
)
from stilt.footprint import Footprint
from stilt.hysplit import HYSPLITDriver
from stilt.meteorology import MetID, MetStream
from stilt.project import (
    SIMULATION_LOG_FILENAME,
    SIMULATION_MET_DIRNAME,
    resolve_directory,
    simulation_prefix,
)
from stilt.receptors import LocationID, Receptor, ReceptorID
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

_ERROR_PARAM_FIELDS = frozenset(ErrorParams.XYERR_PARAMS) | frozenset(
    ErrorParams.ZIERR_PARAMS
)

TRAJECTORY = "trajectory"
ERROR_TRAJECTORY = "error_trajectory"


def _read_trajectory_params(path: Path) -> STILTParams | None:
    """Read just the stored params from a trajectory parquet's Arrow metadata."""
    import json

    import pyarrow.parquet as pq

    try:
        meta = pq.ParquetFile(path).schema_arrow.metadata
    except Exception:
        return None
    if not meta or b"stilt:params" not in meta:
        return None
    try:
        return STILTParams.model_validate(json.loads(meta[b"stilt:params"]))
    except Exception:
        return None


def _params_match_ignoring_error(a: STILTParams, b: STILTParams) -> bool:
    """Return whether two param sets agree on everything but the error params."""
    da = a.model_dump()
    db = b.model_dump()
    for field in _ERROR_PARAM_FIELDS:
        da.pop(field, None)
        db.pop(field, None)
    return da == db


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

    met: MetID
    receptor: ReceptorID
    time: dt.datetime
    location: LocationID

    def __new__(cls, id_str: str) -> SimID:
        """Create from canonical ``'{met}_{YYYYMMDDHHMM}_{location_id}'`` string."""
        if not re.fullmatch(r"[a-z0-9]+_\d{12}_.+", id_str):
            raise ValueError(
                f"Invalid sim_id format: {id_str!r}. "
                "Expected '{met}_{YYYYMMDDHHMM}_{location_id}'."
            )
        instance = super().__new__(cls, id_str)
        met_str, recep_str = id_str.split("_", 1)
        receptor_id = ReceptorID(recep_str)

        instance.met = MetID(met_str)
        instance.receptor = receptor_id
        instance.time = receptor_id.time
        instance.location = receptor_id.location
        return instance

    @classmethod
    def from_parts(
        cls,
        met: MetID | str,
        receptor: Receptor,
    ) -> SimID:
        """
        Build a :class:`SimID` from constituent parts.

        Parameters
        ----------
        met : MetID | str
            ID of the meteorology configuration (e.g. ``'hrrr'``).
            Cannot contain underscores.
        receptor : Receptor
            Source receptor; provides the release time and location ID.

        Returns
        -------
        SimID
        """
        return cls(f"{met}_{receptor.id}")

    def __fspath__(self) -> str:
        """Allow Path joins like ``base / sim_id`` without manual str() conversion."""
        return str(self)


class Simulation:
    """
    Container for running and reading one STILT simulation.

    A simulation owns its output filenames, their store keys, and the single
    definition of which outputs exist and whether the simulation is complete.

    Parameters
    ----------
    meteorology, receptor, params
        What to run.
    directory
        Compute-local working directory. Its basename must be the simulation
        id. A temporary directory is created when omitted. Nothing is created
        on disk until an output is written.
    exe_dir
        Directory holding a custom ``hycs_std`` build.
    store
        Output store the outputs are published to and read back from when
        they are not on local disk. When the store's location for this
        simulation *is* ``directory``, publishing is a no-op.
    """

    def __init__(
        self,
        meteorology: MetStream,
        receptor: Receptor,
        params: STILTParams,
        directory: str | Path | None = None,
        exe_dir: Path | None = None,
        store: Store | None = None,
    ):
        if directory is None:
            scratch_dir = resolve_directory(prefix="pystilt_")
            directory = scratch_dir / SimID.from_parts(meteorology.id, receptor)
        self.directory = resolve_directory(directory)
        self.meteorology = meteorology
        self.receptor = receptor
        self.params = params
        self._exe_dir = exe_dir
        self._store = store

        # The sim ID is derived from the directory name so a Model can lay out
        # `{project}/simulations/by-id/{sim_id}` and ad-hoc runs still work.
        self.id = SimID(self.directory.name)
        self.key_prefix = simulation_prefix(str(self.id))

        # Lazy state
        self._source_met_files: list[Path] | None = None
        self._met_files: list[Path] | None = None
        self._trajectories = None
        self._error_trajectories = None
        self._footprints: dict[str, Footprint] = {}
        self._plot: SimulationPlotAccessor | None = None

    def __repr__(self) -> str:
        """Compact developer-facing simulation representation."""
        return f"Simulation(id={self.id!r}, directory={str(self.directory)!r})"

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
        """Compute-local trajectory parquet path."""
        return self.directory / f"{self.id}_traj.parquet"

    @property
    def error_trajectories_path(self) -> Path:
        """Compute-local error-trajectory parquet path."""
        return self.directory / f"{self.id}_error.parquet"

    def footprint_path(self, name: str = "") -> Path:
        """Compute-local footprint netCDF path for one footprint name."""
        suffix = f"_{name}" if name else ""
        return self.directory / f"{self.id}{suffix}_foot.nc"

    def empty_footprint_path(self, name: str = "") -> Path:
        """Compute-local marker path recording that a footprint is legitimately empty."""
        return self.footprint_path(name).with_suffix(".empty")

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
        """Whether the main trajectory parquet exists on disk or in the store."""
        return self.resolve(self.trajectories_path) is not None

    @property
    def has_error_trajectory(self) -> bool:
        """Whether the error-trajectory parquet exists on disk or in the store."""
        return self.resolve(self.error_trajectories_path) is not None

    def has_footprint(self, name: str) -> bool:
        """
        Whether one named footprint is complete.

        The empty marker counts: a run that legitimately produced no footprint
        is a terminal outcome, not missing work.
        """
        return (
            self.resolve(self.footprint_path(name)) is not None
            or self.resolve(self.empty_footprint_path(name)) is not None
        )

    def missing_footprints(self, names: Iterable[str]) -> list[str]:
        """Return the footprint names among *names* that are not yet complete."""
        return [name for name in names if not self.has_footprint(name)]

    def expected_outputs(self, footprints: Iterable[str] = ()) -> tuple[str, ...]:
        """
        Return the outputs this simulation must produce to be complete.

        Always the trajectory, plus the error trajectory when wind-error
        params are set, plus one entry per footprint name. Error *footprints*
        are never required.
        """
        outputs = [TRAJECTORY]
        if self.params.error_enabled:
            outputs.append(ERROR_TRAJECTORY)
        outputs.extend(footprints)
        return tuple(outputs)

    def has_output(self, output: str) -> bool:
        """Whether one named output (trajectory, error_trajectory, or footprint) exists."""
        if output == TRAJECTORY:
            return self.has_trajectory
        if output == ERROR_TRAJECTORY:
            return self.has_error_trajectory
        return self.has_footprint(output)

    def is_complete(self, footprints: Iterable[str] = ()) -> bool:
        """
        Whether every expected output exists.

        Checks the trajectory first so the common incomplete case costs one
        existence check.
        """
        return all(self.has_output(o) for o in self.expected_outputs(footprints))

    def publish(self) -> None:
        """
        Copy this simulation's local outputs into the store.

        A no-op when there is no store or when the store's location for this
        simulation is already ``directory``.
        """
        if self._store is None:
            return
        for path in (
            self.log_path,
            self.trajectories_path,
            self.error_trajectories_path,
        ):
            self._store.publish_file(path, self.key(path))
        if self.directory.exists():
            for path in sorted(self.directory.glob(f"{self.id}*_foot.*")):
                self._store.publish_file(path, self.key(path))

    def write_empty_footprint_marker(self, name: str = "") -> Path:
        """Create the empty-footprint marker for one named footprint."""
        marker = self.empty_footprint_path(name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
        return marker

    def clear_empty_footprint_marker(self, name: str = "") -> None:
        """Remove the empty-footprint marker for one named footprint."""
        self.empty_footprint_path(name).unlink(missing_ok=True)

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
    def status(self) -> str | None:
        """
        Current status of this simulation.

        Returns
        -------
        str or None
            ``'complete'`` if the trajectory parquet exists, a
            ``'failed:<reason>'`` string if HYSPLIT failed, or ``None`` if the
            simulation has not run.
        """
        if self.has_trajectory:
            return "complete"
        log_path = self.resolve(self.log_path)
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
        """
        Compute-local meteorology files staged for HYSPLIT execution.

        The output/source archive paths remain available via
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
        """
        Contents of the HYSPLIT stdout log file.

        Returns
        -------
        str

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self.resolve(self.log_path)
        if log_path is None:
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        return log_path.read_text()

    # -- Footprints ------------------------------------------------------------

    def get_footprint(self, name: str) -> Footprint | None:
        """
        Return a named footprint, loading from disk if not already cached.

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
            path = self.resolve(self.footprint_path(name))
            if path is None:
                return None
            self._footprints[name] = Footprint.from_netcdf(path)
        return self._footprints[name]

    # -- Execution -------------------------------------------------------------

    def _can_reuse_main_for_error(self) -> bool:
        """
        Whether to run only the error pass, reusing an existing main trajectory.

        True when an error trajectory is configured (``winderrtf > 0``) but not
        yet present, the main trajectory already exists, and its stored params
        match the current config on everything but the error params. Lets an
        error-trajectory backfill skip recomputing the (expensive) main run.
        """
        if self.params.winderrtf <= 0:
            return False
        if self.has_error_trajectory:
            return False
        main_path = self.resolve(self.trajectories_path)
        if main_path is None:
            return False
        stored = _read_trajectory_params(main_path)
        return stored is not None and _params_match_ignoring_error(stored, self.params)

    def run_trajectories(
        self,
        timeout: int | None = None,
        rm_dat: bool | None = None,
        write: bool = False,
    ) -> None:
        """
        Run HYSPLIT, populating ``self.trajectories`` and ``self.error_trajectories``.

        Parameters
        ----------
        timeout : int, optional
            Wall-clock cap in seconds for each hycs_std run. Defaults to
            ``params.timeout``, so a project can cap wedged HYSPLIT processes from
            ``config.yaml`` without every caller passing it.
        rm_dat : bool, optional
            Defaults to ``params.rm_dat``.
        write : bool
            If True, persist trajectories (and error trajectories if present) to
            ``self.trajectories_path`` / ``self.error_trajectories_path``.

        Raises
        ------
        HYSPLITTimeoutError, HYSPLITFailureError, NoParticleOutputError,
        EmptyTrajectoryError
        """
        if rm_dat is None:
            rm_dat = self.params.rm_dat
        if timeout is None:
            timeout = getattr(self.params, "timeout", None)

        self.directory.mkdir(parents=True, exist_ok=True)

        # If the main trajectory already exists with matching (non-error) params
        # and only the error trajectory is needed, run the error pass alone — the
        # error run is independent of the main, so there's no need to recompute it.
        error_only = self._can_reuse_main_for_error()

        runner = HYSPLITDriver(
            directory=self.directory,
            receptor=self.receptor,
            params=self.params,
            met_files=self.met_files,
            exe_dir=self._exe_dir,
        )
        runner.prepare()
        result = runner.execute(timeout=timeout, rm_dat=rm_dat, error_only=error_only)

        result_log = getattr(result, "log_path", None)
        if result_log is not None:
            result_log_path = Path(result_log)
            if result_log_path != self.log_path and result_log_path.exists():
                self.log_path.write_text(result_log_path.read_text())
        elif hasattr(result, "stdout"):
            self.log_path.write_text(str(cast(Any, result).stdout))

        if not error_only:
            if result.particles is None or result.particles.empty:
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
        context: TransformContext | None = None,
        **kwargs,
    ) -> Footprint:
        """
        Compute a named footprint and store it in the footprint cache.

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
            Extra particle transforms applied after ``config.transforms`` and
            before rasterization (any object with ``apply(particles, context)``).
        context : TransformContext, optional
            Context handed to every transform. Defaults to one built from the
            receptor, footprint name, and project store.
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
            # timeout=None picks up params.timeout inside run_trajectories.
            self.run_trajectories(write=write)
            traj = self.error_trajectories if error else self.trajectories

        stored_name = f"{name}_error" if error else name
        if traj is None:
            particles = (
                self.trajectories.data.head(0).copy()
                if self.trajectories is not None
                else pd.DataFrame(
                    {
                        "time": pd.Series(dtype="float64"),
                        "indx": pd.Series(dtype="int64"),
                        "long": pd.Series(dtype="float64"),
                        "lati": pd.Series(dtype="float64"),
                        "foot": pd.Series(dtype="float64"),
                    }
                )
            )
        else:
            particles = traj.data
        all_transforms = [*config.transforms, *(transforms or [])]
        if all_transforms:
            particles = apply_transforms(
                particles,
                all_transforms,
                context or self.transform_context(stored_name, error=error),
            )
        foot = Footprint.calculate(
            particles,
            receptor=self.receptor if traj is None else traj.receptor,
            config=config,
            name=stored_name,
        )
        self._footprints[stored_name] = foot
        if write:
            foot.to_netcdf(self.footprint_path(stored_name))
        return foot

    def transform_context(
        self, name: str = "", error: bool = False
    ) -> TransformContext:
        """
        The :class:`~stilt.TransformContext` this simulation hands its transforms.

        Carries the receptor, the footprint *name*, whether the particles are
        the error trajectories, and the project store (so a transform can read
        per-receptor inputs such as an averaging-kernel table). Use it to apply
        a footprint's transforms outside :meth:`generate_footprint`.
        """
        return TransformContext(
            receptor=self.receptor,
            footprint_name=name,
            is_error=error,
            store=self._store,
        )

    # -- Lazy trajectory loading -----------------------------------------------

    @property
    def trajectories(self) -> Trajectories | None:
        """
        Main particle trajectories, loaded from parquet on first access.

        Returns ``None`` if no trajectory parquet exists and the simulation
        has not been run in this process.

        Returns
        -------
        Trajectories or None
        """
        if not self._trajectories:
            traj_path = self.resolve(self.trajectories_path)
            if traj_path is not None:
                self._trajectories = Trajectories.from_parquet(traj_path)
        return self._trajectories

    @property
    def error_trajectories(self) -> Trajectories | None:
        """
        Error-trajectory particles, loaded from parquet on first access.

        Returns ``None`` if no error parquet exists.

        Returns
        -------
        Trajectories or None
        """
        if not self._error_trajectories:
            error_path = self.resolve(self.error_trajectories_path)
            if error_path is not None:
                self._error_trajectories = Trajectories.from_parquet(error_path)
        return self._error_trajectories


__all__ = ["ERROR_TRAJECTORY", "TRAJECTORY", "SimID", "Simulation"]
