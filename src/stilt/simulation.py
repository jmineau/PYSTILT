"""A simulation, one receptor run under one variant, and its results."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from stilt.config import FootprintConfig, STILTParams, TransportSettings, VariantConfig
from stilt.errors import (
    EmptyFootprintError,
    EmptyTrajectoryError,
    identify_failure_reason,
)
from stilt.footprint import Footprint
from stilt.hysplit import HYSPLITDriver
from stilt.meteorology import MetStream
from stilt.output import Footprints, Output, Run
from stilt.project import resolve_directory
from stilt.receptors import Receptor, ReceptorID
from stilt.trajectory import Trajectories
from stilt.transforms import (
    ParticleTransform,
    TransformContext,
)

if TYPE_CHECKING:
    from stilt.visualization import SimulationPlotAccessor

logger = logging.getLogger(__name__)


class SimID(NamedTuple):
    """
    Id of one simulation, a ``(receptor, variant)`` pair.

    Its string form is ``"<receptor_id>/<variant>"``.

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
        """Return the string form, so ``root / sim_id`` gives a working directory."""
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


class VariantOutput:
    """
    Where one variant's results live in an output directory.

    Found once and shared by every simulation of the variant, so a run
    created by one receptor's worker is seen by the others. Nothing is
    created until :meth:`ensure_run` or :meth:`ensure_footprints` is called.

    Parameters
    ----------
    output : Output
        The output directory.
    name : str
        Variant name, which labels the folders.
    settings : TransportSettings
        What identifies the variant's run.
    footprint_config : FootprintConfig or None
        The variant's footprint settings, or ``None`` for particles only.
    """

    def __init__(
        self,
        output: Output,
        name: str,
        settings: TransportSettings,
        footprint_config: FootprintConfig | None,
    ) -> None:
        self.output = output
        self.name = name
        self.settings = settings
        self.footprint_config = footprint_config
        self._run: Run | None = None
        self._footprints: Footprints | None = None

    def __repr__(self) -> str:
        return f"VariantOutput({self.name!r}, {self.output.path.name!r})"

    @property
    def run(self) -> Run | None:
        """The run holding the variant's particles, or ``None`` until one exists."""
        if self._run is None:
            self._run = self.output.find_run(self.settings)
        return self._run

    @property
    def footprints(self) -> Footprints | None:
        """The folder holding the variant's footprints, or ``None`` until one exists or without a grid."""
        if self.footprint_config is None:
            return None
        if self._footprints is None:
            run = self.run
            if run is not None:
                self._footprints = run.find_footprints(self.footprint_config)
        return self._footprints

    def ensure_run(self) -> Run:
        """Return the run, creating its folder on first use."""
        if self._run is None:
            self._run = self.output.run(self.name, self.settings)
        return self._run

    def ensure_footprints(self) -> Footprints | None:
        """Return the footprint folder, creating it on first use, or ``None`` without a grid."""
        if self.footprint_config is None:
            return None
        if self._footprints is None:
            self._footprints = self.ensure_run().footprints(
                self.footprint_config, name=self.name
            )
        return self._footprints


class Simulation:
    """
    One receptor run under one variant.

    A simulation runs HYSPLIT for its receptor and calculates a footprint
    from the particles, and reads both back from the output directory. Its
    particles are shared with every variant that has the same transport
    settings. The simulation knows whether its results exist
    (:meth:`is_complete`).

    You rarely build one yourself. Get it from a model instead, as in
    ``model.simulations[receptor_id, "hrrr"]``.

    Parameters
    ----------
    receptor : Receptor
        Where and when particles are released.
    config : VariantConfig
        Settings of the variant. ``config.name`` is the variant name.
    met : MetStream
        Meteorology for the HYSPLIT run.
    outputs : VariantOutput
        Where the variant's results live.
    directory : str or Path, optional
        Working directory where HYSPLIT runs, on scratch. A temporary
        directory is used when omitted. Nothing is created until HYSPLIT
        runs, and it is removed afterwards unless *keep_scratch* is set or
        the run fails, in which case it is copied into the output directory.
    project_dir : Path, optional
        Directory that relative file names in transform settings are taken
        from (the project directory).
    keep_scratch : bool, default False
        Keep the working directory of a successful run too, under the output
        directory's ``scratch/``.

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
        met: MetStream,
        outputs: VariantOutput,
        directory: str | Path | None = None,
        project_dir: Path | None = None,
        keep_scratch: bool = False,
    ):
        self.id = SimID(receptor.id, config.name)
        self.receptor = receptor
        self.config = config
        self.met = met
        self.outputs = outputs
        self.params: STILTParams = config.stilt_params()
        self.footprint_config: FootprintConfig | None = outputs.footprint_config
        if directory is None:
            directory = resolve_directory(prefix="pystilt_") / self.id
        self.directory = resolve_directory(directory)
        self.project_dir = project_dir
        self.keep_scratch = keep_scratch

        # Lazy state
        self._source_met_files: list[Path] | None = None
        self._met_files: list[Path] | None = None
        self._trajectories: Trajectories | None = None
        self._footprint: Footprint | None = None
        self._plot: SimulationPlotAccessor | None = None

    def __repr__(self) -> str:
        return f"Simulation(id={str(self.id)!r})"

    @property
    def variant(self) -> str:
        """Variant name."""
        return self.id.variant

    @property
    def receptor_id(self) -> str:
        return str(self.id.receptor)

    # -- Where the results are -------------------------------------------------

    @property
    def run(self) -> Run | None:
        """The run holding this simulation's particles, or ``None`` until one exists."""
        return self.outputs.run

    @property
    def footprints(self) -> Footprints | None:
        """The folder holding this variant's footprints, or ``None`` until one exists."""
        return self.outputs.footprints

    @property
    def trajectories_path(self) -> Path | None:
        """Path of the particle file in the output directory, or ``None`` before the run exists."""
        run = self.run
        return None if run is None else run.particles_path(self.receptor_id)

    @property
    def footprint_path(self) -> Path | None:
        """Path of the footprint file in the output directory, or ``None`` before its folder exists."""
        feet = self.footprints
        return None if feet is None else feet.footprint_path(self.receptor_id)

    @property
    def log_path(self) -> Path | None:
        """Path of the run log in the output directory, or ``None`` before the run exists."""
        run = self.run
        return None if run is None else run.log_path(self.receptor_id)

    @property
    def met_dir(self) -> Path:
        """Directory where meteorology files are staged for HYSPLIT, inside the working directory."""
        return self.directory / "met"

    # -- Presence and completion -----------------------------------------------

    @property
    def has_trajectory(self) -> bool:
        """Whether the particle file exists."""
        run = self.run
        return run is not None and run.has_particles(self.receptor_id)

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint file exists.

        An empty footprint (no particles over the grid) is a finished result.
        """
        feet = self.footprints
        return feet is not None and feet.has(self.receptor_id)

    @property
    def makes_footprint(self) -> bool:
        """Whether this simulation makes a footprint, which it does when its variant has a grid."""
        return self.footprint_config is not None

    def is_complete(self) -> bool:
        """
        Return whether every expected result exists.

        That is the particles, and the footprint when the variant has a grid.
        """
        return self.has_trajectory and (not self.makes_footprint or self.has_footprint)

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
        Why the footprint is empty, or ``None`` when it is not, or does not exist.

        ``"outside_domain"`` means no particle reached the grid and
        ``"no_particles"`` that there were none.
        """
        feet = self.footprints
        if feet is None or not feet.has(self.receptor_id):
            return None
        return feet.empty_reason(self.receptor_id)

    @property
    def outcome(self) -> str | None:
        """
        How this simulation ended, read from its results and log.

        Returns
        -------
        str or None
            ``"complete"`` if every expected result exists,
            ``"failed:<reason>"`` if a log exists but results are missing
            (see :class:`~stilt.errors.FailureReason`), or ``None`` if the
            simulation has not run.
        """
        if self.is_complete():
            return "complete"
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            return None
        return f"failed:{identify_failure_reason(log_path)}"

    # -- Lazy accessors --------------------------------------------------------

    @property
    def source_met_files(self) -> list[Path]:
        """Meteorology files in the archive that cover this simulation's period."""
        if not self._source_met_files:
            self._source_met_files = self.met.required_files(
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
            self._met_files = self.met.stage_files_for_simulation(
                r_time=self.receptor.time,
                n_hours=self.params.n_hours,
                target_dir=self.met_dir,
            )
        return self._met_files

    @property
    def log(self) -> str:
        """
        Text of the run log.

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            raise FileNotFoundError(f"No log for {self.id} yet.")
        return log_path.read_text()

    @property
    def trajectories(self) -> Trajectories | None:
        """
        Particle trajectories, or ``None`` if they do not exist yet.

        Read from the output directory on first access, or kept from the last
        :meth:`run_trajectories` call.
        """
        if self._trajectories is None and self.has_trajectory:
            assert self.run is not None
            self._trajectories = self.run.read_particles(self.receptor_id)
        return self._trajectories

    @property
    def footprint(self) -> Footprint | None:
        """
        The footprint, or ``None`` if it does not exist yet or is empty.

        Read from the output directory on first access, or kept from the last
        :meth:`generate_footprint` call with the variant's own settings.
        """
        if self._footprint is None and self.has_footprint:
            assert self.footprints is not None
            self._footprint = self.footprints.read(self.receptor_id)
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

        HYSPLIT runs in :attr:`directory`. The log is copied into the output
        directory whether the run succeeds or fails. The working directory is
        then removed, unless the run failed or ``keep_scratch`` is set, in
        which case it is copied under the output directory's ``scratch/``
        first.

        Parameters
        ----------
        timeout : int, optional
            Time limit for the HYSPLIT run, in seconds. Defaults to
            ``params.timeout``.
        rm_dat : bool, optional
            Delete HYSPLIT's particle output files after reading them.
            Defaults to ``params.rm_dat``.
        write : bool, default False
            Also write the particles to the output directory.

        Raises
        ------
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
        if rm_dat is None:
            rm_dat = self.params.rm_dat
        if timeout is None:
            timeout = self.params.timeout

        run = self.outputs.ensure_run()
        self.directory.mkdir(parents=True, exist_ok=True)
        scratch_log = self.directory / "stilt.log"
        succeeded = False
        try:
            runner = HYSPLITDriver(
                directory=self.directory,
                receptor=self.receptor,
                params=self.params,
                met_files=self.met_files,
            )
            runner.prepare()
            result = runner.execute(timeout=timeout, rm_dat=rm_dat)
            if result.particles.empty:
                raise EmptyTrajectoryError(f"No trajectory data for {self.id}")
            self._trajectories = Trajectories.from_particles(
                result.particles,
                receptor=self.receptor,
                params=self.params,
                met_files=self.source_met_files,
            )
            self._footprint = None
            if write:
                run.write_particles(self._trajectories)
            succeeded = True
        finally:
            if scratch_log.exists():
                run.write_log(self.receptor_id, scratch_log.read_text())
            self._finish_scratch(run, keep=self.keep_scratch or not succeeded)

    def _finish_scratch(self, run: Run, *, keep: bool) -> None:
        """Copy the working directory into the output directory when *keep*, then remove it."""
        if not self.directory.exists():
            return
        if keep:
            target = run.scratch_path(self.receptor_id)
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.directory, target, symlinks=True)
        shutil.rmtree(self.directory, ignore_errors=True)
        self._met_files = None

    def generate_footprint(
        self,
        config: FootprintConfig | None = None,
        write: bool = False,
        transforms: Sequence[ParticleTransform] | None = None,
        context: TransformContext | None = None,
    ) -> Footprint | None:
        """
        Calculate the footprint from the particles.

        Runs HYSPLIT first when there are no particles yet. With the
        variant's own settings the result is kept as :attr:`footprint`.
        Other settings are a footprint of their own: with ``write=True`` it
        goes to its own folder in the output directory, beside the variant's,
        and :attr:`footprint` is left alone. When no particle reaches the grid
        there is no footprint: the method returns ``None`` and, with
        ``write=True``, records that with the reason.

        Parameters
        ----------
        config : FootprintConfig, optional
            Footprint settings. Defaults to the variant's own. Pass other
            settings to try them without a new variant, for example
            ``sim.footprint_config.model_copy(update={"smooth_factor": 0.5})``.
        write : bool, default False
            Also write the footprint to the output directory, and the
            particles when HYSPLIT had to run.
        transforms : sequence of ParticleTransform, optional
            Extra particle transforms, applied after ``config.transforms``
            and recorded with them. Any object with an
            ``apply(particles, context)`` method works.
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
        own = config is None and not transforms
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

        if transforms:
            config = config.model_copy(
                update={"transforms": [*config.transforms, *transforms]}
            )
        feet: Footprints | None = None
        if write:
            feet = (
                self.outputs.ensure_footprints()
                if own
                else self.outputs.ensure_run().footprints(config, name=self.variant)
            )
            assert feet is not None
        try:
            foot = traj.footprint(
                config, name=self.variant, context=context or self.transform_context()
            )
        except EmptyFootprintError as error:
            if own:
                self._footprint = None
            if feet is not None:
                feet.write_empty(self.receptor, error.reason, name=self.variant)
            return None
        if own:
            self._footprint = foot
        if feet is not None:
            feet.write(foot)
        return foot

    def transform_context(self) -> TransformContext:
        """
        Return the :class:`~stilt.TransformContext` passed to this simulation's transforms.

        It holds the receptor, the variant name, and the project directory,
        from which a transform can read per-receptor inputs such as an
        averaging-kernel table. Use it to apply the footprint's transforms
        outside :meth:`generate_footprint`.
        """
        return TransformContext(
            receptor=self.receptor, variant=self.variant, directory=self.project_dir
        )


__all__ = ["SimID", "Simulation", "VariantOutput"]
