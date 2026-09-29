"""Errors raised while running simulations, and the HYSPLIT failures they report."""

from enum import Enum
from pathlib import Path


class FailureReason(str, Enum):
    """Why a HYSPLIT run failed, as read from its ``stilt.log``."""

    MISSING_MET_FILES = "MISSING_MET_FILES"
    MET_COVERAGE = "MET_COVERAGE"
    VARYING_MET_INTERVAL = "VARYING_MET_INTERVAL"
    NO_TRAJECTORY_DATA = "NO_TRAJECTORY_DATA"
    FORTRAN_RUNTIME_ERROR = "FORTRAN_RUNTIME_ERROR"
    EMPTY_LOG = "EMPTY_LOG"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        """Return the underlying reason code."""
        return self.value


#: Phrases written to stilt.log by HYSPLIT, mapped to their FailureReason.
FAILURE_PHRASES: dict[str, FailureReason] = {
    # PYSTILT's MeteorologyError (raised Python-side before HYSPLIT runs, then
    # written to the log by the phase runner) shares HYSPLIT's exact wording, so
    # one phrase classifies both.
    "Insufficient number of meteorological files found": FailureReason.MISSING_MET_FILES,
    "start point not within (x,y,t) any data file": FailureReason.MET_COVERAGE,
    "start time after end of meteorology data": FailureReason.MET_COVERAGE,
    "meteorological data time interval varies": FailureReason.VARYING_MET_INTERVAL,
    "PARTICLE_STILT.DAT does not contain any trajectory data": FailureReason.NO_TRAJECTORY_DATA,
    "Fortran runtime error": FailureReason.FORTRAN_RUNTIME_ERROR,
}


def identify_failure_reason(path: str | Path) -> FailureReason:
    """
    Return why a simulation failed, from the messages in its ``stilt.log``.

    Parameters
    ----------
    path : str or Path
        Simulation directory holding ``stilt.log``.

    Returns
    -------
    FailureReason
        The reason for the first known message in the log. ``EMPTY_LOG``
        when there is no log, and ``UNKNOWN`` when no known message matches.
    """
    log = Path(path) / "stilt.log"
    if not log.exists():
        return FailureReason.EMPTY_LOG
    text = log.read_text()
    for phrase, reason in FAILURE_PHRASES.items():
        if phrase in text:
            return reason
    return FailureReason.UNKNOWN


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class SimulationError(RuntimeError):
    """Base class for errors raised while running a simulation."""


class ConfigValidationError(SimulationError):
    """The model or run settings are invalid or contradict each other."""


class ConfigChangedError(ConfigValidationError):
    """
    A variant that has already run now has different settings.

    The project records the settings each variant ran with, and its outputs
    would no longer match its name if they changed. Declare a new variant
    for the new settings, or delete the old outputs with
    :meth:`stilt.Model.remove` (``stilt rm``) and run it again.
    """


class MeteorologyError(SimulationError):
    """The meteorology files a simulation needs could not be found or staged."""


class HYSPLITTimeoutError(SimulationError):
    """HYSPLIT (``hycs_std``) ran longer than the configured timeout."""


class NoParticleOutputError(SimulationError):
    """HYSPLIT finished without writing ``PARTICLE_STILT.DAT``."""


class EmptyTrajectoryError(SimulationError):
    """HYSPLIT ran, but its particle output holds no particles."""


class EmptyFootprintError(RuntimeError):
    """
    No particle is over the footprint grid, so there is no footprint.

    This is an outcome, not a failure. :meth:`stilt.Simulation.generate_footprint`
    catches it and writes the ``.empty`` marker.

    Parameters
    ----------
    reason : str
        ``"no_particles"`` when the particle table is empty, or
        ``"outside_domain"`` when no particle reached the grid.
    """

    def __init__(self, reason: str):
        super().__init__(f"No particle over the footprint grid ({reason}).")
        self.reason = reason


class HYSPLITFailureError(SimulationError):
    """
    HYSPLIT wrote a known failure message to its log.

    Parameters
    ----------
    reason : FailureReason
        The failure the message identifies.
    log_path : str or Path
        The HYSPLIT log holding the message.

    Attributes
    ----------
    reason : FailureReason
        The failure the message identifies.
    """

    def __init__(self, reason: FailureReason, log_path: str | Path):
        self.reason = reason
        super().__init__(f"HYSPLIT failed with {reason}; see {log_path}")
