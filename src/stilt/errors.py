"""STILT simulation error types and failure diagnostics."""

from enum import StrEnum
from pathlib import Path


class FailureReason(StrEnum):
    """Recognized HYSPLIT failure modes, parsed from stilt.log."""

    MISSING_MET_FILES = "MISSING_MET_FILES"
    VARYING_MET_INTERVAL = "VARYING_MET_INTERVAL"
    NO_TRAJECTORY_DATA = "NO_TRAJECTORY_DATA"
    FORTRAN_RUNTIME_ERROR = "FORTRAN_RUNTIME_ERROR"
    EMPTY_LOG = "EMPTY_LOG"
    UNKNOWN = "UNKNOWN"


#: Phrases written to stilt.log by HYSPLIT, mapped to their FailureReason.
FAILURE_PHRASES: dict[str, FailureReason] = {
    "Insufficient number of meteorological files found": FailureReason.MISSING_MET_FILES,
    "meteorological data time interval varies": FailureReason.VARYING_MET_INTERVAL,
    "PARTICLE_STILT.DAT does not contain any trajectory data": FailureReason.NO_TRAJECTORY_DATA,
    "Fortran runtime error": FailureReason.FORTRAN_RUNTIME_ERROR,
}


def identify_failure_reason(path: str | Path) -> FailureReason:
    """Parse stilt.log to identify why a simulation failed.

    Parameters
    ----------
    path : str or Path
        Simulation directory containing ``stilt.log``.

    Returns
    -------
    FailureReason
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
    """Base class for STILT simulation execution errors."""


class ConfigValidationError(SimulationError):
    """Model or run configuration is invalid or internally inconsistent."""


class MeteorologyError(SimulationError):
    """Meteorology selection/loading failed for a simulation."""


class TrajectoryError(SimulationError):
    """Trajectory-generation phase failed."""


class FootprintError(SimulationError):
    """Footprint-generation phase failed."""


class HYSPLITTimeoutError(TrajectoryError):
    """hycs_std process exceeded the configured timeout."""


class NoParticleOutputError(TrajectoryError):
    """PARTICLE_STILT.DAT was not produced by hycs_std."""


class EmptyTrajectoryError(TrajectoryError):
    """PARTICLE_STILT.DAT exists but contains no trajectory data."""


class HYSPLITFailureError(TrajectoryError):
    """hycs_std reported a recognizable failure phrase in the log.

    Attributes
    ----------
    reason : FailureReason
        The identified failure mode.
    """

    def __init__(self, reason: FailureReason, sim_id: str = ""):
        self.reason = reason
        msg = f"{sim_id}: {reason}" if sim_id else str(reason)
        super().__init__(msg)
