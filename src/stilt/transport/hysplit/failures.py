"""Why a HYSPLIT run failed, read from the messages in its log while it runs."""

from __future__ import annotations

from enum import StrEnum


class FailureReason(StrEnum):
    """
    Why a HYSPLIT run failed.

    The driver reads it from HYSPLIT's log when the run ends, or knows it
    (a timeout, no particle output), and raises a
    :class:`~stilt.exceptions.SimulationError` with it as ``reason``. The
    worker records it with the simulation.
    """

    MISSING_MET_FILES = "MISSING_MET_FILES"
    MET_COVERAGE = "MET_COVERAGE"
    MET_TRUNCATED = "MET_TRUNCATED"
    VARYING_MET_INTERVAL = "VARYING_MET_INTERVAL"
    NO_PARTICLE_DATA = "NO_PARTICLE_DATA"
    FORTRAN_RUNTIME_ERROR = "FORTRAN_RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"


#: Phrases HYSPLIT (or the driver) writes to stilt.log, mapped to their FailureReason.
FAILURE_PHRASES: dict[str, FailureReason] = {
    "Insufficient number of meteorological files found": FailureReason.MISSING_MET_FILES,
    "start point not within (x,y,t) any data file": FailureReason.MET_COVERAGE,
    "start time after end of meteorology data": FailureReason.MET_COVERAGE,
    "meteorological data time interval varies": FailureReason.VARYING_MET_INTERVAL,
    # Written by the HYSPLIT driver, not HYSPLIT (see MET_TRUNCATED_WARNING).
    "Meteorology ends early": FailureReason.MET_TRUNCATED,
    "PARTICLE_STILT.DAT does not contain any trajectory data": FailureReason.NO_PARTICLE_DATA,
    "Fortran runtime error": FailureReason.FORTRAN_RUNTIME_ERROR,
}

#: HYSPLIT's warning for a met file that holds one time period. It is a
#: failure only when the particles also stop before the end of the run.
MET_TRUNCATED_WARNING = "Only one time period of meteo data"


def failure_in(text: str) -> FailureReason | None:
    """Return the reason for the first known failure message in *text*, in the order of :data:`FAILURE_PHRASES`, or ``None``."""
    for phrase, reason in FAILURE_PHRASES.items():
        if phrase in text:
            return reason
    return None
