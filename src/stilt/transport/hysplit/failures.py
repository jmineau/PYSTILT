"""Why a HYSPLIT run failed, read from the messages in its log and its WARNING file."""

from __future__ import annotations

from enum import StrEnum


class FailureReason(StrEnum):
    """
    Why a HYSPLIT run failed.

    The driver reads it from HYSPLIT's log and ``WARNING`` file when the
    run ends, or knows it (a timeout, no particle output), and raises a
    :class:`~stilt.exceptions.SimulationError` with it as ``reason``. The
    worker records it with the simulation.
    """

    MET_COVERAGE = "MET_COVERAGE"
    VARYING_MET_INTERVAL = "VARYING_MET_INTERVAL"
    NO_PARTICLE_DATA = "NO_PARTICLE_DATA"
    FORTRAN_RUNTIME_ERROR = "FORTRAN_RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"


#: Phrases HYSPLIT (or the driver) writes to stilt.log or HYSPLIT's WARNING
#: file, mapped to their FailureReason. The four ``metpos`` warnings say a
#: particle inside the met's grid reached a time no met file holds: the met
#: ran out. HYSPLIT writes only the first such warning of a run, and
#: ``off spatial domain of all grids`` when every particle left the met's
#: domain, which is not a failure.
FAILURE_PHRASES: dict[str, FailureReason] = {
    "start point not within (x,y,t) any data file": FailureReason.MET_COVERAGE,
    "start time after end of meteorology data": FailureReason.MET_COVERAGE,
    "no more meteorology": FailureReason.MET_COVERAGE,
    "too much time between files": FailureReason.MET_COVERAGE,
    "no meteo data at current time": FailureReason.MET_COVERAGE,
    "no data at current time on any grid": FailureReason.MET_COVERAGE,
    "meteorological data time interval varies": FailureReason.VARYING_MET_INTERVAL,
    "PARTICLE_STILT.DAT does not contain any trajectory data": FailureReason.NO_PARTICLE_DATA,
    "Fortran runtime error": FailureReason.FORTRAN_RUNTIME_ERROR,
}


def failure_in(text: str) -> tuple[FailureReason, str] | None:
    """
    Return the first known failure in HYSPLIT's messages, as its reason and the line that says it.

    Phrases are tried in the order of :data:`FAILURE_PHRASES`. ``None``
    when the log holds none of them.
    """
    for phrase, reason in FAILURE_PHRASES.items():
        if phrase in text:
            line = next(line for line in text.splitlines() if phrase in line)
            return reason, line.strip()
    return None
