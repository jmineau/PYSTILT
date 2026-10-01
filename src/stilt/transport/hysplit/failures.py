"""Why a HYSPLIT run failed, read from the messages in its log."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class FailureReason(StrEnum):
    """Why a HYSPLIT run failed, as read from its ``stilt.log``."""

    MISSING_MET_FILES = "MISSING_MET_FILES"
    MET_COVERAGE = "MET_COVERAGE"
    MET_TRUNCATED = "MET_TRUNCATED"
    VARYING_MET_INTERVAL = "VARYING_MET_INTERVAL"
    NO_PARTICLE_DATA = "NO_PARTICLE_DATA"
    FORTRAN_RUNTIME_ERROR = "FORTRAN_RUNTIME_ERROR"
    EMPTY_LOG = "EMPTY_LOG"
    UNKNOWN = "UNKNOWN"


#: Phrases written to stilt.log by HYSPLIT or PYSTILT, mapped to their FailureReason.
FAILURE_PHRASES: dict[str, FailureReason] = {
    # PYSTILT's MeteorologyError (raised Python-side before HYSPLIT runs, then
    # written to the log by the phase runner) shares HYSPLIT's exact wording, so
    # one phrase classifies both.
    "Insufficient number of meteorological files found": FailureReason.MISSING_MET_FILES,
    "start point not within (x,y,t) any data file": FailureReason.MET_COVERAGE,
    "start time after end of meteorology data": FailureReason.MET_COVERAGE,
    "meteorological data time interval varies": FailureReason.VARYING_MET_INTERVAL,
    # Written by the HYSPLIT driver, not HYSPLIT (see MET_TRUNCATED_WARNING).
    "Meteorology ends early": FailureReason.MET_TRUNCATED,
    "PARTICLE_STILT.DAT does not contain any trajectory data": FailureReason.NO_PARTICLE_DATA,
    # PYSTILT's own errors for a run without particles, by the class name the
    # worker writes to the log ("Type: ...").
    "NoParticleOutputError": FailureReason.NO_PARTICLE_DATA,
    "EmptyParticleOutputError": FailureReason.NO_PARTICLE_DATA,
    "Fortran runtime error": FailureReason.FORTRAN_RUNTIME_ERROR,
}

#: HYSPLIT's warning for a met file that holds one time period. It is a
#: failure only when the particles also stop before the end of the run.
MET_TRUNCATED_WARNING = "Only one time period of meteo data"


def identify_failure_reason(path: str | Path) -> FailureReason:
    """
    Return why a simulation failed, from the messages in its log.

    Parameters
    ----------
    path : str or Path
        The log file, or a directory holding ``stilt.log``.

    Returns
    -------
    FailureReason
        The reason for the first known message in the log. ``EMPTY_LOG``
        when there is no log, and ``UNKNOWN`` when no known message matches.
    """
    log = Path(path)
    if log.is_dir():
        log = log / "stilt.log"
    if not log.exists():
        return FailureReason.EMPTY_LOG
    text = log.read_text()
    for phrase, reason in FAILURE_PHRASES.items():
        if phrase in text:
            return reason
    return FailureReason.UNKNOWN
