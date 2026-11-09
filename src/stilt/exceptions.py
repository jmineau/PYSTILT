"""
Exceptions PYSTILT raises.

Every one is a :class:`StiltError`, so ``except stilt.StiltError`` catches
anything PYSTILT raised on purpose. Each also subclasses the builtin that
describes it, so ``except RuntimeError`` or ``except FileNotFoundError``
keeps working. Plain input checks raise builtins such as ``ValueError``.
"""

from __future__ import annotations


class StiltError(Exception):
    """Base class for every exception PYSTILT raises."""


class SimulationError(StiltError, RuntimeError):
    """
    A simulation failed while it ran.

    The worker catches it, records it with the simulation
    (:attr:`stilt.Simulation.failure`), and goes on to the next one.

    Parameters
    ----------
    message : str
        What went wrong.
    reason : str, optional
        A short name for the cause, such as ``"MET_COVERAGE"`` or
        ``"TIMEOUT"`` (:class:`stilt.transport.hysplit.FailureReason` lists
        HYSPLIT's). ``stilt status`` counts failures by it.
    log : str, optional
        The transport model's log of the failed run, kept with the
        simulation (:attr:`stilt.Simulation.log`).

    Attributes
    ----------
    reason : str or None
        The cause as a plain string (an enum member such as a
        ``FailureReason`` is stored by its value), or ``None`` when it has no
        short name.
    log : str
        The model's log, empty when it wrote none.
    """

    reason: str | None = None
    log: str = ""

    def __init__(self, message: str, reason: str | None = None, log: str = ""):
        super().__init__(message)
        if reason is not None:
            self.reason = str(reason)  # a plain string, as the failure record stores it
        self.log = log


class MeteorologyError(SimulationError):
    """The meteorology files a simulation needs could not be found or staged."""

    reason = "MET_COVERAGE"


class HYSPLITNotFoundError(StiltError, FileNotFoundError):
    """
    The HYSPLIT executable (``hycs_std``) is missing.

    Raised when there is no bundled build for this platform, and when
    ``exe_dir`` holds no ``hycs_std``. Build ``hycs_std`` and set
    ``exe_dir`` in ``config.yaml`` to the directory that holds it.
    """


class EmptyFootprint(StiltError):
    """
    No particle is over the footprint grid, so there is no footprint.

    An empty footprint is a finished result, not a failure. The worker
    catches it and writes a footprint file with no rows, marked empty.
    """

    def __init__(self) -> None:
        super().__init__("No particle is over the footprint grid.")


class WorkerDied(StiltError, RuntimeError):
    """
    A worker process died without Python's say, and the run stopped.

    A run on ``cpus`` above 1 raises it when a worker is killed outright
    (``SIGKILL``, as the kernel's out-of-memory killer sends). The pool
    cannot say which receptor the worker held, so the message names every
    receptor that was in progress. What finished is kept and nothing is marked
    failed: run again, and only what is missing runs. ``stilt run`` ends
    with exit code 3.
    """


__all__ = [
    "EmptyFootprint",
    "HYSPLITNotFoundError",
    "MeteorologyError",
    "SimulationError",
    "StiltError",
    "WorkerDied",
]
