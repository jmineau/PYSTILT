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

    Attributes
    ----------
    reason : str or None
        The cause as a plain string (an enum member such as a
        ``FailureReason`` is stored by its value), or ``None`` when it has no
        short name.
    """

    reason: str | None = None

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        if reason is not None:
            self.reason = str(reason)  # a plain string, as the failure record stores it


class MeteorologyError(SimulationError):
    """The meteorology files a simulation needs could not be found or staged."""

    reason = "MISSING_MET_FILES"


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
    catches it and writes a footprint file with no rows and the reason.

    Parameters
    ----------
    reason : str
        ``"no_particles"`` when the particle table is empty, or
        ``"outside_domain"`` when no particle reached the grid.

    Attributes
    ----------
    reason : str
        Why the footprint is empty.
    """

    def __init__(self, reason: str):
        super().__init__(f"No particle over the footprint grid ({reason}).")
        self.reason = reason


__all__ = [
    "EmptyFootprint",
    "HYSPLITNotFoundError",
    "MeteorologyError",
    "SimulationError",
    "StiltError",
]
