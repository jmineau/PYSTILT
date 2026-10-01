"""
Exceptions PYSTILT raises.

Every one is a :class:`StiltError`, so ``except stilt.StiltError`` catches
anything PYSTILT raised on purpose. Each also subclasses the builtin that
describes it, so ``except RuntimeError`` or ``except FileNotFoundError``
keeps working. Plain input checks raise builtins such as ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stilt.hysplit.failures import FailureReason


class StiltError(Exception):
    """Base class for every exception PYSTILT raises."""


class SimulationError(StiltError, RuntimeError):
    """A simulation failed while it ran."""


class MeteorologyError(SimulationError):
    """The meteorology files a simulation needs could not be found or staged."""


class HYSPLITTimeoutError(SimulationError):
    """HYSPLIT (``hycs_std``) ran longer than the configured timeout."""


class NoParticleOutputError(SimulationError):
    """HYSPLIT finished without writing ``PARTICLE_STILT.DAT``."""


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


class EmptyTrajectoryError(SimulationError):
    """HYSPLIT ran, but its particle output holds no particles."""


class HYSPLITNotFoundError(StiltError, FileNotFoundError):
    """
    The HYSPLIT executable (``hycs_std``) is missing.

    Raised when there is no bundled build for this platform, when
    ``exe_dir`` holds no ``hycs_std``, and when the executable is gone by
    the time the run starts. Build ``hycs_std`` and set ``exe_dir`` in
    ``config.yaml`` to the directory that holds it.
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
    "EmptyTrajectoryError",
    "HYSPLITFailureError",
    "HYSPLITNotFoundError",
    "HYSPLITTimeoutError",
    "MeteorologyError",
    "NoParticleOutputError",
    "SimulationError",
    "StiltError",
]
