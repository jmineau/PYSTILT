"""
The exception PYSTILT adds to runtraj's.

Every exception a run raises on purpose is a
:class:`runtraj.RuntrajError` (:mod:`runtraj.exceptions`), so
``except stilt.RuntrajError`` catches those and this one.
"""

from __future__ import annotations

from runtraj.exceptions import RuntrajError


class EmptyFootprint(RuntrajError):
    """
    No particle is over the footprint grid, so there is no footprint.

    An empty footprint is a finished result, not a failure. The footprint
    stage catches it and writes a footprint file with no rows, marked empty.
    The message says why it is empty: the particles are off the grid, no
    particle row has a positive ``foot``, or the table has no rows.
    """

    def __init__(
        self, message: str = "No particle is over the footprint grid."
    ) -> None:
        super().__init__(message)


__all__ = ["EmptyFootprint"]
