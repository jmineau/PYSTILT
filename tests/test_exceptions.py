"""Tests for stilt.exceptions: the exception PYSTILT adds to runtraj's."""

from runtraj.exceptions import RuntrajError, SimulationError

import stilt
from stilt import exceptions
from stilt.exceptions import EmptyFootprint


def test_empty_footprint_is_the_only_exception_here():
    assert exceptions.__all__ == ["EmptyFootprint"]
    assert EmptyFootprint.__bases__ == (RuntrajError,)


def test_empty_footprint_is_not_a_failure():
    assert not issubclass(EmptyFootprint, (SimulationError, RuntimeError))
    assert str(EmptyFootprint()) == "No particle is over the footprint grid."


def test_the_base_class_is_runtrajs():
    assert stilt.RuntrajError is RuntrajError
