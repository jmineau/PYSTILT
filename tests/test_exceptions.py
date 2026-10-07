"""Tests for stilt.exceptions: one StiltError base, each under its builtin."""

import inspect

import pytest

import stilt
from stilt import exceptions
from stilt.exceptions import (
    EmptyFootprint,
    HYSPLITNotFoundError,
    MeteorologyError,
    SimulationError,
    StiltError,
)
from stilt.transport.hysplit import FailureReason

#: Each class and the parents it must have (#80).
PARENTS = {
    StiltError: (Exception,),
    SimulationError: (StiltError, RuntimeError),
    MeteorologyError: (SimulationError,),
    HYSPLITNotFoundError: (StiltError, FileNotFoundError),
    EmptyFootprint: (StiltError,),
}


def test_every_exception_class_is_in_the_table():
    classes = {
        obj
        for _, obj in inspect.getmembers(exceptions, inspect.isclass)
        if issubclass(obj, BaseException) and obj.__module__ == exceptions.__name__
    }
    assert classes == set(PARENTS)
    assert set(exceptions.__all__) == {cls.__name__ for cls in PARENTS}


@pytest.mark.parametrize("cls", list(PARENTS), ids=lambda cls: cls.__name__)
def test_parents(cls):
    assert cls.__bases__ == PARENTS[cls]
    assert issubclass(cls, StiltError)


def test_stilt_error_is_exported():
    assert stilt.StiltError is StiltError


def test_empty_footprint_is_not_a_failure():
    assert not issubclass(EmptyFootprint, (SimulationError, RuntimeError))
    assert str(EmptyFootprint()) == "No particle is over the footprint grid."


def test_hysplit_not_found_is_not_a_failed_run():
    """A missing executable is a setup problem, so a worker reports it as an error."""
    assert not issubclass(HYSPLITNotFoundError, SimulationError)


def test_a_simulation_error_carries_its_reason():
    err = SimulationError(
        "HYSPLIT failed (MET_COVERAGE).", reason=FailureReason.MET_COVERAGE
    )
    assert err.reason == "MET_COVERAGE"
    assert type(err.reason) is str  # the failure record's YAML writes only plain types
    assert SimulationError("no short name").reason is None


def test_a_meteorology_error_is_a_met_coverage_failure():
    assert (
        MeteorologyError("No met file for 2021-01-14 00:00.").reason == "MET_COVERAGE"
    )
