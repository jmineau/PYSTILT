"""Tests for stilt.transport.hysplit.failures: why a HYSPLIT run failed, from its log."""

import pytest

from stilt.transport.hysplit import FailureReason
from stilt.transport.hysplit.failures import failure_in


def test_failure_reason_is_str():
    assert FailureReason.MET_COVERAGE == "MET_COVERAGE"
    assert isinstance(FailureReason.MET_COVERAGE, str)


def test_all_failure_reasons_exist():
    expected = {
        "MET_COVERAGE",
        "VARYING_MET_INTERVAL",
        "NO_PARTICLE_DATA",
        "FORTRAN_RUNTIME_ERROR",
        "TIMEOUT",
    }
    assert {r.value for r in FailureReason} == expected


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("meteorological data time interval varies", "VARYING_MET_INTERVAL"),
        ("PARTICLE_STILT.DAT does not contain any trajectory data", "NO_PARTICLE_DATA"),
        ("Fortran runtime error: end of file", "FORTRAN_RUNTIME_ERROR"),
        ("start point not within (x,y,t) any data file", "MET_COVERAGE"),
    ],
)
def test_failure_in_names_the_reason_and_the_line_for_a_known_message(line, reason):
    assert failure_in(f"some output\n  {line}\nmore output\n") == (reason, line)


def test_failure_in_returns_none_for_an_unknown_log():
    assert failure_in("something completely unrecognized\n") is None
