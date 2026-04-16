"""Tests for stilt.errors - FailureReason, identify_failure_reason, exception hierarchy."""

import pytest

from stilt.errors import (
    EmptyTrajectoryError,
    FailureReason,
    HYSPLITFailureError,
    HYSPLITTimeoutError,
    NoParticleOutputError,
    SimulationError,
    identify_failure_reason,
)

# ---------------------------------------------------------------------------
# FailureReason
# ---------------------------------------------------------------------------


def test_failure_reason_is_str():
    assert FailureReason.MISSING_MET_FILES == "MISSING_MET_FILES"
    assert isinstance(FailureReason.MISSING_MET_FILES, str)


def test_all_failure_reasons_exist():
    expected = {
        "MISSING_MET_FILES",
        "VARYING_MET_INTERVAL",
        "NO_TRAJECTORY_DATA",
        "FORTRAN_RUNTIME_ERROR",
        "EMPTY_LOG",
        "UNKNOWN",
    }
    assert {r.value for r in FailureReason} == expected


# ---------------------------------------------------------------------------
# identify_failure_reason
# ---------------------------------------------------------------------------


def test_identify_failure_reason_no_log(tmp_path):
    assert identify_failure_reason(tmp_path) is FailureReason.EMPTY_LOG


def test_identify_failure_reason_missing_met(tmp_path):
    (tmp_path / "stilt.log").write_text(
        "Insufficient number of meteorological files found for time step\n"
    )
    assert identify_failure_reason(tmp_path) is FailureReason.MISSING_MET_FILES


def test_identify_failure_reason_varying_met(tmp_path):
    (tmp_path / "stilt.log").write_text("meteorological data time interval varies\n")
    assert identify_failure_reason(tmp_path) is FailureReason.VARYING_MET_INTERVAL


def test_identify_failure_reason_no_traj(tmp_path):
    (tmp_path / "stilt.log").write_text(
        "PARTICLE_STILT.DAT does not contain any trajectory data\n"
    )
    assert identify_failure_reason(tmp_path) is FailureReason.NO_TRAJECTORY_DATA


def test_identify_failure_reason_fortran(tmp_path):
    (tmp_path / "stilt.log").write_text("Fortran runtime error: end of file\n")
    assert identify_failure_reason(tmp_path) is FailureReason.FORTRAN_RUNTIME_ERROR


def test_identify_failure_reason_unknown(tmp_path):
    (tmp_path / "stilt.log").write_text("something completely unrecognized\n")
    assert identify_failure_reason(tmp_path) is FailureReason.UNKNOWN


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_simulation_error_is_runtime_error():
    assert issubclass(SimulationError, RuntimeError)


def test_all_subclasses_inherit_simulation_error():
    for cls in (
        HYSPLITTimeoutError,
        NoParticleOutputError,
        EmptyTrajectoryError,
        HYSPLITFailureError,
    ):
        assert issubclass(cls, SimulationError)


def test_hysplit_failure_error_stores_reason():
    err = HYSPLITFailureError(FailureReason.MISSING_MET_FILES)
    assert err.reason is FailureReason.MISSING_MET_FILES
    assert "MISSING_MET_FILES" in str(err)


def test_hysplit_failure_error_includes_sim_id():
    err = HYSPLITFailureError(FailureReason.FORTRAN_RUNTIME_ERROR, "202301011200_abc")
    assert "202301011200_abc" in str(err)


def test_hysplit_failure_error_catchable_as_simulation_error():
    with pytest.raises(SimulationError):
        raise HYSPLITFailureError(FailureReason.UNKNOWN)
