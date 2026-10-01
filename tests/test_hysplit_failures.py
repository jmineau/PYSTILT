"""Tests for stilt.transport.hysplit.failures: reading why a HYSPLIT run failed from its log."""

from stilt.transport.hysplit import FailureReason, identify_failure_reason


def test_failure_reason_is_str():
    assert FailureReason.MISSING_MET_FILES == "MISSING_MET_FILES"
    assert isinstance(FailureReason.MISSING_MET_FILES, str)


def test_all_failure_reasons_exist():
    expected = {
        "MISSING_MET_FILES",
        "MET_COVERAGE",
        "MET_TRUNCATED",
        "VARYING_MET_INTERVAL",
        "NO_PARTICLE_DATA",
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
    assert identify_failure_reason(tmp_path) is FailureReason.NO_PARTICLE_DATA


def test_identify_failure_reason_fortran(tmp_path):
    (tmp_path / "stilt.log").write_text("Fortran runtime error: end of file\n")
    assert identify_failure_reason(tmp_path) is FailureReason.FORTRAN_RUNTIME_ERROR


def test_identify_failure_reason_met_truncated(tmp_path):
    (tmp_path / "stilt.log").write_text(
        "Meteorology ends early: the particles stop 13 h into a 24 h run.\n"
    )
    assert identify_failure_reason(tmp_path) is FailureReason.MET_TRUNCATED


def test_identify_failure_reason_unknown(tmp_path):
    (tmp_path / "stilt.log").write_text("something completely unrecognized\n")
    assert identify_failure_reason(tmp_path) is FailureReason.UNKNOWN
