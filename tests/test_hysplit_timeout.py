"""
A wedged hycs_std must be capped, not waited on forever.

Without a deadline a single spinning HYSPLIT process holds its batch worker until Slurm
kills the task, which on a preempted-and-requeued guest job can be days. `params.timeout`
makes that a HYSPLITTimeoutError the execution loop already handles.
"""

import pytest

from stilt import simulation as simmod
from stilt.config.params import STILTParams


class _StopDriver:
    """Stands in for HYSPLITDriver and records the timeout it was handed."""

    seen: dict = {}

    def __init__(self, **kwargs):
        pass

    def prepare(self):
        pass

    def execute(
        self, timeout=None, rm_dat=None, error_only=False, error_realizations=(0,)
    ):
        _StopDriver.seen["timeout"] = timeout
        raise RuntimeError("stop before running HYSPLIT")


@pytest.fixture
def sim(monkeypatch, tmp_path):
    """A Simulation with the HYSPLIT driver stubbed out."""
    monkeypatch.setattr(simmod, "HYSPLITDriver", _StopDriver)
    _StopDriver.seen.clear()

    def _make(timeout):
        s = object.__new__(simmod.Simulation)
        s.directory = tmp_path
        s.receptor = None
        s.params = STILTParams(timeout=timeout)
        s._exe_dir = None
        monkeypatch.setattr(
            type(s), "met_files", property(lambda self: []), raising=False
        )
        monkeypatch.setattr(
            type(s), "_can_reuse_main_for_error", lambda self: False, raising=False
        )
        return s

    return _make


def test_timeout_defaults_to_none():
    assert STILTParams().timeout is None


def test_timeout_is_configurable():
    assert STILTParams(timeout=900).timeout == 900


def test_run_trajectories_falls_back_to_params_timeout(sim):
    with pytest.raises(RuntimeError):
        simmod.Simulation.run_trajectories(sim(600))
    assert _StopDriver.seen["timeout"] == 600


def test_explicit_timeout_wins_over_params(sim):
    with pytest.raises(RuntimeError):
        simmod.Simulation.run_trajectories(sim(600), timeout=30)
    assert _StopDriver.seen["timeout"] == 30


def test_no_timeout_configured_still_waits_indefinitely(sim):
    with pytest.raises(RuntimeError):
        simmod.Simulation.run_trajectories(sim(None))
    assert _StopDriver.seen["timeout"] is None
