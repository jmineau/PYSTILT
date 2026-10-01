"""
A wedged hycs_std must be capped, not waited on forever.

Without a deadline a single spinning HYSPLIT process holds its batch worker until Slurm
kills the task, which on a preempted-and-requeued guest job can be days. `params.timeout`
makes that a HYSPLITTimeoutError the execution loop already handles.
"""

import pytest

from stilt.config.params import STILTParams


class _StopDriver:
    """Stands in for HYSPLITDriver and records the timeout it was handed."""

    seen: dict = {}

    def __init__(self, **kwargs):
        pass

    def prepare(self):
        pass

    def execute(self, timeout=None, rm_dat=None):
        _StopDriver.seen["timeout"] = timeout
        raise RuntimeError("stop before running HYSPLIT")


class _FakeMet:
    def required_files(self, **kwargs):
        return []

    def readable(self, files):
        return files


@pytest.fixture
def sim(monkeypatch, tmp_path, point_receptor):
    """A Simulation with the HYSPLIT driver stubbed out, and a runner for it."""
    from stilt.config import MetConfig, TransportSettings, VariantConfig
    from stilt.output import Output
    from stilt.simulation import Simulation
    from stilt.transport.hysplit import model

    monkeypatch.setattr(model, "HYSPLITDriver", _StopDriver)
    _StopDriver.seen.clear()
    met_config = MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )

    def _make(timeout):
        params = STILTParams(n_hours=-24, numpar=10, timeout=timeout)
        config = VariantConfig(
            name="hrrr",
            group="hrrr",
            met="hrrr",
            transport=TransportSettings.build(params, met_config),
        )
        return Simulation(point_receptor, config, Output(tmp_path / "output"))

    return _make


def _run(sim, tmp_path, **kwargs):
    from stilt.execution import run_trajectories

    return run_trajectories(sim, met=_FakeMet(), workdir=tmp_path / "scratch", **kwargs)


def test_timeout_defaults_to_none():
    assert STILTParams().timeout is None


def test_timeout_is_configurable():
    assert STILTParams(timeout=900).timeout == 900


def test_run_trajectories_falls_back_to_params_timeout(sim, tmp_path):
    with pytest.raises(RuntimeError):
        _run(sim(600), tmp_path)
    assert _StopDriver.seen["timeout"] == 600


def test_explicit_timeout_wins_over_params(sim, tmp_path):
    with pytest.raises(RuntimeError):
        _run(sim(600), tmp_path, timeout=30)
    assert _StopDriver.seen["timeout"] == 30


def test_no_timeout_configured_still_waits_indefinitely(sim, tmp_path):
    with pytest.raises(RuntimeError):
        _run(sim(None), tmp_path)
    assert _StopDriver.seen["timeout"] is None
