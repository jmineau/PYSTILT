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

    def execute(self, timeout=None, rm_dat=None):
        _StopDriver.seen["timeout"] = timeout
        raise RuntimeError("stop before running HYSPLIT")


@pytest.fixture
def sim(monkeypatch, tmp_path, point_receptor):
    """A Simulation with the HYSPLIT driver stubbed out."""
    from stilt.config import MetConfig, VariantConfig
    from stilt.meteorology import MetStream
    from stilt.output import Output
    from stilt.simulation import VariantOutput

    monkeypatch.setattr(simmod, "HYSPLITDriver", _StopDriver)
    _StopDriver.seen.clear()
    met_config = MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )

    def _make(timeout):
        config = VariantConfig(
            name="hrrr",
            group="hrrr",
            met="hrrr",
            n_hours=-24,
            numpar=10,
            timeout=timeout,
        )
        outputs = VariantOutput(
            Output(tmp_path / "output"),
            "hrrr",
            config.transport_settings(met_config),
            None,
        )
        s = simmod.Simulation(
            point_receptor,
            config,
            met=MetStream("hrrr", met_config),
            outputs=outputs,
            directory=tmp_path / "scratch",
        )
        monkeypatch.setattr(
            type(s), "met_files", property(lambda self: []), raising=False
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
