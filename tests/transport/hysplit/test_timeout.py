"""
A wedged hycs_std must be capped, not waited on forever.

Without a deadline a single spinning HYSPLIT process holds its batch worker until Slurm
kills the task, which on a preempted-and-requeued guest job can be days. `execution.timeout`
makes that a HYSPLITTimeoutError the execution loop already handles.
"""

import pytest

from stilt.config import ExecutionConfig
from stilt.transport.hysplit.config import HysplitConfig


class _StopDriver:
    """Stands in for HYSPLITDriver and records the timeout it was handed."""

    seen: dict = {}

    def __init__(self, **kwargs):
        pass

    def prepare(self):
        pass

    def execute(self, timeout=None):
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
    from stilt.config import MetConfig
    from stilt.output import Output
    from stilt.simulation import Simulation
    from stilt.transport import ModelInfo
    from stilt.transport.hysplit import model
    from stilt.variants import Variant

    monkeypatch.setattr(model, "HYSPLITDriver", _StopDriver)
    _StopDriver.seen.clear()
    met_config = MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )

    def _make():
        params = HysplitConfig(n_hours=-24, numpar=10)
        config = Variant(
            name="hrrr",
            group="hrrr",
            met="hrrr",
            met_config=met_config,
            transport=params,
            model=ModelInfo(version="v5.1.0"),
        )
        return Simulation(point_receptor, config, Output(tmp_path / "output"))

    return _make


def _run(sim, tmp_path, **kwargs):
    from stilt.execution import run_particles

    return run_particles(sim, met=_FakeMet(), workdir=tmp_path / "scratch", **kwargs)


def test_timeout_is_an_execution_setting():
    assert ExecutionConfig().timeout is None
    assert ExecutionConfig(timeout=900).timeout == 900
    assert "timeout" not in HysplitConfig.model_fields


def test_run_particles_hands_the_model_its_timeout(sim, tmp_path):
    with pytest.raises(RuntimeError):
        _run(sim(), tmp_path, timeout=30)
    assert _StopDriver.seen["timeout"] == 30


def test_no_timeout_waits_indefinitely(sim, tmp_path):
    with pytest.raises(RuntimeError):
        _run(sim(), tmp_path)
    assert _StopDriver.seen["timeout"] is None
