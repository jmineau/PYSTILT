"""Tests for the transport model boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from stilt.config import MetConfig, STILTParams, TransportSettings, VariantConfig
from stilt.execution import run_trajectories, worker
from stilt.hysplit import HysplitModel
from stilt.model import ModelRun, get_model
from stilt.output import Output
from stilt.simulation import Simulation


def test_get_model_returns_hysplit_by_default_and_by_name():
    assert isinstance(get_model(), HysplitModel)
    assert get_model("hysplit").name == "hysplit"


def test_get_model_names_the_models_it_knows():
    with pytest.raises(
        ValueError, match=r"Unknown transport model 'flexpart'.*hysplit"
    ):
        get_model("flexpart")


def test_hysplit_model_version_is_the_bundled_build_or_exe_dirs(tmp_path):
    from stilt.config import hysplit_version

    model = HysplitModel()
    assert model.version(STILTParams(n_hours=-1)) == hysplit_version()

    (tmp_path / "version").write_text("v9.9.9+patched\n")
    assert model.version(STILTParams(n_hours=-1, exe_dir=tmp_path)) == "v9.9.9+patched"


def test_the_settings_record_the_model_that_makes_the_particles(tmp_path):
    met = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    settings = TransportSettings.build(STILTParams(n_hours=-1), met)
    model = get_model(settings.model.name)
    assert settings.model.version == model.version(settings)


class _FakeDriver:
    """Stands in for HYSPLITDriver: records what it was built with."""

    built: dict = {}

    def __init__(self, **kwargs):
        _FakeDriver.built = kwargs

    def prepare(self):
        _FakeDriver.built["prepared"] = True

    def execute(self, timeout=None, rm_dat=None):
        _FakeDriver.built.update(timeout=timeout, rm_dat=rm_dat)
        return type("Result", (), {"particles": pd.DataFrame({"indx": [1]})})()


class _FakeMet:
    def __init__(self, source, read):
        self.source, self.read = source, read

    def required_files(self, r_time, n_hours):
        return self.source

    def readable(self, files):
        assert files == self.source
        return self.read


def test_hysplit_model_reads_the_met_in_place_and_records_the_source(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.hysplit import model as model_module

    monkeypatch.setattr(model_module, "HYSPLITDriver", _FakeDriver)
    source = [tmp_path / "archive" / "20230101_12"]
    cropped = [tmp_path / "crops" / "20230101_12"]
    params = STILTParams(n_hours=-1, timeout=30, rm_dat=False)

    result = HysplitModel().run(
        point_receptor,
        params,
        _FakeMet(source, cropped),
        tmp_path / "work",  # type: ignore[arg-type]
    )

    assert isinstance(result, ModelRun)
    assert result.met_files == source  # the record names the source files
    assert _FakeDriver.built["met_files"] == cropped  # HYSPLIT reads the crops
    assert _FakeDriver.built["directory"] == tmp_path / "work"
    assert _FakeDriver.built["prepared"]
    assert (_FakeDriver.built["timeout"], _FakeDriver.built["rm_dat"]) == (30, False)


def test_run_trajectories_goes_through_the_model_the_settings_name(
    tmp_path, point_receptor, monkeypatch
):
    calls: list[dict] = []

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir):
            calls.append(
                {"receptor": receptor, "timeout": params.timeout, "workdir": workdir}
            )
            particles = pd.DataFrame(
                {
                    "time": [-60.0],
                    "indx": [1.0],
                    "long": [-111.9],
                    "lati": [40.7],
                    "zagl": [10.0],
                    "foot": [1e-5],
                }
            )
            return ModelRun(particles=particles, met_files=[tmp_path / "met_file"])

    asked: list[str] = []
    monkeypatch.setattr(
        worker, "get_model", lambda name: asked.append(name) or _Model()
    )
    met_config = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    variant = VariantConfig(
        name="hrrr",
        group="hrrr",
        met="hrrr",
        transport=TransportSettings.build(
            STILTParams(n_hours=-1, numpar=1, hnf_plume=False), met_config
        ),
    )
    sim = Simulation(point_receptor, variant, Output(tmp_path / "output"))

    traj = run_trajectories(sim, met=object(), workdir=tmp_path / "work", timeout=45)  # type: ignore[arg-type]

    assert asked == ["hysplit"]
    assert calls[0]["receptor"] == point_receptor
    assert calls[0]["timeout"] == 45  # the override reaches the model
    assert traj.met_files == [tmp_path / "met_file"]
    assert sim.has_trajectory
