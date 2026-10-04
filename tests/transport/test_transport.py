"""Tests for the transport model boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from stilt.config import ProjectConfig
from stilt.execution import run_particles, worker
from stilt.meteorology import MetConfig
from stilt.output import Output
from stilt.simulation import Simulation
from stilt.transport import ModelInfo, ModelRun, get_model
from stilt.transport.hysplit import HysplitConfig, HysplitModel
from stilt.variants import Variant, resolve


def test_get_model_returns_hysplit_by_default_and_by_name():
    assert isinstance(get_model(), HysplitModel)
    assert get_model("hysplit").name == "hysplit"


def test_get_model_names_the_models_it_knows():
    with pytest.raises(
        ValueError, match=r"Unknown transport model 'flexpart'.*hysplit"
    ):
        get_model("flexpart")


def test_hysplit_model_version_is_the_bundled_build_or_exe_dirs(tmp_path):
    from stilt.transport.hysplit.model import hysplit_version

    model = HysplitModel()
    assert model.version(HysplitConfig(n_hours=-1)) == hysplit_version()

    (tmp_path / "version").write_text("v9.9.9+patched\n")
    assert (
        model.version(HysplitConfig(n_hours=-1, exe_dir=tmp_path)) == "v9.9.9+patched"
    )


def test_the_settings_record_the_model_that_makes_the_particles(tmp_path):
    met = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    variant = resolve(ProjectConfig(mets={"hrrr": met}, n_hours=-1))["hrrr"]
    model = get_model(variant.model.name)
    assert variant.model.version == model.version(variant.transport)


def _fake_hysplit(monkeypatch) -> dict:
    """Stand in for writing the inputs, running hycs_std, and reading its particles; return what they were given."""
    from stilt.transport.hysplit import model as model_module

    seen: dict = {}

    def write_inputs(workdir, receptor, config, met_files):
        workdir.mkdir(parents=True)
        seen.update(workdir=workdir, met_files=met_files)

    def run_hycs_std(workdir, timeout):
        seen["timeout"] = timeout
        (workdir / "stilt.log").write_text("")
        (workdir / "PARTICLE_STILT.DAT").write_text("")

    monkeypatch.setattr(model_module, "write_inputs", write_inputs)
    monkeypatch.setattr(model_module, "_run_hycs_std", run_hycs_std)
    monkeypatch.setattr(
        model_module,
        "read_particle_dat",
        lambda path, columns: pd.DataFrame({"indx": [1]}),
    )
    return seen


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
    seen = _fake_hysplit(monkeypatch)
    source = [tmp_path / "archive" / "20230101_12"]
    cropped = [tmp_path / "crops" / "20230101_12"]
    params = HysplitConfig(n_hours=-1, hnf_plume=False)

    result = HysplitModel().run(
        point_receptor,
        params,
        _FakeMet(source, cropped),  # type: ignore[arg-type]
        tmp_path / "work",
        timeout=30,
    )

    assert isinstance(result, ModelRun)
    assert result.met_files == source  # the record names the source files
    assert seen["met_files"] == cropped  # HYSPLIT reads the crops
    assert seen["workdir"] == tmp_path / "work"
    assert seen["timeout"] == 30


def test_run_particles_goes_through_the_model_the_settings_name(
    tmp_path, point_receptor, monkeypatch
):
    calls: list[dict] = []

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir, timeout=None):
            calls.append({"receptor": receptor, "timeout": timeout, "workdir": workdir})
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
    variant = Variant(
        name="hrrr",
        group="hrrr",
        met="hrrr",
        met_config=met_config,
        transport=HysplitConfig(n_hours=-1, numpar=1, hnf_plume=False),
        model=ModelInfo(version="v5.1.0"),
    )
    sim = Simulation(point_receptor, variant, Output(tmp_path / "output"))

    traj = run_particles(sim, met=object(), workdir=tmp_path / "work", timeout=45)  # type: ignore[arg-type]

    assert asked == ["hysplit"]
    assert calls[0]["receptor"] == point_receptor
    assert calls[0]["timeout"] == 45  # the timeout reaches the model
    assert len(traj) == 1
    assert sim.has_particles
    assert sim.met_files == [tmp_path / "met_file"]
