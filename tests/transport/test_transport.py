"""Tests for the transport model boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from stilt.config import ProjectConfig
from stilt.execution.worker import run_particles
from stilt.meteorology import run_window
from stilt.output import Output
from stilt.particles import particles_metadata
from stilt.simulation import Simulation
from stilt.transport import ModelRun, get_model
from stilt.transport.hysplit import HysplitConfig, HysplitModel, Met, MetConfig

from ..fixtures.factories import make_met_config, make_met_files, make_variant


def test_get_model_returns_hysplit_by_name():
    assert isinstance(get_model("hysplit"), HysplitModel)
    assert get_model("hysplit").name == "hysplit"


def test_get_model_imports_a_model_by_its_path():
    model = get_model("stilt.transport.hysplit.HysplitModel")
    assert isinstance(model, HysplitModel)
    with pytest.raises(ImportError, match="Install the package"):
        get_model("nopkg.models.Missing")


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
    met = make_met_config(tmp_path)
    variant = (
        ProjectConfig(mets={"hrrr": met}, n_hours=-1, variants={"hrrr": {}})
    ).resolve()["hrrr"]
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
    monkeypatch.setattr(model_module, "run_hycs_std", run_hycs_std)
    monkeypatch.setattr(
        model_module,
        "read_particle_dat",
        lambda path, columns: pd.DataFrame({"particle": [1]}),
    )
    return seen


def test_hysplit_model_reads_the_met_in_place_and_records_the_source(
    tmp_path, point_receptor, monkeypatch
):
    seen = _fake_hysplit(monkeypatch)
    params = HysplitConfig(n_hours=-1, hnf_plume=False)
    make_met_files(tmp_path / "archive", point_receptor.time, params.n_hours)
    met = make_met_config(tmp_path / "archive")
    window = run_window(point_receptor.time, params.n_hours)
    source = Met("met", met).files_for(window, hour_after=True)
    cropped = [tmp_path / "crops" / path.name for path in source]
    for path, crop in zip(source, cropped, strict=True):
        crop.parent.mkdir(exist_ok=True)
        crop.write_bytes(path.read_bytes())
    # A cropped met hands HYSPLIT its crops in place of the source files.
    monkeypatch.setattr(Met, "readable", lambda self, files: cropped)

    result = HysplitModel().run(
        point_receptor, params, met, window, tmp_path / "work", timeout=30
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

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            calls.append({"receptor": receptor, "timeout": timeout, "workdir": workdir})
            particles = pd.DataFrame(
                {
                    "time": [-60.0],
                    "particle": [1.0],
                    "lon": [-111.9],
                    "lat": [40.7],
                    "zagl": [10.0],
                    "foot": [1e-5],
                }
            )
            return ModelRun(particles=particles, met_files=[tmp_path / "met_file"])

    asked: list[str] = []
    monkeypatch.setattr(
        "stilt.transport.get_model", lambda name: asked.append(name) or _Model()
    )
    met_config = make_met_config(tmp_path)
    variant = make_variant(met_config=met_config, n_hours=-1, numpar=1, hnf_plume=False)
    sim = Simulation(point_receptor, variant, Output(tmp_path / "output"))

    traj = run_particles(sim, workdir=tmp_path / "work", timeout=45)

    assert asked == ["hysplit"]
    assert calls[0]["receptor"] == point_receptor
    assert calls[0]["timeout"] == 45  # the timeout reaches the model
    assert len(traj) == 1
    assert sim.has_particles
    assert particles_metadata(sim.particles_path).met_files == [tmp_path / "met_file"]


class _EchoModel:
    """A transport model that records what it was given and returns one particle."""

    name = "echo"
    config_class = HysplitConfig
    met_config_class = MetConfig
    seen: dict = {}

    def version(self, config):
        return "1"

    def data_files(self, config):
        return None

    def run(self, receptor, config, met, window, workdir=None, timeout=None):
        if workdir is not None:
            (workdir / "CONTROL").write_text("")
        type(self).seen = {
            "config": config,
            "met": met,
            "workdir": workdir,
            "window": window,
            "timeout": timeout,
        }
        particles = pd.DataFrame(
            {
                "particle": [1, 1],
                "time": [-1.0, config.n_hours * 60.0],
                "lon": [-111.9, -112.0],
                "lat": [40.7, 40.6],
                "zagl": [5.0, 20.0],
            }
        )
        return ModelRun(particles, met_files=[])


@pytest.fixture
def echo(monkeypatch):
    from stilt import transport

    monkeypatch.setitem(transport.MODELS, "echo", _EchoModel)
    return _EchoModel


def _met(tmp_path) -> dict:
    return make_met_config(tmp_path, file_tres="6h")


def test_run_trajectories_returns_the_particles_and_removes_its_workdir(
    tmp_path, point_receptor, echo, caplog
):
    from stilt import run_trajectories

    with caplog.at_level("WARNING", logger="stilt.transport"):
        particles = run_trajectories(
            point_receptor, _met(tmp_path), model="echo", numpar=50, timeout=9
        )
    # The echo model writes none of the columns the near-field correction reads.
    assert "near-field correction skipped" in caplog.text
    assert "foot_no_hnf_dilution" not in particles.columns

    assert particles["particle"].unique().tolist() == [1]
    assert echo.seen["config"].numpar == 50
    assert echo.seen["timeout"] == 9
    assert echo.seen["met"].directory == tmp_path.resolve()
    assert echo.seen["workdir"] is None  # the model makes its own when it needs one
    start, end = echo.seen["window"]
    assert end == point_receptor.time and (end - start).total_seconds() == 24 * 3600


def test_run_trajectories_keeps_a_workdir_it_was_given(tmp_path, point_receptor, echo):
    from stilt import run_trajectories

    run_trajectories(
        point_receptor, _met(tmp_path), model="echo", workdir=tmp_path / "w"
    )
    assert (tmp_path / "w" / "CONTROL").exists()
    with pytest.raises(ValueError, match="not empty"):
        run_trajectories(
            point_receptor, _met(tmp_path), model="echo", workdir=tmp_path / "w"
        )


def test_run_trajectories_checks_the_models_parameters(tmp_path, point_receptor, echo):
    from stilt import run_trajectories

    with pytest.raises(ValueError, match="nparticles"):
        run_trajectories(point_receptor, _met(tmp_path), model="echo", nparticles=5)
