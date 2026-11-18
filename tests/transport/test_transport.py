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
from stilt.transport import ModelRun, TransportConfig, get_model
from stilt.transport.hysplit import HysplitConfig, HysplitModel, Met, MetConfig
from tests.fixtures.factories import blocks

from ..fixtures.factories import make_met_config, make_met_files, make_variant


def test_get_model_returns_hysplit_by_name():
    assert isinstance(get_model("hysplit"), HysplitModel)
    assert get_model("hysplit").name == "hysplit"


def test_get_model_imports_a_model_by_its_path():
    model = get_model("stilt.transport.hysplit.HysplitModel")
    assert isinstance(model, HysplitModel)
    with pytest.raises(
        ImportError,
        match="config.yaml names nopkg.models.Missing, but nopkg is not "
        "installed here. Install nopkg and run this again.",
    ):
        get_model("nopkg.models.Missing")


def test_get_model_names_the_models_it_knows():
    with pytest.raises(
        ValueError, match=r"Unknown transport model 'flexpart'.*hysplit"
    ):
        get_model("flexpart")


def test_stilt_r_is_a_built_in_model_that_cannot_run(point_receptor):
    from stilt.exceptions import SimulationError
    from stilt.transport.stilt_r import StiltRMetConfig, StiltRModel

    model = get_model("stilt-r")
    assert isinstance(model, StiltRModel)
    met = StiltRMetConfig(met_path="/data/hrrr")
    assert met.settings() == {"met_path": "/data/hrrr"}
    window = run_window(point_receptor.time, -24)
    with pytest.raises(SimulationError) as caught:
        model.run(point_receptor, None, met, window)
    assert caught.value.reason == "NOT_RUNNABLE"


def test_stilt_r_settings_have_no_defaults():
    from stilt.transport.stilt_r import StiltRConfig

    settings = HysplitConfig(n_hours=-24).settings()
    with pytest.raises(ValueError, match="missing: numpar"):
        StiltRConfig.model_validate(
            {k: v for k, v in settings.items() if k != "numpar"}
        )
    config = StiltRConfig.model_validate({**settings, "version": "1d936c1"})
    with pytest.raises(ValueError, match="valid dictionary"):
        StiltRConfig.model_validate(["not", "settings"])
    # The version is a setting: it tells imports of two STILT-R versions apart.
    assert config.settings() == {**settings, "version": "1d936c1"}
    assert get_model("stilt-r").version(config) == "1d936c1"
    assert get_model("stilt-r").sha256(config) is None
    with pytest.raises(ValueError, match="missing: version"):
        StiltRConfig.model_validate(settings)


def test_hysplit_model_build_is_the_hycs_std_on_path_or_in_directory(tmp_path):
    import hashlib

    from stilt.transport.hysplit import find_hycs_std

    from ..fixtures.factories import make_hysplit_dir

    model = HysplitModel()
    on_path = find_hycs_std()
    assert model.sha256(HysplitConfig(n_hours=-1)) == (
        hashlib.sha256(on_path.read_bytes()).hexdigest()
    )

    folder = make_hysplit_dir(tmp_path / "hysplit", "hysplit.v9.9.9")
    exe = folder / "exec" / "hycs_std"
    config = HysplitConfig(n_hours=-1, directory=folder)
    assert model.version(config) == "hysplit.v9.9.9"
    assert model.sha256(config) == hashlib.sha256(exe.read_bytes()).hexdigest()


def test_the_settings_record_the_model_that_makes_the_particles(tmp_path):
    met = make_met_config(tmp_path)
    variant = (
        ProjectConfig(**blocks(mets={"hrrr": met}, n_hours=-1, variants={"hrrr": {}}))
    ).resolve()["hrrr"]
    model = get_model(variant.model)
    assert variant.build.version == model.version(variant.transport)
    assert variant.run_settings["model"]["sha256"] == model.sha256(variant.transport)


def _fake_hysplit(monkeypatch) -> dict:
    """Stand in for writing the inputs, running hycs_std, and reading its particles; return what they were given."""
    from stilt.transport.hysplit import model as model_module

    seen: dict = {}

    def write_inputs(workdir, release_time, locations, config, met_files, kmsl):
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
    params = HysplitConfig(n_hours=-1)
    make_met_files(tmp_path / "archive", point_receptor.time, params.n_hours)
    met = make_met_config(tmp_path / "archive")
    window = run_window(point_receptor.time, params.n_hours)
    source = Met("met", met).files_for(window)
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
                    "age": [-60.0],
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
    variant = make_variant(met_config=met_config, n_hours=-1, numpar=1)
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

    def sha256(self, config):
        return None

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
                "age": [-1.0, config.n_hours * 60.0],
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
    return make_met_config(tmp_path)


def test_run_trajectories_returns_the_particles_and_removes_its_workdir(
    tmp_path, point_receptor, echo
):
    from stilt import run_trajectories

    particles = run_trajectories(
        point_receptor, _met(tmp_path), model="echo", n_hours=-24, numpar=50, timeout=9
    )
    # The footprint applies the near-field correction, not the run.
    assert "foot_no_hnf_dilution" not in particles.columns

    assert particles["particle"].unique().tolist() == [1]
    # time is the receptor time plus age, as a table read from a file has.
    expected = pd.Timestamp(point_receptor.time) + pd.to_timedelta(
        particles["age"], unit="min"
    )
    pd.testing.assert_series_equal(particles["time"], expected, check_names=False)
    assert echo.seen["config"].numpar == 50
    assert echo.seen["timeout"] == 9
    assert echo.seen["met"].directory == tmp_path.resolve()
    assert echo.seen["workdir"] is None  # the model makes its own when it needs one
    start, end = echo.seen["window"]
    assert end == point_receptor.time and (end - start).total_seconds() == 24 * 3600


def test_run_trajectories_keeps_a_workdir_it_was_given(tmp_path, point_receptor, echo):
    from stilt import run_trajectories

    run_trajectories(
        point_receptor,
        _met(tmp_path),
        model="echo",
        n_hours=-24,
        workdir=tmp_path / "w",
    )
    assert (tmp_path / "w" / "CONTROL").exists()
    with pytest.raises(ValueError, match="not empty"):
        run_trajectories(
            point_receptor,
            _met(tmp_path),
            model="echo",
            n_hours=-24,
            workdir=tmp_path / "w",
        )


def test_run_trajectories_checks_the_models_parameters(tmp_path, point_receptor, echo):
    from stilt import run_trajectories

    with pytest.raises(ValueError, match="nparticles"):
        run_trajectories(
            point_receptor, _met(tmp_path), model="echo", n_hours=-24, nparticles=5
        )
    with pytest.raises(ValueError, match="n_hours"):
        run_trajectories(point_receptor, _met(tmp_path), model="echo")


def test_finish_leaves_foot_for_the_footprint_to_correct():
    """The near-field correction is the footprint's: a run's particles keep the model's foot."""
    from stilt.transport import _finish

    particles = _positions_only().assign(
        foot=[1e-5, 2e-5], dens=1.2, samt=1.0, sigw=0.1, tlgr=10.0, mlht=500.0
    )
    run = _finish(
        ModelRun(particles, met_files=[]), _receptor(), TransportConfig(n_hours=-24)
    )

    assert run.particles["foot"].tolist() == [1e-5, 2e-5]
    assert "foot_no_hnf_dilution" not in run.particles.columns


def _positions_only() -> pd.DataFrame:
    """Particles of a transport model that writes only positions."""
    return pd.DataFrame(
        {
            "particle": [1, 1],
            "age": [-1.0, -2.0],
            "lon": [-111.9, -112.0],
            "lat": [40.7, 40.6],
            "zagl": [5.0, 900.0],
        }
    )


def test_finish_adds_no_foot_to_a_model_that_writes_none(tmp_path):
    """A particle file holds what the model wrote: the footprint computes foot when it needs one."""
    from stilt.transport import _finish

    run = _finish(
        ModelRun(_positions_only(), log="model log\n", met_files=[tmp_path / "met"]),
        _receptor(),
        TransportConfig(n_hours=-24),
    )

    assert "foot" not in run.particles.columns
    assert run.log == "model log\n"


@pytest.mark.parametrize("setting", ["footprint_depth", "pbl_height", "pbl_height_min"])
def test_a_transport_config_has_none_of_the_footprints_foot_settings(setting):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=setting):
        TransportConfig(n_hours=-24, **{setting: 100.0})


def _receptor():
    from stilt.receptors import PointReceptor

    return PointReceptor(
        time="2023-01-01 12:00", longitude=-111.9, latitude=40.7, altitude=5.0
    )
