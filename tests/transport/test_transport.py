"""Tests for the transport model boundary."""

from __future__ import annotations

import pandas as pd
import pytest
from runtraj.meteorology import run_window
from runtraj.models import MODELS, ModelRun, TransportConfig, TransportModel, get_model
from runtraj.models.hysplit import HysplitConfig, HysplitMetConfig

from ..fixtures.factories import make_met_config


def test_stilt_r_is_no_built_in_model():
    # PYSTILT's model of imported runs is named by its import path.
    assert "stilt-r" not in MODELS
    with pytest.raises(
        ValueError,
        match=r"Unknown transport model 'stilt-r'",
    ):
        get_model("stilt-r")


def test_stilt_r_is_a_model_named_by_import_path_that_cannot_run(point_release):
    from runtraj.exceptions import SimulationError

    from stilt.migration import STILT_R_MODEL, StiltRMetConfig, StiltRModel

    model = get_model(STILT_R_MODEL)
    assert isinstance(model, StiltRModel)
    met = StiltRMetConfig(met_path="/data/hrrr")
    assert met.settings() == {"met_path": "/data/hrrr"}
    window = run_window(point_release.time, -24)
    with pytest.raises(SimulationError) as caught:
        model.run(point_release, None, met, window)
    assert caught.value.reason == "NOT_RUNNABLE"


def test_stilt_r_settings_have_no_defaults():
    from stilt.migration import STILT_R_MODEL, StiltRConfig

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
    assert get_model(STILT_R_MODEL).version(config) == "1d936c1"
    assert get_model(STILT_R_MODEL).sha256(config) is None
    with pytest.raises(ValueError, match="missing: version"):
        StiltRConfig.model_validate(settings)


def test_stilt_r_settings_are_particle_mode_only():
    """STILT-R ran HYSPLIT's particle mode only, so a stilt-r block refuses the others."""
    from stilt.migration import StiltRConfig

    settings = {**HysplitConfig(n_hours=-24).settings(), "version": "1d936c1"}
    StiltRConfig.model_validate({**settings, "mode": "particle"})
    for mode in ("concentration", "trajectory"):
        with pytest.raises(ValueError, match=f"mode: {mode} is not read by a stilt-r"):
            StiltRConfig.model_validate({**settings, "mode": mode})
    # ichem is HYSPLIT's own check: a particle run takes 8 or none.
    StiltRConfig.model_validate({**settings, "ichem": 8})
    with pytest.raises(ValueError, match="ichem=2 is not a setting of mode: particle"):
        StiltRConfig.model_validate({**settings, "ichem": 2})


def _fake_hysplit(monkeypatch) -> dict:
    """Stand in for writing the inputs, running hycs_std, and reading its particles; return what they were given."""
    from runtraj.models.hysplit import model as model_module

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


class _EchoModel(TransportModel):
    """A transport model that records what it was given and returns one particle."""

    name = "echo"
    config_class = HysplitConfig
    met_config_class = HysplitMetConfig
    seen: dict = {}

    def sha256(self, config):
        return None

    def version(self, config):
        return "1"

    def data_files(self, config):
        return None

    def run(self, release, config, met, window, workdir=None, timeout=None):
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
    from runtraj import models

    monkeypatch.setitem(models.MODELS, "echo", _EchoModel)
    return _EchoModel


def _met(tmp_path) -> dict:
    return make_met_config(tmp_path)


def test_run_trajectories_returns_the_particles_and_removes_its_workdir(
    tmp_path, point_release, echo
):
    from stilt import run_trajectories

    particles = run_trajectories(
        point_release, _met(tmp_path), model="echo", n_hours=-24, numpar=50, timeout=9
    )
    # The footprint applies the near-field correction, not the run.
    assert "foot_no_hnf_dilution" not in particles.columns

    assert particles["particle"].unique().tolist() == [1]
    # time is the release time plus age, as a table read from a file has.
    expected = pd.Timestamp(point_release.time) + pd.to_timedelta(
        particles["age"], unit="min"
    )
    pd.testing.assert_series_equal(particles["time"], expected, check_names=False)
    assert echo.seen["config"].numpar == 50
    assert echo.seen["timeout"] == 9
    assert echo.seen["met"].directory == tmp_path.resolve()
    assert echo.seen["workdir"] is None  # the model makes its own when it needs one
    start, end = echo.seen["window"]
    assert end == point_release.time and (end - start).total_seconds() == 24 * 3600


@pytest.mark.parametrize("setting", ["footprint_depth", "pbl_height", "pbl_height_min"])
def test_a_transport_config_has_none_of_the_footprints_foot_settings(setting):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=setting):
        TransportConfig(n_hours=-24, **{setting: 100.0})
