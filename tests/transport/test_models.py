"""Tests for a transport model as a variant axis: a variant may run another model."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from stilt import transport
from stilt.config import ProjectConfig
from stilt.identity import read_run_settings, settings_hash
from stilt.transport import ModelInfo, TransportConfig

from ..fixtures.factories import make_met_config

GRID = {"xmin": -112, "xmax": -111, "ymin": 40, "ymax": 41, "xres": 0.1, "yres": 0.1}


class ToyConfig(TransportConfig):
    """A second model's config: its own parameters, and the base class for the rest."""

    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"build_dir"})

    nparticles: int = 100
    build_dir: str | None = None


class ToyModel:
    name = "toy"
    config_class = ToyConfig

    def version(self, config: ToyConfig) -> str:
        return "1.0"

    def data_files(self, config: ToyConfig) -> dict[str, str] | None:
        return None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setitem(transport.MODELS, "toy", ToyModel)


def _mets(tmp_path):
    return {"hrrr": make_met_config(tmp_path / "met", file_tres="6h")}


def test_a_variant_may_run_another_model(tmp_path, toy):
    config = ProjectConfig(
        mets=_mets(tmp_path),
        grid=GRID,
        numpar=500,
        variants={"hrrr": {}, "toy": {"model": "toy", "nparticles": 10}},
    )
    variants = config.resolve()
    hrrr, toy_variant = variants["hrrr"], variants["toy"]
    assert hrrr.model.name == "hysplit" and hrrr.transport.numpar == 500
    # Another model's variant gives its own parameters and inherits the footprint.
    assert toy_variant.model.name == "toy" and toy_variant.transport.nparticles == 10
    assert toy_variant.footprint == hrrr.footprint
    assert variants["toy"].model == ModelInfo(name="toy", version="1.0")
    assert variants["toy"].particles_hash != variants["hrrr"].particles_hash
    recorded = variants["toy"].run_settings
    assert recorded["model"]["name"] == "toy" and recorded["nparticles"] == 10
    # A stored record reads back through the model's own config class.
    assert settings_hash(read_run_settings(recorded)) == variants["toy"].particles_hash


def test_another_models_variant_inherits_the_shared_parameters_only(tmp_path, toy):
    """n_hours, seed, hnf_plume, and veght are every model's; numpar and krand are HYSPLIT's."""
    config = ProjectConfig(
        mets=_mets(tmp_path),
        n_hours=-6,
        numpar=300,
        krand=2,
        variants={"hrrr": {}, "toy": {"model": "toy", "hnf_plume": False}},
    )
    toy_variant = config.resolve()["toy"]
    assert toy_variant.transport.n_hours == -6  # inherited from the defaults
    assert toy_variant.transport.hnf_plume is False
    assert "numpar" not in type(toy_variant.transport).model_fields
    for hysplits_own in ("krand", "numpar"):
        with pytest.raises(ValueError, match=f"'{hysplits_own}' is not a setting"):
            ProjectConfig(
                mets=_mets(tmp_path),
                variants={"toy": {"model": "toy", hysplits_own: 2}},
            )


def test_the_projects_model_takes_its_parameters_at_the_top(tmp_path, toy):
    config = ProjectConfig(
        mets=_mets(tmp_path), model="toy", nparticles=7, variants={"hrrr": {}}
    )
    assert config.transport.nparticles == 7
    assert config.resolve()["hrrr"].model.name == "toy"
    assert (
        ProjectConfig(
            mets=_mets(tmp_path), model="toy", seed=7, variants={"hrrr": {}}
        ).transport.seed
        == 7
    )  # a shared parameter
    with pytest.raises(ValueError, match="'varsiwant' is not a setting"):
        ProjectConfig(
            mets=_mets(tmp_path), model="toy", varsiwant=["time"], variants={"hrrr": {}}
        )


def test_an_unknown_model_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown transport model 'nope'"):
        ProjectConfig(mets=_mets(tmp_path), model="nope", variants={"hrrr": {}})
    with pytest.raises(ValueError, match="Unknown transport model 'nope'"):
        ProjectConfig(mets=_mets(tmp_path), variants={"hrrr": {"model": "nope"}})


def test_hysplit_is_the_default_model_and_its_parameters_are_flat(tmp_path):
    config = ProjectConfig(
        mets=_mets(tmp_path), numpar=300, seed=4, krand=2, variants={"hrrr": {}}
    )
    assert config.model == "hysplit"
    assert config.transport.numpar == 300
    text = config.to_yaml()
    assert "numpar: 300" in text and "model:" not in text
    assert ProjectConfig.from_yaml(_write(tmp_path, text)).transport.seed == 4


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_a_model_config_inherits_the_recorded_settings_and_realizations():
    config = ToyConfig(n_hours=-6, seed=5, nparticles=10, build_dir="/opt/toy")

    assert config.settings() == {
        "n_hours": -6,
        "hnf_plume": True,
        "veght": 0.5,
        "seed": 5,
        "nparticles": 10,
    }
    assert [r.seed for r in config.realizations(3)] == [5, 6, 7]
    assert [r.seed for r in ToyConfig().realizations(2)] == [None, None]
    with pytest.raises(ValueError, match="extra"):
        ToyConfig(unknown=1)
