"""Tests for a transport model as a variant axis: a variant may run another model."""

from __future__ import annotations

from typing import Any, ClassVar, Self

import pytest
from pydantic import BaseModel, ConfigDict

from stilt import transport
from stilt.config import ProjectConfig
from stilt.identity import read_run_settings, settings_hash
from stilt.transport import ModelInfo
from stilt.variants import resolve

GRID = {"xmin": -112, "xmax": -111, "ymin": 40, "ymax": 41, "xres": 0.1, "yres": 0.1}


class ToyConfig(BaseModel):
    """A second model's config, with parameters of its own."""

    model_config = ConfigDict(extra="forbid")

    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"build_dir"})

    n_hours: int = -24
    nparticles: int = 100
    build_dir: str | None = None

    def settings(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude=set(self.UNRECORDED))

    def realizations(self, n: int) -> list[Self]:
        return [self.model_copy() for _ in range(n)]


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
    return {
        "hrrr": {
            "directory": tmp_path / "met",
            "file_format": "%Y%m%d_%H",
            "file_tres": "6h",
        }
    }


def test_a_variant_may_run_another_model(tmp_path, toy):
    config = ProjectConfig(
        mets=_mets(tmp_path),
        grid=GRID,
        numpar=500,
        variants={"hrrr": {}, "toy": {"model": "toy", "nparticles": 10}},
    )
    hrrr, toy_variant = config.variant_configs["hrrr"], config.variant_configs["toy"]
    assert hrrr.model == "hysplit" and hrrr.transport.numpar == 500
    # Another model's variant gives its own parameters and inherits the footprint.
    assert toy_variant.model == "toy" and toy_variant.transport.nparticles == 10
    assert toy_variant.footprint == hrrr.footprint

    variants = resolve(config)
    assert variants["toy"].model == ModelInfo(name="toy", version="1.0")
    assert variants["toy"].particles_hash != variants["hrrr"].particles_hash
    recorded = variants["toy"].run_settings
    assert recorded["model"]["name"] == "toy" and recorded["nparticles"] == 10
    # A stored record reads back through the model's own config class.
    assert settings_hash(read_run_settings(recorded)) == variants["toy"].particles_hash


def test_another_models_variant_takes_only_its_own_parameters(tmp_path, toy):
    with pytest.raises(ValueError, match="numpar"):
        ProjectConfig(
            mets=_mets(tmp_path),
            variants={"toy": {"model": "toy", "numpar": 5}},
        )


def test_the_projects_model_takes_its_parameters_at_the_top(tmp_path, toy):
    config = ProjectConfig(mets=_mets(tmp_path), model="toy", nparticles=7)
    assert config.transport.nparticles == 7
    assert config.variant_configs["hrrr"].model == "toy"
    with pytest.raises(ValueError, match="numpar"):
        ProjectConfig(mets=_mets(tmp_path), model="toy", numpar=7)


def test_an_unknown_model_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown transport model 'nope'"):
        ProjectConfig(mets=_mets(tmp_path), model="nope")
    with pytest.raises(ValueError, match="Unknown transport model 'nope'"):
        ProjectConfig(mets=_mets(tmp_path), variants={"hrrr": {"model": "nope"}})


def test_hysplit_is_the_default_model_and_its_parameters_are_flat(tmp_path):
    config = ProjectConfig(mets=_mets(tmp_path), numpar=300, seed=4, krand=2)
    assert config.model == "hysplit"
    assert config.transport.numpar == 300
    text = config.to_yaml()
    assert "numpar: 300" in text and "model:" not in text
    assert ProjectConfig.from_yaml(_write(tmp_path, text)).transport.seed == 4


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path
