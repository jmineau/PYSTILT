"""Tests for a transport model as a variant axis: a variant may run another model."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict
from runtraj import models
from runtraj.config import ProjectConfig
from runtraj.identity import read_run_settings, settings_hash
from runtraj.models import ModelInfo, TransportConfig, TransportModel

from stilt.footprint import FOOTPRINT_STAGE

from ..fixtures.factories import make_met_config

GRID = {"xmin": -112, "xmax": -111, "ymin": 40, "ymax": 41, "xres": 0.1, "yres": 0.1}


class ToyConfig(TransportConfig):
    """A second model's config: its own parameters, and the base class for the rest."""

    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"build_dir"})

    nparticles: int = 100
    build_dir: str | None = None


class ToyMet(BaseModel):
    """The toy model's met: it reads no files, and accepts the keys of a HYSPLIT met it shares."""

    model_config = ConfigDict(extra="allow")

    def settings(self) -> dict[str, Any]:
        return {"source": "toy"}


class ToyModel(TransportModel):
    name = "toy"
    config_class = ToyConfig
    met_config_class = ToyMet

    def sha256(self, config):
        return None

    def version(self, config: ToyConfig) -> str:
        return "1.0"

    def data_files(self, config: ToyConfig) -> dict[str, str] | None:
        return None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setitem(models.MODELS, "toy", ToyModel)


def _mets(tmp_path):
    return {"hrrr": make_met_config(tmp_path / "met")}


def test_a_variant_may_run_another_model(tmp_path, toy):
    config = ProjectConfig(
        mets=_mets(tmp_path),
        models={"hysplit": {"n_hours": -24, "numpar": 500}, "toy": {"n_hours": -24}},
        stages={"footprint": {"class": FOOTPRINT_STAGE, "grid": GRID}},
        variants={
            "hrrr": {"model": "hysplit"},
            "toy": {"model": "toy", "met": "hrrr", "nparticles": 10},
        },
    )
    variants = config.resolve()
    hrrr, toy_variant = variants["hrrr"], variants["toy"]
    assert hrrr.model == "hysplit" and hrrr.transport.numpar == 500
    # Another model's variant gives its own parameters and gets the same stages.
    assert toy_variant.model == "toy" and toy_variant.transport.nparticles == 10
    assert toy_variant.stages.get("footprint") == hrrr.stages.get("footprint")
    assert variants["toy"].build == ModelInfo(name="toy", version="1.0")
    assert variants["toy"].result_hash != variants["hrrr"].result_hash
    recorded = variants["toy"].run_settings
    assert recorded["model"]["name"] == "toy" and recorded["nparticles"] == 10
    # A stored record reads back through the model's own config class.
    assert settings_hash(read_run_settings(recorded)) == variants["toy"].result_hash
