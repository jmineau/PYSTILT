"""Tests for a transport model as a variant axis: a variant may run another model."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

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


class ToyMet(BaseModel):
    """The toy model's met: it reads no files, and accepts the keys of a HYSPLIT met it shares."""

    model_config = ConfigDict(extra="allow")

    def settings(self) -> dict[str, Any]:
        return {"source": "toy"}


class ToyModel:
    name = "toy"
    config_class = ToyConfig
    met_config_class = ToyMet

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


#: The toy model by its import path, as config.yaml names a model in another package.
TOY_PATH = f"{__name__}.ToyModel"


def test_a_model_is_named_by_its_import_path_without_registering_it(tmp_path):
    config = ProjectConfig(
        mets=_mets(tmp_path), variants={"toy": {"model": TOY_PATH, "nparticles": 3}}
    )
    variant = config.resolve()["toy"]
    assert variant.transport.nparticles == 3
    # The record names the import path, so another process imports it again.
    assert variant.model == ModelInfo(name=TOY_PATH, version="1.0")
    stored = read_run_settings(variant.run_settings)
    assert settings_hash(stored) == variant.particles_hash


def test_a_record_of_a_model_not_installed_still_reads(tmp_path):
    """Another project's folder in a shared output must not take status down."""
    stored = {
        "n_hours": -6,
        "nparticles": 3,
        "model": {"name": "nopkg.models.Missing", "version": "1.0"},
        "met": {"source": "toy"},
    }
    with pytest.warns(UserWarning, match="could not be loaded"):
        assert read_run_settings(stored) == stored


def test_a_model_in_its_own_package_runs_from_a_fresh_interpreter(tmp_path):
    """No registration in the process: the import path alone finds the model."""
    package = tmp_path / "toypkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd
            from pydantic import BaseModel
            from stilt.transport import ModelRun, TransportConfig

            class ToyConfig(TransportConfig):
                nparticles: int = 2

            class ToyMet(BaseModel):
                weather: str

                def settings(self):
                    return {"weather": self.weather}

            class ToyModel:
                name = "toy"
                config_class = ToyConfig
                met_config_class = ToyMet

                def version(self, config):
                    return "1.0"

                def data_files(self, config):
                    return None

                def run(self, receptor, config, met, window, workdir=None, timeout=None):
                    n = config.nparticles
                    particles = pd.DataFrame({
                        "particle": list(range(1, n + 1)) * 2,
                        "time": [0] * n + [-60] * n,
                        "lon": [receptor.longitude] * 2 * n,
                        "lat": [receptor.latitude] * 2 * n,
                        "zagl": [receptor.altitude] * 2 * n,
                        "foot": [0.0] * n + [0.01] * n,
                    })
                    return ModelRun(particles=particles)
            """
        )
    )
    script = textwrap.dedent(
        """
        import stilt
        receptor = stilt.PointReceptor(
            time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, altitude=5
        )
        met = {"weather": "hrrr"}  # no directory: the toy reads no met files
        particles = stilt.run_trajectories(
            receptor, met, model="toypkg.ToyModel", n_hours=-1, nparticles=4, hnf_plume=False
        )
        print(len(particles), sorted(int(p) for p in particles["particle"].unique()))
        """
    )
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n{script}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "8 [1, 2, 3, 4]"
