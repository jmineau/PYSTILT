"""Tests for a transport model as a variant axis: a variant may run another model."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any, ClassVar

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from stilt import transport
from stilt.config import FOOTPRINT_STAGE, ProjectConfig
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
    monkeypatch.setitem(transport.MODELS, "toy", ToyModel)


#: The toy model by its import path, as config.yaml names a model in another package.
TOY_PATH = f"{__name__}.ToyModel"


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
    assert variants["toy"].particles_hash != variants["hrrr"].particles_hash
    recorded = variants["toy"].run_settings
    assert recorded["model"]["name"] == "toy" and recorded["nparticles"] == 10
    # A stored record reads back through the model's own config class.
    assert settings_hash(read_run_settings(recorded)) == variants["toy"].particles_hash


def test_each_models_block_holds_its_own_defaults(tmp_path, toy):
    """A variant starts from its own model's block: nothing passes from one model to another."""
    config = ProjectConfig(
        mets=_mets(tmp_path),
        models={
            "hysplit": {"n_hours": -6, "numpar": 300, "krand": 2},
            "toy": {"n_hours": -12, "nparticles": 7},
        },
        variants={
            "hrrr": {"model": "hysplit"},
            "toy": {"model": "toy", "met": "hrrr"},
            "toy-long": {"model": "toy", "met": "hrrr", "n_hours": -48},
        },
    )
    variants = config.resolve()
    assert variants["hrrr"].transport.n_hours == -6
    assert variants["toy"].transport.n_hours == -12
    assert variants["toy"].transport.nparticles == 7
    assert variants["toy-long"].transport.n_hours == -48
    assert "numpar" not in type(variants["toy"].transport).model_fields
    for hysplits_own in ("krand", "numpar"):
        with pytest.raises(ValueError, match=f"'{hysplits_own}' is not a setting"):
            ProjectConfig(
                mets=_mets(tmp_path),
                models={"toy": {}},
                variants={"hrrr": {hysplits_own: 2}},
            )
        with pytest.raises(
            ValueError, match=f"models: toy: '{hysplits_own}' is not a setting"
        ):
            ProjectConfig(
                mets=_mets(tmp_path),
                models={"toy": {hysplits_own: 2}},
                variants={"hrrr": {}},
            )


def test_a_variant_names_its_model_when_there_are_several(tmp_path, toy):
    models = {"hysplit": {"n_hours": -24}, "toy": {"n_hours": -24}}
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr' must name its model \(one of \['hysplit', 'toy'\]\)",
    ):
        ProjectConfig(mets=_mets(tmp_path), models=models, variants={"hrrr": {}})
    with pytest.raises(
        ValueError, match=r"names the model 'nope', which models: does not have"
    ):
        ProjectConfig(
            mets=_mets(tmp_path), models=models, variants={"hrrr": {"model": "nope"}}
        )
    # With one model, a variant need not say it.
    only = ProjectConfig(
        mets=_mets(tmp_path), models={"toy": {"n_hours": -24}}, variants={"hrrr": {}}
    )
    assert only.resolve()["hrrr"].model == "toy"


def test_a_met_is_checked_by_the_model_that_reads_it(tmp_path, toy):
    """``met_config`` takes the name of a models block, and needs it when there are several."""
    one = ProjectConfig(
        mets=_mets(tmp_path), models={"toy": {"n_hours": -24}}, variants={"hrrr": {}}
    )
    assert isinstance(one.met_config("hrrr"), ToyMet)  # the only model

    both = ProjectConfig(
        mets=_mets(tmp_path),
        models={"hysplit": {"n_hours": -24}, "toy": {"n_hours": -24}},
        variants={"hrrr": {"model": "hysplit"}},
    )
    assert isinstance(both.met_config("hrrr", "toy"), ToyMet)
    assert not isinstance(both.met_config("hrrr", "hysplit"), ToyMet)
    with pytest.raises(
        ValueError,
        match=r"Name the model that reads met 'hrrr': one of \['hysplit', 'toy'\]",
    ):
        both.met_config("hrrr")
    with pytest.raises(ValueError, match="There is no model 'nope' under models:"):
        both.met_config("hrrr", "nope")


def test_with_several_models_a_met_no_variant_uses_is_not_checked(tmp_path, toy):
    """No variant pairs the spare met with a model, so there is no met config to check it with."""
    mets = {**_mets(tmp_path), "spare": {"subgrid_enable": "maybe"}}
    config = ProjectConfig(
        mets=mets,
        models={"hysplit": {"n_hours": -24}, "toy": {"n_hours": -24}},
        variants={"hrrr": {"model": "hysplit"}},
    )
    # HYSPLIT's met config refuses it once it is asked.
    with pytest.raises(ValueError, match=r"Met 'spare' \(hysplit\)"):
        config.met_config("spare", "hysplit")


def test_a_models_block_must_name_a_model(tmp_path, toy):
    with pytest.raises(
        ValueError, match="models: nope: 'nope' is not a built-in transport model"
    ):
        ProjectConfig(mets=_mets(tmp_path), models={"nope": {}}, variants={"hrrr": {}})
    with pytest.raises(ValueError, match="class must be the import path"):
        ProjectConfig(
            mets=_mets(tmp_path),
            models={"mine": {"class": "toy"}},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="models: has no block"):
        ProjectConfig(mets=_mets(tmp_path), models={}, variants={"hrrr": {}})


def test_a_model_has_one_block(tmp_path):
    """Two blocks that name one class are refused: the second set of settings is a variant."""
    with pytest.raises(
        ValueError,
        match=f"models: a and b are both {TOY_PATH}. A model has one block",
    ):
        ProjectConfig(
            mets=_mets(tmp_path),
            models={"a": {"class": TOY_PATH}, "b": {"class": TOY_PATH}},
            variants={"hrrr": {"model": "a"}},
        )


def test_hysplit_runs_when_no_model_is_named(tmp_path):
    config = ProjectConfig(mets=_mets(tmp_path), variants={"hrrr": {"n_hours": -24}})
    assert config.resolve()["hrrr"].model == "hysplit"
    assert "models:" not in config.to_yaml()  # only what was given is written

    given = ProjectConfig(
        mets=_mets(tmp_path),
        models={"hysplit": {"n_hours": -24, "numpar": 300, "seed": 4, "krand": 2}},
        variants={"hrrr": {}},
    )
    text = given.to_yaml()
    assert yaml.safe_load(text)["models"] == {
        "hysplit": {"n_hours": -24, "numpar": 300, "seed": 4, "krand": 2}
    }
    loaded = ProjectConfig.from_yaml(_write(tmp_path, text))
    assert loaded.resolve()["hrrr"].transport.seed == 4


def test_an_empty_models_block_runs_the_models_defaults(tmp_path):
    """``hysplit:`` with nothing under it is read as no settings."""
    path = _write(
        tmp_path,
        f"mets:\n  hrrr: {tmp_path / 'met'}\nmodels:\n  hysplit:\n"
        "variants:\n  hrrr: {n_hours: -24}\n",
    )
    variant = ProjectConfig.from_yaml(path).resolve()["hrrr"]
    assert variant.transport == type(variant.transport)(n_hours=-24)


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_a_model_config_inherits_the_recorded_settings_and_realizations():
    config = ToyConfig(n_hours=-6, seed=5, nparticles=10, build_dir="/opt/toy")

    assert config.settings() == {
        "n_hours": -6,
        "footprint_depth": 0.5,
        "seed": 5,
        "nparticles": 10,
    }
    assert [r.seed for r in config.realizations(3)] == [5, 6, 7]
    assert [r.seed for r in ToyConfig(n_hours=-6).realizations(2)] == [None, None]
    with pytest.raises(ValueError, match="extra"):
        ToyConfig(n_hours=-6, unknown=1)
    with pytest.raises(ValueError, match="n_hours"):
        ToyConfig()  # a config must say how long the run is


def test_a_model_is_named_by_its_import_path_without_registering_it(tmp_path):
    """The block's name is the variant's handle; ``class`` is what runs and what is recorded."""
    config = ProjectConfig(
        mets=_mets(tmp_path),
        models={"mine": {"class": TOY_PATH, "n_hours": -6, "nparticles": 3}},
        variants={"toy": {"model": "mine", "met": "hrrr"}},
    )
    variant = config.resolve()["toy"]
    assert variant.transport.nparticles == 3
    assert yaml.safe_load(config.to_yaml())["models"] == {
        "mine": {"class": TOY_PATH, "n_hours": -6, "nparticles": 3}
    }
    # The record names the import path, so another process imports it again.
    assert variant.build == ModelInfo(name=TOY_PATH, version="1.0")
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

                def sha256(self, config):
                    return None

                def version(self, config):
                    return "1.0"

                def data_files(self, config):
                    return None

                def run(self, receptor, config, met, window, workdir=None, timeout=None):
                    n = config.nparticles
                    particles = pd.DataFrame({
                        "particle": list(range(1, n + 1)) * 2,
                        "age": [0] * n + [-60] * n,
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
            receptor, met, model="toypkg.ToyModel", n_hours=-1, nparticles=4
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
