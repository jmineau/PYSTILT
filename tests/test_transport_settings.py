"""Tests for stilt.config.transport: what identifies a run."""

import pytest
import yaml

from stilt.config import (
    MetConfig,
    MetSettings,
    ModelInfo,
    ProjectConfig,
    STILTParams,
    TransportSettings,
    hysplit_version,
    settings_hash,
)
from stilt.output import Output


def _met(tmp_path, **overrides) -> MetConfig:
    return MetConfig(
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="6h",
        **overrides,
    )


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def test_settings_hash_ignores_key_order_and_number_spelling():
    a = {"numpar": 1000, "ziscale": 1.0, "grid": {"xres": 0.01, "yres": 0.01}}
    b = {"grid": {"yres": 0.01, "xres": 0.01}, "ziscale": 1, "numpar": 1000.0}
    assert settings_hash(a) == settings_hash(b)
    assert settings_hash(a) != settings_hash({**a, "ziscale": 0.8})


# ---------------------------------------------------------------------------
# model version
# ---------------------------------------------------------------------------


def test_bundled_hysplit_version():
    assert hysplit_version() == "v5.1.0"


def test_custom_build_needs_a_version_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="version"):
        hysplit_version(tmp_path)
    (tmp_path / "version").write_text("v5.3.2+t0-rows\n")
    assert hysplit_version(tmp_path) == "v5.3.2+t0-rows"


# ---------------------------------------------------------------------------
# MetSettings
# ---------------------------------------------------------------------------


def test_met_settings_are_the_config_without_its_directories(tmp_path):
    met = _met(tmp_path, subgrid_dir=tmp_path / "sub", n_min=2)
    settings = met.settings()
    assert isinstance(met, MetSettings)
    assert type(settings) is MetSettings
    assert settings.n_min == 2
    assert (
        "directory" not in settings.model_dump()
        and "subgrid_dir" not in settings.model_dump()
    )
    moved = met.model_copy(update={"directory": tmp_path / "elsewhere"})
    assert moved.settings() == settings


# ---------------------------------------------------------------------------
# TransportSettings
# ---------------------------------------------------------------------------


def test_identity_leaves_out_what_changes_no_particle(tmp_path):
    met = _met(tmp_path)
    base = TransportSettings.build(STILTParams(numpar=100), met)
    identity = base.identity()
    for name in ("timeout", "rm_dat", "exe_dir"):
        assert name not in identity
    assert identity["met"] == met.settings().model_dump(mode="json")
    assert identity["model"] == {"name": "hysplit", "version": "v5.1.0"}
    assert identity["maxpar"] == 100  # unset maxpar is numpar, as HYSPLIT receives it

    same = TransportSettings.build(
        STILTParams(numpar=100, timeout=60, rm_dat=False), met
    )
    assert same.hash == base.hash
    assert (
        TransportSettings.build(STILTParams(numpar=100, maxpar=100), met).hash
        == base.hash
    )
    assert TransportSettings.build(STILTParams(numpar=200), met).hash != base.hash
    other_model = TransportSettings.build(
        STILTParams(numpar=100), met, model=ModelInfo(version="v5.3.2+t0-rows")
    )
    assert other_model.hash != base.hash


def test_stored_settings_re_validate_to_the_same_hash(tmp_path):
    settings = TransportSettings.build(STILTParams(numpar=100), _met(tmp_path))
    stored = settings.identity()
    assert TransportSettings.model_validate(stored).hash == settings.hash

    # A field added later, with a default: an older file without it still matches.
    del stored["capemin"]
    assert TransportSettings.model_validate(stored).hash == settings.hash

    # A default that changed meaning: the stored value differs, so the run differs.
    stored["capemin"] = -2
    assert TransportSettings.model_validate(stored).hash != settings.hash


def test_variants_that_differ_only_in_footprint_fields_share_settings(tmp_path):
    met = _met(tmp_path)
    config = ProjectConfig(
        mets={"hrrr": met},
        grid={
            "xmin": -112,
            "xmax": -111,
            "ymin": 40,
            "ymax": 41,
            "xres": 0.1,
            "yres": 0.1,
        },
        numpar=100,
        variants={
            "hrrr": {},
            "hrrr-smooth": {"smooth_factor": 0.5},
            "hrrr-coarse": {"grid": {"xres": 0.5, "yres": 0.5}},
            "hrrr-err": {
                "siguverr": 2.0,
                "tluverr": 100.0,
                "zcoruverr": 200.0,
                "horcoruverr": 10.0,
            },
        },
    )
    variants = config.resolve_variants()
    hashes = {name: v.transport.hash for name, v in variants.items()}
    assert hashes["hrrr"] == hashes["hrrr-smooth"] == hashes["hrrr-coarse"]
    assert hashes["hrrr-err"] != hashes["hrrr"]


def test_output_finds_a_run_whose_stored_settings_predate_a_field(tmp_path):
    out = Output(tmp_path / "output")
    settings = TransportSettings.build(STILTParams(numpar=100), _met(tmp_path))
    run = out.particles("hrrr", settings)

    record = yaml.safe_load((run.path / "_settings.yaml").read_text())
    del record["settings"]["capemin"]  # as if written before the field existed
    (run.path / "_settings.yaml").write_text(yaml.safe_dump(record))

    found = out.find_particles(settings)
    assert found is not None and found.path == run.path
    assert out.particles("hrrr-renamed", settings).path == run.path
