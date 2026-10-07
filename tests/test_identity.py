"""Tests for stilt.identity: what identifies a run and a footprint, and reading it back."""

import shutil
from dataclasses import replace

import pytest
import yaml

from stilt.config import ProjectConfig, Variant
from stilt.identity import read_run_settings, run_settings, settings_hash
from stilt.meteorology import MetConfig
from stilt.output import Output
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig
from stilt.transport.hysplit.model import hysplit_version

from .fixtures.factories import make_met_config

GRID = {"xmin": -112, "xmax": -111, "ymin": 40, "ymax": 41, "xres": 0.1, "yres": 0.1}


def _met(tmp_path, **overrides) -> MetConfig:
    return make_met_config(tmp_path / "met", file_tres="6h", **overrides)


def _variant(tmp_path, met: MetConfig | None = None, **transport) -> Variant:
    """The resolved ``hrrr`` variant of a one-met project with these transport fields."""
    config = ProjectConfig(
        mets={"hrrr": met or _met(tmp_path)}, **{"variants": {"hrrr": {}}, **transport}
    )
    return config.resolve()["hrrr"]


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
# what a run's settings hold
# ---------------------------------------------------------------------------


def test_a_runs_met_is_recorded_without_its_directories(tmp_path):
    met = _met(tmp_path, subgrid_dir=tmp_path / "sub", n_min=2)
    recorded = run_settings(
        HysplitConfig(), met, ModelInfo(name="hysplit", version="v5.1.0"), None
    )
    assert not {"directory", "subgrid_dir", "n_min", "download_from"} & set(
        recorded["met"]
    )
    moved = met.model_copy(update={"directory": tmp_path / "elsewhere"})
    assert run_settings(
        HysplitConfig(), moved, ModelInfo(name="hysplit", version="v5.1.0"), None
    ) == (recorded)


def test_run_settings_leave_out_what_changes_no_particle(tmp_path):
    base = _variant(tmp_path, numpar=100)
    recorded = base.run_settings
    assert "exe_dir" not in recorded
    assert recorded["model"] == {"name": "hysplit", "version": "v5.1.0"}
    assert recorded["maxpar"] == 100  # unset maxpar is numpar, as HYSPLIT receives it

    assert (
        _variant(tmp_path, numpar=100, maxpar=100).particles_hash == base.particles_hash
    )
    assert _variant(tmp_path, numpar=200).particles_hash != base.particles_hash
    other_model = replace(
        base, model=ModelInfo(name="hysplit", version="v5.3.2+t0-rows")
    )
    assert other_model.particles_hash != base.particles_hash


def test_where_met_is_downloaded_from_does_not_identify_the_run(tmp_path):
    base = _variant(tmp_path, numpar=100)
    for override in ({"download_from": "ftp"}, {"n_min": 3}):
        other = _variant(tmp_path, met=_met(tmp_path, **override), numpar=100)
        assert other.particles_hash == base.particles_hash


def test_nested_and_flat_ziscale_are_one_run(tmp_path):
    """STILT-R's ``[[0.8, 0.9]]`` and ``[0.8, 0.9]`` are the same factors and the same hash."""
    flat = _variant(tmp_path, ziscale=[0.8, 0.9])
    nested = _variant(tmp_path, ziscale=[[0.8, 0.9]])
    assert nested.transport.ziscale == [0.8, 0.9]
    assert nested.particles_hash == flat.particles_hash
    with pytest.raises(ValueError, match="Per-simulation"):
        HysplitConfig(ziscale=[[0.8], [0.9]])


def test_variants_that_differ_only_in_footprint_fields_share_particles(tmp_path):
    config = ProjectConfig(
        mets={"hrrr": _met(tmp_path)},
        grid=GRID,
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
    variants = config.resolve()
    runs = {name: v.particles_hash for name, v in variants.items()}
    assert runs["hrrr"] == runs["hrrr-smooth"] == runs["hrrr-coarse"]
    assert runs["hrrr-err"] != runs["hrrr"]
    feet = {name: v.footprint_hash for name, v in variants.items()}
    assert len(set(feet.values())) == 4


# ---------------------------------------------------------------------------
# reading stored settings back
# ---------------------------------------------------------------------------


def test_stored_settings_read_back_to_the_same_hash(tmp_path):
    variant = _variant(tmp_path, numpar=100)
    stored = dict(variant.run_settings)
    assert settings_hash(read_run_settings(stored)) == variant.particles_hash

    # A field added later, with a default: an older record without it still matches.
    del stored["capemin"]
    assert settings_hash(read_run_settings(stored)) == variant.particles_hash

    # A default that changed meaning: the stored value differs, so the run differs.
    stored["capemin"] = -2
    assert settings_hash(read_run_settings(stored)) != variant.particles_hash


def _make_folder(directory, variant):
    """Make *variant*'s particles folder in the output *directory*, and return it."""
    out = Output(directory)
    out.write_log(variant, "202301011200_-111.85_40.77_5", "")
    return out.folder("particles", variant)


def _rewrite_record(folder, change) -> None:
    path = folder / "_settings.yaml"
    record = yaml.safe_load(path.read_text())
    change(record["settings"])
    path.write_text(yaml.safe_dump(record))


def test_a_folder_stored_with_download_settings_is_still_found(tmp_path):
    """Folders written while download_from and n_min were hashed are found by re-hashing."""
    variant = _variant(tmp_path, numpar=100)
    folder = _make_folder(tmp_path / "out", variant)
    _rewrite_record(folder, lambda s: s["met"].update(download_from="ftp", n_min=3))

    assert Output(tmp_path / "out").folder("particles", variant) == folder


def test_a_folder_with_a_setting_this_version_lacks_still_loads(tmp_path):
    """A setting removed after a folder was written is ignored, so the folder is found."""
    variant = _variant(tmp_path, numpar=100)
    folder = _make_folder(tmp_path / "out", variant)

    def add_removed(settings):
        settings["removed_setting"] = 3
        settings["model"]["removed_too"] = "x"

    _rewrite_record(folder, add_removed)
    assert Output(tmp_path / "out").folder("particles", variant) == folder


def test_output_finds_a_run_whose_stored_settings_predate_a_field(tmp_path):
    variant = _variant(tmp_path, numpar=100)
    run = _make_folder(tmp_path / "output", variant)
    _rewrite_record(run, lambda s: s.pop("capemin"))  # as if written before the field

    out = Output(tmp_path / "output")
    assert out.folder("particles", variant) == run
    renamed = replace(variant, name="hrrr-renamed")
    assert out.folder("particles", renamed) == run


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------


def _bundled_table(name: str):
    from stilt.transport.hysplit.driver import _bundled_data_dir

    return _bundled_data_dir() / name


def test_a_data_dir_of_bundled_copies_is_the_same_run(tmp_path):
    """Pointing data_dir at copies of the bundled tables changes nothing."""
    data = tmp_path / "tables"
    data.mkdir()
    shutil.copy(_bundled_table("LANDUSE.ASC"), data / "LANDUSE.ASC")
    base = _variant(tmp_path, numpar=100)
    same = _variant(tmp_path, numpar=100, data_dir=data)
    assert same.model.data_files is None
    assert "data_dir" not in same.run_settings
    assert same.particles_hash == base.particles_hash


def test_a_changed_data_table_is_recorded_and_makes_another_run(tmp_path):
    """A land-use table that differs from the bundled one changes the particles."""
    data = tmp_path / "tables"
    data.mkdir()
    (data / "LANDUSE.ASC").write_bytes(
        _bundled_table("LANDUSE.ASC").read_bytes() + b"\n"
    )
    base = _variant(tmp_path, numpar=100)
    other = _variant(tmp_path, numpar=100, data_dir=data)
    assert other.model.data_files is not None
    assert set(other.model.data_files) == {"LANDUSE.ASC"}
    assert other.run_settings["model"]["data_files"] == other.model.data_files
    assert other.particles_hash != base.particles_hash
