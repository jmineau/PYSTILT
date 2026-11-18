"""Tests for stilt.identity: what identifies a run and a footprint, and reading it back."""

import hashlib
from dataclasses import replace

import pytest
import yaml

from stilt.config import ProjectConfig, Variant
from stilt.exceptions import HYSPLITNotFoundError
from stilt.identity import read_run_settings, run_settings, settings_hash
from stilt.output import Output
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig, MetConfig
from stilt.transport.hysplit.model import hysplit_version
from tests.fixtures.factories import blocks

from .conftest import STAND_IN_VERSION
from .fixtures.factories import (
    make_hysplit_dir,
    make_met_config,
    make_receptor,
    write_arl_file,
)

GRID = {"xmin": -112, "xmax": -111, "ymin": 40, "ymax": 41, "xres": 0.1, "yres": 0.1}


def _met(tmp_path, **overrides) -> MetConfig:
    return make_met_config(tmp_path / "met", **overrides)


def _variant(tmp_path, met: MetConfig | None = None, **transport) -> Variant:
    """The resolved ``hrrr`` variant of a one-met project with these transport fields."""
    config = ProjectConfig(
        **blocks(
            mets={"hrrr": met or _met(tmp_path)},
            **{"variants": {"hrrr": {}}, **transport},
        )
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
# the model build
# ---------------------------------------------------------------------------


def test_hysplit_version_is_what_hycs_std_prints(tmp_path):
    folder = make_hysplit_dir(tmp_path / "b", "hysplit.v5.2.2")
    assert hysplit_version(folder) == "hysplit.v5.2.2"


def test_hysplit_version_without_directory_is_of_the_hycs_std_on_path():
    assert hysplit_version() == STAND_IN_VERSION


def test_hysplit_version_is_none_when_hycs_std_prints_none(tmp_path):
    assert hysplit_version(make_hysplit_dir(tmp_path / "b", version=None)) is None


def test_hysplit_version_is_none_for_a_hycs_std_that_cannot_run_here(tmp_path):
    """Another platform's build: the version is recorded only, so it does not stop a run."""
    folder = make_hysplit_dir(tmp_path / "b")
    (folder / "exec" / "hycs_std").write_bytes(b"\x00not a program for this machine")
    assert hysplit_version(folder) is None


def test_hysplit_version_is_none_for_a_hycs_std_that_does_not_stop(
    tmp_path, monkeypatch
):
    from stilt.transport.hysplit import model

    monkeypatch.setattr(model, "_VERSION_TIMEOUT", 0.5)
    folder = make_hysplit_dir(tmp_path / "b", version=None, body="exec sleep 30\n")
    assert hysplit_version(folder) is None


def test_without_hycs_std_hysplit_version_says_what_to_set(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(HYSPLITNotFoundError, match="directory.*PATH"):
        hysplit_version()


def test_the_build_is_the_checksum_of_hycs_std(tmp_path):
    """Two builds that print one version are two runs; the version is recorded beside."""
    one = _variant(tmp_path, directory=make_hysplit_dir(tmp_path / "one", body="# 1\n"))
    two = _variant(tmp_path, directory=make_hysplit_dir(tmp_path / "two", body="# 2\n"))
    exe = tmp_path / "one" / "exec" / "hycs_std"
    assert one.build.sha256 == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert one.build.version == two.build.version == "hysplit.v5.1.0"
    assert one.run_settings["model"] == {
        "name": "hysplit",
        "sha256": one.build.sha256,
        "data_files": one.build.data_files,
    }
    assert one.build.data_files == two.build.data_files
    assert one.particles_hash != two.particles_hash


def test_moving_the_hysplit_folder_keeps_the_hash(tmp_path):
    one = _variant(tmp_path, directory=make_hysplit_dir(tmp_path / "one"))
    moved = _variant(tmp_path, directory=make_hysplit_dir(tmp_path / "moved"))
    assert "directory" not in one.run_settings
    assert one.particles_hash == moved.particles_hash


# ---------------------------------------------------------------------------
# what a run's settings hold
# ---------------------------------------------------------------------------


def test_a_runs_met_is_its_product_and_crop(tmp_path):
    met = _met(tmp_path, subgrid_dir=tmp_path / "sub")
    recorded = run_settings(
        HysplitConfig(n_hours=-24), met.settings(), ModelInfo(name="hysplit")
    )
    assert recorded["met"] == {"source": "HRRR", "crop": None}


def test_moving_or_renaming_the_met_files_keeps_the_hash(tmp_path):
    base = _variant(tmp_path, numpar=100)
    write_arl_file(tmp_path / "moved" / "hrrr.20240718.t00z")
    moved = _met(tmp_path).model_copy(update={"directory": tmp_path / "moved"})
    assert _variant(tmp_path, met=moved, numpar=100).particles_hash == (
        base.particles_hash
    )


def test_two_products_named_alike_are_two_runs(tmp_path):
    """An HRRR and a NAM archive with the same file names hash apart (R2-24)."""
    hrrr = _variant(tmp_path, numpar=100)
    write_arl_file(tmp_path / "nam" / "20240718_00", source="NAM")
    nam = _met(tmp_path).model_copy(update={"directory": tmp_path / "nam"})
    assert _variant(tmp_path, met=nam, numpar=100).particles_hash != (
        hrrr.particles_hash
    )


def test_a_single_run_records_no_realization(tmp_path):
    recorded = _variant(tmp_path).run_settings
    assert "realization" not in recorded
    assert "ensemble" not in recorded


def test_run_settings_leave_out_what_changes_no_particle(tmp_path):
    base = _variant(tmp_path, numpar=100)
    recorded = base.run_settings
    assert "directory" not in recorded
    assert set(recorded["model"]) == {"name", "sha256", "data_files"}
    # Unset, maxpar depends on the receptor (#237), so it is recorded unset,
    # and a set maxpar, which can stop a multipoint release, is another run.
    assert recorded["maxpar"] is None
    capped = _variant(tmp_path, numpar=100, maxpar=100)
    assert capped.run_settings["maxpar"] == 100
    assert capped.particles_hash != base.particles_hash
    assert _variant(tmp_path, numpar=200).particles_hash != base.particles_hash


def test_where_met_is_downloaded_from_does_not_identify_the_run(tmp_path):
    base = _variant(tmp_path, numpar=100)
    for override in ({"download_from": "ftp"},):
        other = _variant(tmp_path, met=_met(tmp_path, **override), numpar=100)
        assert other.particles_hash == base.particles_hash


def test_nested_and_flat_ziscale_are_one_run(tmp_path):
    """STILT-R's ``[[0.8, 0.9]]`` and ``[0.8, 0.9]`` are the same factors and the same hash."""
    flat = _variant(tmp_path, ziscale=[0.8, 0.9])
    nested = _variant(tmp_path, ziscale=[[0.8, 0.9]])
    assert nested.transport.ziscale == [0.8, 0.9]
    assert nested.particles_hash == flat.particles_hash
    with pytest.raises(ValueError, match="Per-simulation"):
        HysplitConfig(n_hours=-24, ziscale=[[0.8], [0.9]])


def test_variants_that_differ_only_in_footprint_fields_share_particles(tmp_path):
    config = ProjectConfig(
        **blocks(
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


def test_footprints_written_before_tres_are_hourly_and_still_found(tmp_path):
    """A footprints folder from before ``tres`` is an hourly variant's."""
    config = ProjectConfig(
        **blocks(
            mets={"hrrr": _met(tmp_path)},
            grid=GRID,
            numpar=100,
            variants={"hrrr": {}, "hrrr-quarters": {"tres": "15min"}},
        )
    )
    variants = config.resolve()
    hourly, quarters = variants["hrrr"], variants["hrrr-quarters"]
    assert hourly.particles_hash == quarters.particles_hash
    assert hourly.footprint_hash != quarters.footprint_hash

    out = Output(tmp_path / "out")
    out.write_empty_footprint(hourly, make_receptor())
    folder = out.folder("footprints", hourly)
    _rewrite_record(folder, lambda s: s.pop("tres"))

    out = Output(tmp_path / "out")
    assert out.folder("footprints", hourly) == folder
    assert out.folder("footprints", quarters) is None


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------


def test_every_data_table_is_recorded_by_its_checksum(tmp_path):
    """PYSTILT includes no tables, so a run records the four it read."""
    variant = _variant(tmp_path, numpar=100)
    assert set(variant.build.data_files) == {
        "ASCDATA.CFG",
        "LANDUSE.ASC",
        "ROUGLEN.ASC",
        "TERRAIN.ASC",
    }
    assert variant.run_settings["model"]["data_files"] == variant.build.data_files


def test_where_ascdata_cfg_says_the_tables_are_is_the_same_run(tmp_path):
    """HYSPLIT's ASCDATA.CFG says ../bdyfiles/; a run reads them where it links them, so the line changes nothing."""
    one = make_hysplit_dir(tmp_path / "one")
    two = make_hysplit_dir(tmp_path / "two")
    ascdata = two / "bdyfiles" / "ASCDATA.CFG"
    lines = ascdata.read_text().splitlines(keepends=True)
    lines[5] = "'./'  directory location of data files\n"
    ascdata.write_text("".join(lines))
    first = _variant(tmp_path, numpar=100, directory=one)
    second = _variant(tmp_path, numpar=100, directory=two)
    assert first.particles_hash == second.particles_hash


def test_a_changed_data_table_makes_another_run(tmp_path):
    """Another land use changes the particles."""
    one = make_hysplit_dir(tmp_path / "one")
    two = make_hysplit_dir(tmp_path / "two")
    (two / "bdyfiles" / "LANDUSE.ASC").write_text("other land use\n")
    first = _variant(tmp_path, numpar=100, directory=one)
    second = _variant(tmp_path, numpar=100, directory=two)
    assert first.build.sha256 == second.build.sha256
    differs = {
        name
        for name, digest in second.build.data_files.items()
        if first.build.data_files[name] != digest
    }
    assert differs == {"LANDUSE.ASC"}
    assert first.particles_hash != second.particles_hash


def test_without_directory_the_tables_beside_the_hycs_std_on_path_are_used(
    tmp_path, monkeypatch
):
    """A HYSPLIT folder whose exec/ is on PATH, as a cluster's HYSPLIT module sets it."""
    folder = make_hysplit_dir(tmp_path / "hysplit")
    terrain = folder / "bdyfiles" / "TERRAIN.ASC"
    terrain.write_text("other terrain\n")
    monkeypatch.setenv("PATH", str(folder / "exec"))
    variant = _variant(tmp_path, numpar=100)
    expected = hashlib.sha256(terrain.read_bytes()).hexdigest()
    assert variant.build.data_files["TERRAIN.ASC"] == expected


def test_a_hysplit_folder_without_its_tables_says_where_they_go(tmp_path):
    folder = make_hysplit_dir(tmp_path / "hysplit", tables=False)
    variant = _variant(tmp_path, numpar=100, directory=folder)
    with pytest.raises(HYSPLITNotFoundError, match="bdyfiles/ folder"):
        _ = variant.build
