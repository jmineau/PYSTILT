"""Tests for runtraj.identity: what identifies a run and a footprint, and reading it back."""

import pytest
import yaml
from runtraj.config import ProjectConfig, Variant
from runtraj.identity import read_run_settings, settings_hash
from runtraj.models.hysplit import MetConfig
from runtraj.output import Output

from stilt.footprint.stage import store_empty_footprint
from tests.fixtures.factories import blocks

from .fixtures.factories import (
    make_met_config,
    make_release,
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


# ---------------------------------------------------------------------------
# what a run's settings hold
# ---------------------------------------------------------------------------


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
    feet = {name: v.stage_hashes.get("footprint") for name, v in variants.items()}
    assert len(set(feet.values())) == 4


# ---------------------------------------------------------------------------
# reading stored settings back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gone",
    [{"footprint_depth": 0.3}, {"footprint_depth": 0.5}, {"pbl_height": "richardson"}],
)
def test_a_stored_setting_this_version_does_not_have_keeps_the_run_apart(
    tmp_path, gone
):
    """
    A run made when the depth was a run setting is not taken for HYSPLIT's ``veght``.

    Dropping the setting would read a run made with ``footprint_depth: 0.3``,
    or with a ``foot`` computed under ``pbl_height``, as one made with the
    default ``veght: 0.5``, and a variant would find particles made otherwise.
    """
    variant = _variant(tmp_path, numpar=100)
    stored = {k: v for k, v in variant.run_settings.items() if k != "veght"}
    stored.update(gone)
    record = read_run_settings(stored)
    assert settings_hash(record) != variant.particles_hash
    assert {k: record[k] for k in gone} == gone


def _rewrite_record(folder, change) -> None:
    path = folder / "_settings.yaml"
    record = yaml.safe_load(path.read_text())
    change(record["settings"])
    path.write_text(yaml.safe_dump(record))


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
    assert hourly.stage_hashes.get("footprint") != quarters.stage_hashes.get(
        "footprint"
    )

    out = Output(tmp_path / "out")
    store_empty_footprint(out, hourly, make_release())
    folder = out.folder("footprint", hourly)
    _rewrite_record(folder, lambda s: s.pop("tres"))

    out = Output(tmp_path / "out")
    assert out.folder("footprint", hourly) == folder
    assert out.folder("footprint", quarters) is None


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------
