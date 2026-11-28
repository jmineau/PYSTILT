"""The footprint stage: what ``class: stilt.footprint.FootprintStage`` gives a project."""

from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd
import pytest
import xarray as xr
from runtraj.identity import stage_hash
from runtraj.models import MetConfig
from runtraj.stages import check_stage, get_stage

from stilt.footprint import FootprintConfig, FootprintStage
from stilt.footprint.grid import Grid
from stilt.footprint.targets import Mesh

from ..fixtures.factories import make_variant

GRID = Grid(xmin=-113.0, xmax=-110.5, ymin=40.0, ymax=42.0, xres=0.01, yres=0.01)
H3 = {
    "kind": "h3",
    "resolution": 8,
    "bounds": {"xmin": -112.3, "xmax": -111.6, "ymin": 40.4, "ymax": 41.0},
}
WINDOWS = {"kind": "windows", "coords": [(-111.85, 40.77)], "size": 0.2}
LIFETIME = {"kind": "first_order_lifetime", "lifetime_hours": 5.0}

#: The settings a HYSPLIT run records that the stage reads.
COLUMNS = ["time", "indx", "long", "lati", "zagl", "foot", "mlht", "dens", "samt"]
ALL_COLUMNS = [*COLUMNS, "sigw", "tlgr"]


def _resolve(config: FootprintConfig, directory=None, **model_settings):
    return FootprintStage().resolve(config, "hysplit", model_settings, directory)


def test_the_stage_is_what_the_import_path_names():
    stage = get_stage("stilt.footprint.FootprintStage")
    assert isinstance(stage, FootprintStage)
    assert (stage.name, stage.suffix, stage.reads) == ("footprint", ".parquet", ())
    assert stage.config_class is FootprintConfig


@pytest.mark.parametrize(
    "config",
    [
        FootprintConfig(grid=GRID),
        FootprintConfig(grid=GRID, smooth_factor=0.5, tres="15min", hnf_plume=False),
        FootprintConfig(grid=GRID, geometry=H3, geometry_hash="abc123"),
        FootprintConfig(grid=GRID, transforms=[LIFETIME]),
        FootprintConfig(grid=GRID, transforms=[{"kind": "pressure_weighting"}]),
    ],
)
def test_the_config_reads_back_what_the_stage_records(config):
    check_stage(FootprintStage(), config)


#: The footprint hashes of three variants on particles with the hash below,
#: pinned so that a change that moves every stored footprint folder does so
#: on purpose. The footprint's foot settings (footprint_depth, pbl_height,
#: pbl_height_min) moved them last.
PARTICLES = "90841c632620fa42b5797789b8e3b2f071ca6077ad48fe8a7a77d99f72758bb9"
PINNED = {
    "plain": (
        FootprintConfig(grid=GRID),
        "4ed38a01979b47520a600e3e7b1f0e08ebd928201df0c70fda8df14ac05e340c",
    ),
    "smooth": (
        FootprintConfig(grid=GRID, smooth_factor=0.5, tres="15min", hnf_plume=False),
        "3bbe31b35cbf35a0fcb2d4ca18b133dc4786abdea103ea0630a96dbf1693e481",
    ),
    "geometry": (
        FootprintConfig(grid=GRID, geometry=H3, geometry_hash="abc123"),
        "34c811e2966074cc0d5dbcc19319fed2008a0bfe6912330ce054b8d04b37f4c5",
    ),
}


@pytest.mark.parametrize("name", list(PINNED))
def test_the_footprint_hash_is_pinned(name):
    config, expected = PINNED[name]
    variant = make_variant(footprint=config)
    assert variant.stage_hashes_on(PARTICLES) == {"footprint": expected}
    # On the variant's own particles, the same rule.
    assert variant.stage_hashes == variant.stage_hashes_on(variant.result_hash)
    assert variant.stage_hashes["footprint"] == stage_hash(
        "footprint", FootprintStage().settings(config), variant.result_hash
    )


# -- resolve ------------------------------------------------------------------


def test_resolve_keeps_a_grid_and_records_no_geometry():
    resolved = _resolve(FootprintConfig(grid=GRID))
    assert resolved.grid == GRID
    assert resolved.geometry_hash is None


def test_resolve_derives_the_grid_from_a_geometry_and_records_its_hash():
    config = FootprintConfig(geometry=WINDOWS, cells_per_target=4)
    resolved = _resolve(config)
    mesh = Mesh.from_spec(config.geometry)
    assert resolved.grid == mesh.to_grid(cells_per_target=4)
    assert resolved.geometry_hash == mesh.hash
    assert resolved.geometry == config.geometry  # as written
    # With a grid of its own, the geometry is recorded and the grid kept.
    both = _resolve(FootprintConfig(grid=GRID, geometry=WINDOWS))
    assert both.grid == GRID and both.geometry_hash == mesh.hash


def test_resolve_replaces_a_geometry_hash_written_in_the_config():
    written = FootprintConfig(grid=GRID, geometry_hash="written-by-hand")
    assert _resolve(written).geometry_hash is None
    stale = FootprintConfig(geometry=WINDOWS, geometry_hash="stale")
    assert _resolve(stale).geometry_hash == Mesh.from_spec(stale.geometry).hash


def test_resolve_reads_a_geometry_file_once_and_again_when_it_changes(
    tmp_path, monkeypatch
):
    pytest.importorskip("geopandas")
    from stilt.footprint import stage as stage_module

    def write(shift: float) -> None:
        feature = {
            "type": "Feature",
            "properties": {"NAME": "a"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-112.0 + shift, 40.5],
                        [-111.5 + shift, 40.5],
                        [-111.5 + shift, 41.0],
                        [-112.0 + shift, 41.0],
                        [-112.0 + shift, 40.5],
                    ]
                ],
            },
        }
        collection = {"type": "FeatureCollection", "features": [feature]}
        (tmp_path / "cells.geojson").write_text(json.dumps(collection))

    reads_of_file = []
    real = Mesh.from_spec.__func__

    def counting(cls, spec):
        reads_of_file.append(spec.path)
        return real(cls, spec)

    monkeypatch.setattr(Mesh, "from_spec", classmethod(counting))
    stage_module._read_mesh.cache_clear()
    write(0.0)
    config = FootprintConfig(geometry={"kind": "file", "path": "cells.geojson"})
    first = _resolve(config, tmp_path)
    again = _resolve(config, tmp_path)
    assert first.geometry_hash == again.geometry_hash
    assert reads_of_file == [str(tmp_path / "cells.geojson")]
    assert first.geometry.path == "cells.geojson"  # the settings keep it as written
    # A changed file is read again, and is another geometry.
    import os

    write(0.25)
    os.utime(tmp_path / "cells.geojson", ns=(1, 1))
    moved = _resolve(config, tmp_path)
    assert len(reads_of_file) == 2
    assert moved.geometry_hash != first.geometry_hash


def test_a_footprint_with_neither_a_grid_nor_a_geometry_is_refused():
    with pytest.raises(ValueError, match="the footprint has no grid. Give it a grid"):
        _resolve(FootprintConfig())


def test_the_near_field_correction_needs_its_columns_from_the_model():
    with pytest.raises(
        ValueError,
        match=r"near-field correction \(hnf_plume\) reads the particle columns "
        r"\['sigw', 'tlgr'\]. Add them to varsiwant, or set hnf_plume: false",
    ):
        _resolve(FootprintConfig(grid=GRID), varsiwant=COLUMNS)
    # With the columns, with the correction off, or for a model that does not
    # say which columns it writes, there is nothing to refuse.
    _resolve(FootprintConfig(grid=GRID), varsiwant=ALL_COLUMNS)
    _resolve(FootprintConfig(grid=GRID, hnf_plume=False), varsiwant=COLUMNS)
    _resolve(FootprintConfig(grid=GRID))
    _resolve(FootprintConfig(grid=GRID), varsiwant=None)


def test_resolve_leaves_configs_with_equal_settings_equal():
    """What the stage learns of the model at resolve is checked there and kept nowhere."""
    with_columns = _resolve(FootprintConfig(grid=GRID), varsiwant=ALL_COLUMNS)
    without = _resolve(FootprintConfig(grid=GRID))
    assert with_columns == without == FootprintConfig(grid=GRID)


# -- a variant's overrides ----------------------------------------------------


def test_a_variants_grid_changes_the_blocks_grid_field_by_field():
    block = FootprintConfig(grid=GRID, smooth_factor=2).model_dump()
    merged = FootprintConfig.override(block, {"grid": {"xres": 0.05, "yres": 0.05}})
    config = FootprintConfig.model_validate(merged)
    assert (config.grid.xres, config.grid.xmin) == (0.05, -113.0)
    assert config.smooth_factor == 2


def test_a_variants_geometry_drops_the_blocks_grid():
    block = FootprintConfig(grid=GRID).model_dump()
    merged = FootprintConfig.override(block, {"geometry": WINDOWS})
    assert FootprintConfig.model_validate(merged).grid is None
    # Unless the variant gives a grid as well.
    own = {"geometry": WINDOWS, "grid": {"xres": 0.05, "yres": 0.05}}
    assert (
        FootprintConfig.model_validate(FootprintConfig.override(block, own)).grid.xres
        == 0.05
    )


def test_a_variant_cannot_change_part_of_a_grid_derived_from_the_blocks_geometry():
    block = FootprintConfig(geometry=WINDOWS).model_dump()
    with pytest.raises(ValueError, match="changes part of a grid that is derived"):
        FootprintConfig.override(block, {"grid": {"xres": 0.05}})


# -- check --------------------------------------------------------------------


def _rows(**columns) -> pd.DataFrame:
    """Return release-point rows of two releases, the second outside :data:`GRID`."""
    rows = {
        "release": ["202407010000_-111.85_40.77_5", "202407010000_-100_40.77_5"],
        "longitude": [-111.85, -100.0],
        "latitude": [40.77, 40.77],
        "altitude": [5.0, 5.0],
        "altitude_ref": ["agl", "agl"],
        "kind": ["point", "point"],
    }
    return pd.DataFrame({**rows, **columns})


def test_check_lists_the_releases_outside_the_grid():
    config = _resolve(FootprintConfig(grid=GRID), varsiwant=ALL_COLUMNS)
    found = FootprintStage().check(config, _rows(), None)
    assert found == [
        (
            "grid",
            "202407010000_-100_40.77_5",
            "longitude -100 and latitude 40.77 are outside the footprint grid "
            "(-113 to -110.5 longitude, 40 to 42 latitude)",
        )
    ]


def test_the_settings_say_what_needs_zsfc_above_sea_level():
    assert FootprintConfig(grid=GRID).zsfc_reasons == [
        "the near-field correction (hnf_plume) is skipped for them"
    ]
    assert FootprintConfig(grid=GRID, hnf_plume=False).zsfc_reasons == []
    weighted = FootprintConfig(
        grid=GRID, hnf_plume=False, transforms=[{"kind": "pressure_weighting"}]
    )
    assert weighted.zsfc_reasons == [
        "the pressure_weighting transform fails their footprints"
    ]


class CoarseFootprint(FootprintStage):
    """The footprint stage under another name, as a package that builds on it would have."""

    name = "coarse_footprint"


def test_a_footprint_stage_under_another_name_writes_its_own_files(tmp_path):
    """The file records the settings the stage was handed and the hash of its own folder."""
    import pyarrow.parquet as pq
    from runtraj.identity import file_stamp
    from runtraj.output import Output
    from runtraj.simulation import Simulation

    from stilt.footprint import read_footprint

    from ..fixtures.factories import make_release
    from ..fixtures.particles import finished, random_walk

    config = FootprintConfig(grid=GRID, smooth_factor=0.5)
    variant = make_variant()
    variant = type(variant)(
        **{
            **{f: getattr(variant, f) for f in variant.__dataclass_fields__},
            "stages": {"coarse_footprint": config},
            "stage_classes": {
                "coarse_footprint": "tests.footprint.test_stage.CoarseFootprint"
            },
        }
    )
    release = make_release(dt.datetime(2024, 1, 15, 18))  # where the walk starts
    sim = Simulation(release, variant, Output(tmp_path / "output"))
    sim.__dict__["particles"] = finished(random_walk(), release)
    path = sim.output.make_path("coarse_footprint", variant, release.id)
    path.parent.mkdir(parents=True)

    CoarseFootprint().run(config, sim, path)

    assert path.parent.parent.parent.name == "coarse_footprint"
    meta = pq.read_metadata(path).metadata
    stamp = file_stamp(variant.stage_hashes["coarse_footprint"])
    assert {key: meta[key] for key in stamp} == stamp
    assert meta[b"stilt:particles_hash"].decode() == variant.result_hash
    foot = read_footprint(path)
    assert foot is not None and foot.stilt.config.smooth_factor == 0.5


# -- the depth and foot -------------------------------------------------------


def test_the_foot_settings_are_the_footprints_and_read_back():
    config = FootprintConfig(
        grid=GRID, footprint_depth=0.3, pbl_height="richardson", pbl_height_min=150
    )
    check_stage(FootprintStage(), config)
    settings = FootprintStage().settings(config)
    assert (
        settings["footprint_depth"],
        settings["pbl_height"],
        settings["pbl_height_min"],
    ) == (0.3, "richardson", 150.0)


@pytest.mark.parametrize("value", [0, -10.0, "pblh"])
def test_pbl_height_refuses_a_height_that_is_not_above_the_ground(value):
    with pytest.raises(ValueError, match="pbl_height"):
        FootprintConfig(pbl_height=value)


@pytest.mark.parametrize(
    ("setting", "value"),
    [("pbl_height_min", -1.0), ("footprint_depth", 0.0), ("footprint_depth", -1.0)],
)
def test_a_negative_depth_is_refused(setting, value):
    with pytest.raises(ValueError, match=setting):
        FootprintConfig(**{setting: value})


def test_on_hysplits_foot_the_footprint_takes_the_runs_veght():
    """The near-field correction reads the depth HYSPLIT made its foot with, and the record says it."""
    resolved = _resolve(FootprintConfig(grid=GRID), veght=0.3, varsiwant=ALL_COLUMNS)
    assert resolved.footprint_depth == 0.3


def test_a_footprint_depth_written_equal_to_veght_is_kept():
    resolved = _resolve(
        FootprintConfig(grid=GRID, footprint_depth=0.3),
        veght=0.3,
        varsiwant=ALL_COLUMNS,
    )
    assert resolved.footprint_depth == 0.3


def test_a_footprint_depth_unlike_veght_on_hysplits_foot_is_refused():
    with pytest.raises(
        ValueError,
        match=r"footprint_depth is 0.3, and the footprint is made from the hysplit "
        r"model's own foot, which it made with veght 0.5. Set veght: 0.3 in the "
        r"model's settings for that depth, or set pbl_height so that PYSTILT "
        r"computes foot with footprint_depth.",
    ):
        _resolve(FootprintConfig(grid=GRID, footprint_depth=0.3), veght=0.5)


@pytest.mark.parametrize(
    ("config", "model_settings"),
    [
        # PYSTILT computes foot: with pbl_height set ...
        ({"pbl_height": "met"}, {"veght": 0.5}),
        # ... for particles HYSPLIT is not asked to give a foot ...
        ({}, {"veght": 0.5, "varsiwant": ["time", "indx", "long", "lati", "zagl"]}),
        # ... and for a model that records no depth.
        ({}, {}),
    ],
    ids=["pbl_height", "no foot in varsiwant", "no veght"],
)
def test_a_foot_pystilt_computes_keeps_the_footprints_depth(config, model_settings):
    resolved = _resolve(
        FootprintConfig(grid=GRID, footprint_depth=0.3, hnf_plume=False, **config),
        **model_settings,
    )
    assert resolved.footprint_depth == 0.3


def _particles(**columns) -> pd.DataFrame:
    """Two rows of one particle, with *columns* added."""
    return pd.DataFrame(
        {
            "particle": [1.0, 1.0],
            "age": [-1.0, -2.0],
            "lon": [-111.9, -112.0],
            "lat": [40.7, 40.6],
            "zagl": [5.0, 900.0],
            **columns,
        }
    )


@pytest.fixture
def add_foot(monkeypatch):
    """Stand in for add_foot, recording what it was given."""
    from stilt.footprint import stage

    seen: dict = {}

    def fake(particles, met_files, footprint_depth, **mixed_layer):
        seen.update(
            columns=list(particles.columns),
            met_files=met_files,
            footprint_depth=footprint_depth,
            **mixed_layer,
        )
        return particles.assign(foot=5.0, mlht=300.0)

    monkeypatch.setattr(stage, "add_foot", fake)
    return seen


def test_the_models_own_foot_is_used_as_it_is(add_foot):
    from stilt.footprint import footprint_particles

    particles = _particles(foot=[1e-3, 2e-3])
    out = footprint_particles(particles, FootprintConfig(), [])
    assert out is particles
    assert add_foot == {}


def test_a_positions_only_run_gets_a_foot_from_its_met_files(
    add_foot, tmp_path, caplog
):
    from stilt.footprint import footprint_particles

    met = tmp_path / "met"
    met.touch()
    particles = _particles(time=pd.Timestamp("2023-01-01 11:59"))
    config = FootprintConfig(footprint_depth=0.3)
    with caplog.at_level("INFO", logger="stilt.footprint.stage"):
        out = footprint_particles(particles, config, [met], z_top=10000.0)

    assert out["foot"].tolist() == [5.0, 5.0]
    assert "foot" not in particles.columns  # stored nowhere, the input untouched
    assert add_foot["met_files"] == [met]
    assert add_foot["footprint_depth"] == 0.3
    assert (add_foot["pbl_height"], add_foot["pbl_height_min"]) == (None, 0.0)
    assert add_foot["z_top"] == 10000.0
    assert "foot computed by PYSTILT from the met's PBLH" in caplog.text


def test_the_variants_model_opens_the_met_around_the_particles(
    add_foot, tmp_path, monkeypatch
):
    from runtraj import Bounds

    from stilt.footprint import footprint_particles

    met = tmp_path / "met"
    met.touch()
    variant = make_variant()
    opened = xr.Dataset()
    calls = []

    def open_files(self, files, bounds=None):
        calls.append((self, list(files), bounds))
        return opened

    monkeypatch.setattr(type(variant.met_config), "open", open_files)
    particles = _particles(time=pd.Timestamp("2023-01-01 11:59"))
    footprint_particles(particles, FootprintConfig(), [met], variant=variant)

    assert calls == [
        (
            variant.met_config,
            [met],
            Bounds(xmin=-112.0, xmax=-111.9, ymin=40.6, ymax=40.7),
        )
    ]
    assert add_foot["met_files"] is opened


def test_hysplits_met_files_reach_foot_as_the_paths_they_are(add_foot, tmp_path):
    from stilt.footprint import footprint_particles

    met = tmp_path / "met"
    met.touch()
    particles = _particles(time=pd.Timestamp("2023-01-01 11:59"))
    footprint_particles(particles, FootprintConfig(), [met], variant=make_variant())

    assert add_foot["met_files"] == [met]


@pytest.mark.parametrize(
    ("lons", "lats", "box"),
    [
        ([-111.9, -112.0], [40.7, 40.6], (-112.0, -111.9, 40.6, 40.7)),
        ([179.5, -179.5, 179.9], [10.0, 11.0, 10.5], (179.5, -179.5, 10.0, 11.0)),
        ([-112.0, -112.0], [40.6, 40.6], (-112.01, -111.99, 40.59, 40.61)),
        ([-112.0, float("nan")], [40.6, 41.0], (-112.01, -111.99, 40.59, 40.61)),
    ],
    ids=["a box", "across 180", "one point", "nan rows left out"],
)
def test_the_met_is_opened_in_the_smallest_box_around_the_particles(lons, lats, box):
    from stilt.footprint.stage import _particle_bounds

    bounds = _particle_bounds(pd.DataFrame({"lon": lons, "lat": lats}))

    assert (bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax) == pytest.approx(box)


def test_particles_with_no_position_open_the_whole_met():
    from stilt.footprint.stage import _particle_bounds

    nowhere = pd.DataFrame({"lon": [float("nan")], "lat": [40.0]})
    assert _particle_bounds(nowhere) is None


def test_stored_particles_of_a_model_with_its_own_met_need_the_variant(
    add_foot, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from stilt.footprint import stage

    met = tmp_path / "met"
    met.touch()
    meta = SimpleNamespace(met_files=[met], settings={"model": {"name": "gribmodel"}})
    monkeypatch.setattr(stage, "particles_metadata", lambda path: meta)

    class GribMet(MetConfig):
        def settings(self):
            return {}

        def open(self, files, bounds=None):
            return None

    monkeypatch.setattr(
        stage, "get_model", lambda name: SimpleNamespace(met_config_class=GribMet)
    )
    particles = _particles(time=pd.Timestamp("2023-01-01 11:59"))

    with pytest.raises(ValueError, match="gribmodel.*variant=sim.variant"):
        stage.stored_particles(particles, FootprintConfig(), tmp_path / "p.parquet")
    assert "met_files" not in add_foot


def test_pbl_height_replaces_the_models_foot(add_foot, tmp_path, caplog):
    from stilt.footprint import footprint_particles

    met = tmp_path / "met"
    met.touch()
    particles = _particles(time=pd.Timestamp("2023-01-01"), foot=1e-3, mlht=800.0)
    config = FootprintConfig(pbl_height="richardson", pbl_height_min=150)
    with caplog.at_level("INFO", logger="stilt.footprint.stage"):
        out = footprint_particles(particles, config, [met])

    assert "foot" not in add_foot["columns"]
    assert (add_foot["pbl_height"], add_foot["pbl_height_min"]) == ("richardson", 150)
    assert out["foot"].tolist() == [5.0, 5.0]
    assert particles["foot"].tolist() == [1e-3, 1e-3]
    assert (
        "the mixed layer found from the met's profile (HYSPLIT's kmixd 3), at "
        "least 150 m,"
    ) in caplog.text
    assert "in place of the model's foot" in caplog.text


def test_a_foot_to_compute_needs_the_met_files_where_the_run_read_them(
    add_foot, tmp_path
):
    from stilt.footprint import footprint_particles

    with pytest.raises(FileNotFoundError, match="the run named none"):
        footprint_particles(_particles(), FootprintConfig(), [])
    gone = tmp_path / "hrrr_20230101"
    with pytest.raises(
        FileNotFoundError,
        match=rf"{re.escape(str(gone))} is not there any more \(nor 1 more\). Put the met files back",
    ):
        footprint_particles(_particles(), FootprintConfig(), [gone, tmp_path / "b"])
    assert add_foot == {}


def test_a_stored_runs_met_files_are_read_only_to_compute_foot(
    add_foot, tmp_path, monkeypatch
):
    from runtraj.particles import write_particles
    from runtraj.releases import PointRelease

    from stilt.footprint import stage

    met = tmp_path / "met"
    met.touch()
    release = PointRelease(
        time="2023-01-01 12:00", longitude=-111.9, latitude=40.7, altitude=5.0
    )
    path = write_particles(
        tmp_path / "p.parquet", _particles(foot=1e-3), release, {}, [met]
    )
    particles = _particles(foot=1e-3)  # in memory, without time

    # HYSPLIT's foot: the file's metadata is not read.
    monkeypatch.setattr(
        stage, "particles_metadata", lambda p: pytest.fail("read the metadata")
    )
    assert stage.stored_particles(particles, FootprintConfig(), path) is particles
    monkeypatch.undo()
    monkeypatch.setattr(stage, "add_foot", _record_time(add_foot))

    out = stage.stored_particles(
        particles, FootprintConfig(pbl_height=500.0), path, start=release.time
    )
    assert add_foot["met_files"] == [met]
    assert add_foot["time"] == [
        pd.Timestamp("2023-01-01 11:59"),
        pd.Timestamp("2023-01-01 11:58"),
    ]
    assert "time" not in out.columns  # given for add_foot, then taken off


def _record_time(seen: dict):
    """An add_foot that records the time and met files it was given."""

    def fake(particles, met_files, footprint_depth, **mixed_layer):
        seen.update(met_files=met_files, time=particles["time"].tolist())
        return particles.drop(columns="foot", errors="ignore").assign(foot=5.0)

    return fake


def test_a_variant_without_a_footprint_corrects_with_its_runs_veght():
    from stilt.footprint.stage import variant_footprint

    plain = make_variant(veght=0.3)
    assert variant_footprint(plain).footprint_depth == 0.3
    own = make_variant(footprint=FootprintConfig(grid=GRID, footprint_depth=0.7))
    assert variant_footprint(own).footprint_depth == 0.7
