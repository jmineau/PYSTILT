"""The footprint stage: what ``class: stilt.footprint.FootprintStage`` gives a project."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stilt.footprint import FootprintConfig, FootprintStage
from stilt.footprint.targets import Mesh
from stilt.identity import stage_hash
from stilt.spatial import Grid
from stilt.stages import check_stage, get_stage, reads

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
    assert (stage.name, stage.suffix, reads(stage)) == ("footprint", ".parquet", ())
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


#: The footprint hashes of three variants before footprints were a stage, on
#: particles with the hash below. A footprint made then must be found now.
PARTICLES = "90841c632620fa42b5797789b8e3b2f071ca6077ad48fe8a7a77d99f72758bb9"
BEFORE_STAGES = {
    "plain": (
        FootprintConfig(grid=GRID),
        "4389b89ed19c2b791a491280a0639e980e4865186d51d04c1143b7c4a1de03fe",
    ),
    "smooth": (
        FootprintConfig(grid=GRID, smooth_factor=0.5, tres="15min", hnf_plume=False),
        "063eca80fece2765cfd04fb2e813e6f2ab071d162a28bade0057e8fdc8f75aed",
    ),
    "geometry": (
        FootprintConfig(grid=GRID, geometry=H3, geometry_hash="abc123"),
        "36adbc0225a639ee1541079e3c1cb88d3aec41bc4246054d0ac33dd48d65b99f",
    ),
}


@pytest.mark.parametrize("name", list(BEFORE_STAGES))
def test_the_footprint_hash_is_what_it_was_before_stages(name):
    config, expected = BEFORE_STAGES[name]
    variant = make_variant(footprint=config)
    assert variant.stage_hashes_on(PARTICLES) == {"footprint": expected}
    # On the variant's own particles, the same rule.
    assert variant.stage_hashes == variant.stage_hashes_on(variant.particles_hash)
    assert variant.stage_hashes["footprint"] == stage_hash(
        "footprint", FootprintStage().settings(config), variant.particles_hash
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
    """Return release-point rows of two receptors, the second outside :data:`GRID`."""
    rows = {
        "receptor": ["202407010000_-111.85_40.77_5", "202407010000_-100_40.77_5"],
        "longitude": [-111.85, -100.0],
        "latitude": [40.77, 40.77],
        "altitude": [5.0, 5.0],
        "altitude_ref": ["agl", "agl"],
        "kind": ["point", "point"],
    }
    return pd.DataFrame({**rows, **columns})


def test_check_lists_the_receptors_outside_the_grid():
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
