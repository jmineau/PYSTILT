"""The map stage: each footprint drawn as an image for a web map."""

from __future__ import annotations

import base64
import datetime as dt
import json

import numpy as np
import pytest
import xarray as xr
from runtraj.stages import check_stage, get_stage, ordered

from stilt.footprint import FootprintConfig, FootprintStage, MapConfig, MapStage, maps
from stilt.footprint.grid import Grid

from ..fixtures.factories import make_release, make_variant


def footprint(values, lat0=40.0, lon0=-112.0, res=0.5, hours=2):
    """A footprint on a regular grid, the same in each hour, from rows south to north."""
    values = np.asarray(values, dtype=float)
    lat = lat0 + res * np.arange(values.shape[0])
    lon = lon0 + res * np.arange(values.shape[1])
    data = np.repeat(values[None] / hours, hours, axis=0)
    coords = {"time": np.arange(hours), "lat": lat, "lon": lon}
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords=coords,
        attrs={"units": "ppm (umol m-2 s-1)-1"},
    )


# -- the stage ----------------------------------------------------------------


def test_the_stage_is_what_the_import_path_names():
    stage = get_stage("stilt.footprint.MapStage")
    assert isinstance(stage, MapStage)
    assert (stage.name, stage.suffix, stage.reads) == ("map", ".json", ("footprint",))
    assert stage.config_class is MapConfig


def test_the_map_runs_after_the_footprint():
    stages = {"map": MapStage(), "footprint": FootprintStage()}
    assert ordered(stages) == ["footprint", "map"]


@pytest.mark.parametrize(
    "config", [MapConfig(), MapConfig(vmin=1e-6, vmax=1.0, colormap="magma")]
)
def test_the_config_reads_back_what_the_stage_records(config):
    check_stage(MapStage(), config)


def test_a_scale_that_does_not_go_up_is_refused():
    with pytest.raises(ValueError, match="below vmax"):
        MapConfig(vmin=1.0, vmax=0.1)


def test_an_unknown_colormap_is_refused_when_the_variants_are_read():
    with pytest.raises(ValueError, match="not one of matplotlib's"):
        MapStage().resolve(MapConfig(colormap="no-such-map"), "hysplit", {}, None)


GRID = Grid(xmin=-112.5, xmax=-111.0, ymin=40.0, ymax=41.5, xres=0.01, yres=0.01)


def _sim(tmp_path):
    """A simulation of a variant with a footprint and a map, and its particles."""
    from runtraj.output import Output
    from runtraj.simulation import Simulation

    from ..fixtures.particles import finished, random_walk

    variant = make_variant(footprint=FootprintConfig(grid=GRID))
    variant = type(variant)(
        **{
            **{f: getattr(variant, f) for f in variant.__dataclass_fields__},
            "stages": {"footprint": variant.stages["footprint"], "map": MapConfig()},
            "stage_classes": {
                **variant.stage_classes,
                "map": "stilt.footprint.MapStage",
            },
        }
    )
    release = make_release(dt.datetime(2024, 1, 15, 18))  # where the walk starts
    sim = Simulation(release, variant, Output(tmp_path / "output"))
    sim.__dict__["particles"] = finished(random_walk(), release)
    return sim


def _run(stage, config, sim):
    path = sim.output.make_path(stage.name, sim.variant, sim.release.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage.run(config, sim, path)
    return path


def test_the_map_file_holds_the_image_and_where_it_goes(tmp_path):
    sim = _sim(tmp_path)
    _run(FootprintStage(), sim.variant.stages["footprint"], sim)
    path = _run(MapStage(), MapConfig(), sim)

    assert path.name == f"{sim.release.id}.json"
    entry = json.loads(path.read_text())
    assert entry["release"] == sim.release.id
    assert entry["time"] == "2024-01-15T18:00:00Z"
    assert entry["location"]["longitude"] == sim.release.longitude
    assert entry["units"]
    assert (entry["vmin"], entry["vmax"], entry["colormap"]) == (1e-5, 0.1, "viridis")
    (south, west), (north, east) = entry["bounds"]
    assert south < sim.release.latitude < north and west < sim.release.longitude < east
    head, data = entry["image"].split(",", 1)
    assert head == "data:image/png;base64"
    assert base64.b64decode(data).startswith(b"\x89PNG")


def test_an_empty_footprint_gets_a_map_file_with_no_image(tmp_path):
    from stilt.footprint.stage import store_empty_footprint

    sim = _sim(tmp_path)
    store_empty_footprint(sim.output, sim.variant, sim.release)
    entry = json.loads(_run(MapStage(), MapConfig(), sim).read_text())
    assert entry["image"] is None and entry["bounds"] is None
    assert entry["release"] == sim.release.id


def test_the_map_entry_takes_the_scale_it_is_given():
    config = MapConfig(vmin=1e-3, vmax=1.0, colormap="magma")
    entry = maps.map_entry(make_release(), footprint([[1e-2]]), config)
    assert (entry["vmin"], entry["vmax"], entry["colormap"]) == (1e-3, 1.0, "magma")


# -- the image ----------------------------------------------------------------


def test_bounds_are_the_outer_cell_edges():
    foot = footprint(np.ones((3, 2)))
    assert maps.bounds(foot) == [[39.75, -112.25], [41.25, -111.25]]


def _on(grid, values=None):
    """A footprint labeled as calc_footprint labels one on *grid*."""
    x, y = grid.axes
    values = np.ones((y.size, x.size)) if values is None else values
    y_dim, x_dim = grid.dims
    return xr.DataArray(values, dims=(y_dim, x_dim), coords={y_dim: y, x_dim: x})


def test_a_grid_across_the_antimeridian_is_drawn_east_past_180():
    grid = Grid(xmin=170, xmax=-170, ymin=-5, ymax=5, xres=0.5, yres=0.5)
    values = np.zeros((20, 40))
    values[3, 30] = (
        1e-2  # one strong cell, 4th row from the south, 31st column from 170
    )
    foot = _on(grid, values)
    assert foot["lon"].values[[0, -1]].tolist() == [170.25, 189.75]
    assert maps.bounds(foot) == [[-5.0, 170.0], [5.0, 190.0]]
    rgba = maps.image(foot)
    assert rgba.shape[:2] == (20, 40)
    # The image's last row is the south, so row 3 from the south is 16 from the top.
    assert np.argwhere(rgba[..., 3] > 0).tolist() == [[16, 30]]


@pytest.mark.parametrize("draw", [maps.bounds, maps.image, maps.map_entry])
def test_a_projected_footprint_is_refused_in_words(draw):
    grid = Grid(
        xmin=380_000,
        xmax=500_000,
        ymin=4_430_000,
        ymax=4_600_000,
        xres=10_000,
        yres=10_000,
        crs="EPSG:32612",
    )
    args = (make_release(), _on(grid)) if draw is maps.map_entry else (_on(grid),)
    with pytest.raises(ValueError, match="longitude/latitude grid"):
        draw(*args)


def test_image_puts_north_at_the_top_and_hides_small_values():
    # South row strong, north row below the scale.
    rgba = maps.image(footprint([[1e-2, 1e-2], [1e-9, 1e-9]]))
    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.uint8
    assert (rgba[0, :, 3] == 0).all()  # north: transparent
    assert (rgba[1, :, 3] > 0).all()  # south: drawn


def test_image_is_summed_over_time_on_a_fixed_scale():
    weak, strong = maps.image(footprint([[1e-4]])), maps.image(footprint([[1e-2]]))
    assert weak[0, 0, 3] < strong[0, 0, 3]
    top = maps.image(footprint([[10.0]]))
    assert top[0, 0, 3] == round(0.9 * 255)  # above vmax: the top of the scale


def test_image_rows_follow_web_mercator():
    # Over 40-80 N, equal Mercator bands hold more rows of the southern grid:
    # the north of the image takes only the northmost grid rows, and the
    # image has more rows than the grid so that no grid row is left out.
    values = np.arange(1, 9)[:, None] * np.ones((8, 1)) * 1e-3
    rgba = maps.image(footprint(values, lat0=40.0, res=5.0, hours=1))
    colors = [tuple(row[0]) for row in rgba]
    assert len(rgba) > 8
    assert colors[0] == colors[1]  # two image rows from the same grid row, in the north
    assert len(set(colors)) == 8


def test_png_is_a_png():
    assert maps.png(footprint(np.ones((2, 2)) * 1e-3)).startswith(b"\x89PNG")


def mercator(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def shown_rows(lat0, n, res):
    """
    Return the grid row (north first) each image row shows, and the image's rows.

    Each grid row has its own color. A row of the image is matched to the
    grid row it looks like, using the same footprint on a grid so thin that
    the rows are drawn one to one.
    """
    values = 10 ** (-4.95 + 3.9 * np.arange(n) / max(n - 1, 1))
    foot = footprint(values[:, None] * np.ones((1, 2)), lat0=lat0, res=res, hours=1)
    thin = footprint(values[:, None] * np.ones((1, 2)), lat0=0.0, res=1e-4, hours=1)
    reference = maps.image(thin)[:, 0, :].astype(int)
    rgba = maps.image(foot)
    got = [
        int(np.abs(reference - row[0].astype(int)).sum(axis=1).argmin()) for row in rgba
    ]
    return np.array(got), len(rgba)


def expected(lat0, n, res, rows):
    """For each of *rows* equal Mercator bands, the grid row (north first) it overlaps most."""
    lat = (lat0 + res * np.arange(n))[::-1]
    south, north = lat.min() - res / 2, lat.max() + res / 2
    bands = np.linspace(mercator(north), mercator(south), rows + 1)
    top, bottom = mercator(lat + res / 2), mercator(lat - res / 2)
    overlap = np.minimum(bands[:-1, None], top) - np.maximum(bands[1:, None], bottom)
    return overlap.argmax(axis=1)


GRIDS = {
    "the WBB grid": (40.3, 90, 0.01),
    "one degree at 41 N": (40.5, 10, 0.1),
    "20 to 60 N, 0.25 degree": (20.0, 161, 0.25),
    "20 to 60 N, 1 degree": (20.0, 41, 1.0),
    "70 to 85 N, 0.1 degree": (70.0, 151, 0.1),
    "equator to 80 N, 1 degree": (0.0, 81, 1.0),
    "60 to 20 S, 0.5 degree": (-60.0, 81, 0.5),
}


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_image_rows_show_the_grid_row_a_web_map_would(grid):
    got, rows = shown_rows(*grid)
    assert (got == expected(*grid, rows)).all()


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_image_shows_every_grid_row(grid):
    got, _ = shown_rows(*grid)
    assert len(set(got)) == grid[1]


def test_image_of_a_small_grid_has_a_row_per_grid_row():
    """The WBB grid has no row to add: its image has a row per grid row."""
    foot = footprint(np.ones((90, 100)), lat0=40.3, res=0.01)
    assert maps.image(foot).shape == (90, 100, 4)


def test_image_keeps_a_lone_strong_cell_on_a_wide_grid():
    values = np.full((161, 2), 1e-7)
    # Grid rows (south first) the plain, one-row-per-cell image would skip.
    plain = np.unique(expected(20.0, 161, 0.25, 161))
    skipped = sorted(set(range(161)) - set(plain))
    assert skipped
    values[161 - 1 - skipped[len(skipped) // 2], :] = 5e-2
    rgba = maps.image(footprint(values, lat0=20.0, res=0.25, hours=1))
    assert (rgba[..., 3] > 0).any()
