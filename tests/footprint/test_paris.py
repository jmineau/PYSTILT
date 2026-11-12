"""Tests for ``foot.stilt.to_paris``."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import stilt
from tests.fixtures.particles import random_walk

RECEPTOR_TIME = pd.Timestamp("2024-01-15 18:00")
RECEPTOR_ID = "202401151800_-111.85_40.77_5"


@pytest.fixture
def receptor():
    """A point receptor 5 m above ground in Salt Lake City."""
    return stilt.PointReceptor(
        time=RECEPTOR_TIME.to_pydatetime(),
        longitude=-111.85,
        latitude=40.77,
        altitude=5,
    )


@pytest.fixture
def paris_grid():
    """A 0.1 degree grid around the receptor."""
    return stilt.Grid(xmin=-113, xmax=-110.5, ymin=40, ymax=42, xres=0.1, yres=0.1)


@pytest.fixture
def particles():
    """A backward particle table of 20 particles over 48 hours."""
    return random_walk()


@pytest.fixture
def footprint(particles, receptor, paris_grid):
    """An hourly footprint of the random walk, as PYSTILT makes it."""
    return stilt.calc_footprint(particles, receptor, paris_grid)


def layered(hours_back, *, dim="time", lat=(40.5, 41.5), lon=(-112.5, -111.5)):
    """A footprint with a layer starting each of *hours_back* hours before the receptor time; its value is that number."""
    hours_back = sorted(hours_back, reverse=True)  # layers in time order
    starts = RECEPTOR_TIME - pd.to_timedelta(hours_back, unit="h")
    values = np.array(hours_back, dtype=float)[:, None, None] * np.ones(
        (1, len(lat), len(lon))
    )
    return xr.DataArray(
        values,
        dims=(dim, "lat", "lon"),
        coords={"time": (dim, starts), "lat": list(lat), "lon": list(lon)},
    ).assign_coords(receptor=RECEPTOR_ID)


def test_the_hour_before_the_receptor_time_is_h_back_1():
    ds = layered([1, 2, 3]).stilt.to_paris()
    per_hour = ds.srr_time_resolved.isel(time=0, latitude=0, longitude=0).values
    # Each layer's value is its hours back, so it must land at that H_back.
    assert per_hour.tolist() == [0, 1, 2, 3] + [0] * 20
    assert per_hour[0] == 0  # the hour after the receptor time stays empty


def test_layers_24_hours_back_and_older_go_to_the_residual():
    ds = layered(range(1, 31)).stilt.to_paris()
    cell = ds.isel(time=0, latitude=0, longitude=0)
    np.testing.assert_array_equal(cell.srr_time_resolved.values, np.arange(24))
    assert float(cell.srr_residual) == sum(range(24, 31))
    assert float(cell.srr) == sum(range(1, 31))
    np.testing.assert_allclose(
        ds.srr_time_resolved.sum("resolution") + ds.srr_residual, ds.srr
    )


def test_the_layer_dimension_may_have_any_name():
    # One receptor of project.footprints(): time runs along the layer dimension.
    for dim in ("hour", "layer"):
        renamed = layered([1, 2, 3], dim=dim).stilt.to_paris()
        xr.testing.assert_identical(renamed, layered([1, 2, 3]).stilt.to_paris())


def test_a_receptor_of_project_footprints():
    one = layered([3, 2, 1], dim="hour")
    many = xr.concat(
        [one.drop_vars("receptor"), one.drop_vars("receptor")], dim="receptor"
    ).assign_coords(receptor=[RECEPTOR_ID, "202401151900_-111.85_40.77_5"])
    first = many.sel(receptor=RECEPTOR_ID).stilt.to_paris()
    xr.testing.assert_identical(first, one.stilt.to_paris())
    with pytest.raises(ValueError, match="Select one receptor"):
        many.stilt.to_paris()


def test_layout_and_metadata():
    ds = layered([1, 2]).stilt.to_paris()
    assert ds.srr.dims == ("time", "latitude", "longitude")
    assert ds.srr_time_resolved.dims == ("time", "latitude", "longitude", "resolution")
    assert ds.srr_residual.dims == ("time", "latitude", "longitude")
    assert ds.srr.attrs["units"] == "(mol/mol)/(mol/m2/s)"
    assert ds.resolution.values.tolist() == list(range(24))
    assert ds.resolution.attrs["units"] == "hours"
    assert ds.time.values.tolist() == [RECEPTOR_TIME.value]
    assert ds.attrs["receptor"] == RECEPTOR_ID
    assert "height" not in ds.dims  # no particles, no particle_locations


def test_a_pystilt_footprint(footprint):
    ds = footprint.stilt.to_paris()
    np.testing.assert_allclose(
        ds.srr.isel(time=0).values, footprint.sum("time").values, rtol=1e-12
    )
    # 48 hours of particles: the last 24 layers are the residual.
    np.testing.assert_allclose(
        ds.srr_residual.isel(time=0).values,
        footprint.where(footprint.time < RECEPTOR_TIME - pd.Timedelta("23h"), 0)
        .sum("time")
        .values,
        rtol=1e-12,
    )


@pytest.mark.parametrize("tres", ["15min", "1D", "2h"])
def test_layers_other_than_an_hour_are_refused(particles, receptor, paris_grid, tres):
    foot = stilt.calc_footprint(particles, receptor, paris_grid, tres=tres)
    with pytest.raises(ValueError, match=f"layers are {tres} long"):
        foot.stilt.to_paris()


def test_uneven_layers_without_settings_are_refused():
    with pytest.raises(ValueError, match="layers are 2h long"):
        layered([2, 4, 6]).stilt.to_paris()
    with pytest.raises(ValueError, match="not evenly spaced"):
        layered([1, 2, 4]).stilt.to_paris()
    with pytest.raises(ValueError, match="one layer and no settings"):
        layered([1]).stilt.to_paris()


def test_a_forward_footprint_is_refused():
    with pytest.raises(ValueError, match="backward footprints only"):
        layered([0, -1]).stilt.to_paris()


def test_a_projected_grid_is_refused():
    foot = layered([1, 2]).rename(lat="y", lon="x")
    with pytest.raises(ValueError, match="longitude/latitude grid"):
        foot.stilt.to_paris()


def test_a_footprint_without_its_receptor_is_refused():
    with pytest.raises(ValueError, match="receptor id"):
        layered([1, 2]).drop_vars("receptor").stilt.to_paris()


# A grid of 4 longitudes (-113 to -109) by 3 latitudes (40 to 43), 1 degree cells.
LON = (-112.5, -111.5, -110.5, -109.5)
LAT = (40.5, 41.5, 42.5)


def exits_table(rows):
    """A table of grid exits from (particle, edge, lon, lat, zagl) tuples."""
    return pd.DataFrame(rows, columns=["particle", "edge", "lon", "lat", "zagl"])


def test_particle_locations_count_each_exit_by_cell_and_height():
    exits = exits_table(
        [
            (1, "n", -111.2, 43.0, 150.0),  # lon cell 1, height layer 0
            (2, "n", -111.9, 43.0, 1500.0),  # lon cell 1, height layer 1
            (3, "e", -109.0, 40.2, 2500.0),  # lat cell 0, layer 2
            (4, "s", -112.9, 40.0, 50.0),  # lon cell 0, layer 0
            (5, "w", -113.0, 42.9, 25000.0),  # lat cell 2, above the top: top layer
        ]
    )
    ds = layered([1, 2], lat=LAT, lon=LON).stilt.to_paris(exits=exits, n_particles=10)

    n = ds.particle_locations_n.isel(time=0)
    assert n.dims == ("height", "longitude")
    assert float(n.sel(height=500, longitude=-111.5)) == 0.1
    assert float(n.sel(height=1500, longitude=-111.5)) == 0.1
    assert (
        float(ds.particle_locations_e.isel(time=0).sel(height=2500, latitude=40.5))
        == 0.1
    )
    assert (
        float(ds.particle_locations_s.isel(time=0).sel(height=500, longitude=-112.5))
        == 0.1
    )
    assert (
        float(ds.particle_locations_w.isel(time=0).sel(height=19500, latitude=42.5))
        == 0.1
    )
    total = sum(float(ds[f"particle_locations_{e}"].sum()) for e in "nesw")
    assert total == pytest.approx(0.5)
    assert ds.height.values.tolist() == list(np.arange(500.0, 20000.0, 1000.0))


def test_an_exit_just_outside_the_grid_counts_in_the_nearest_cell():
    exits = exits_table(
        [
            (1, "n", -113.01, 43.0, 10.0),  # just west of the west edge
            (2, "n", -108.99, 43.0, 10.0),  # just east of the east edge
            (3, "e", -109.0, 43.02, 10.0),  # just north of the north edge
        ]
    )
    ds = layered([1, 2], lat=LAT, lon=LON).stilt.to_paris(exits=exits, n_particles=3)
    n = ds.particle_locations_n.isel(time=0, height=0).values
    assert n.tolist() == pytest.approx([1 / 3, 0, 0, 1 / 3])
    assert ds.particle_locations_e.isel(time=0, height=0).values[-1] == pytest.approx(
        1 / 3
    )


def test_exits_on_a_grid_across_180_degrees():
    lon = (179.5, 180.5, 181.5)
    exits = exits_table(
        [
            (1, "n", -179.2, 43.0, 10.0),  # 180.8 east: cell 1
            (2, "s", 179.9, 40.0, 10.0),  # cell 0
            (3, "n", -178.0, 43.0, 10.0),  # 182, just past the east edge: cell 2
        ]
    )
    ds = layered([1, 2], lat=LAT, lon=lon).stilt.to_paris(exits=exits, n_particles=4)
    assert ds.particle_locations_n.isel(time=0, height=0).values.tolist() == [
        0,
        0.25,
        0.25,
    ]
    assert ds.particle_locations_s.isel(time=0, height=0).values.tolist() == [
        0.25,
        0,
        0,
    ]


def test_exits_need_n_particles():
    with pytest.raises(ValueError, match="n_particles"):
        layered([1, 2]).stilt.to_paris(
            exits=exits_table([(1, "n", -112.0, 42.0, 10.0)])
        )


def test_a_particle_listed_twice_is_refused():
    exits = exits_table([(1, "n", -112.0, 42.0, 10.0), (1, "e", -111.0, 41.0, 10.0)])
    with pytest.raises(ValueError, match="twice"):
        layered([1, 2]).stilt.to_paris(exits=exits, n_particles=2)


def test_an_unknown_edge_is_refused():
    exits = exits_table([(1, "north", -112.0, 42.0, 10.0)])
    with pytest.raises(ValueError, match="Unknown grid edges"):
        layered([1, 2]).stilt.to_paris(exits=exits, n_particles=2)


def test_exits_with_particles_count_the_particles(footprint, particles):
    exits = exits_table([(1, "w", -113.0, 41.0, 10.0)])
    ds = footprint.stilt.to_paris(particles, exits=exits)
    assert float(ds.particle_locations_w.sum()) == pytest.approx(1 / 20)


def straight_lines():
    """Four particles that leave the 0.1 degree grid by its n, e, s and w edges, at 150, 1500, 2500 and 25000 m."""
    rows = []
    for particle, (dlon, dlat), zagl in [
        (1, (0, 1), 150.0),
        (2, (1, 0), 1500.0),
        (3, (0, -1), 2500.0),
        (4, (-1, 0), 25000.0),
    ]:
        for k in range(1, 6):
            rows.append(
                (
                    particle,
                    -10.0 * k,
                    -111.85 + 0.5 * k * dlon,
                    40.77 + 0.5 * k * dlat,
                    zagl,
                )
            )
    return pd.DataFrame(rows, columns=["particle", "age", "lon", "lat", "zagl"])


def test_particles_alone_use_their_grid_exits(footprint):
    lines = straight_lines()
    ds = footprint.stilt.to_paris(lines)

    # The method used is the one on the particles accessor, on the footprint's grid.
    exits = lines.stilt.grid_exits(footprint.stilt.grid)
    assert exits.edge.tolist() == ["n", "e", "s", "w"]
    same = footprint.stilt.to_paris(exits=exits, n_particles=4)
    xr.testing.assert_identical(ds, same)

    for edge, height, axis, at in [
        ("n", 500, "longitude", -111.85),
        ("e", 1500, "latitude", 40.77),
        ("s", 2500, "longitude", -111.85),
        ("w", 19500, "latitude", 40.77),
    ]:
        counts = ds[f"particle_locations_{edge}"].isel(time=0)
        assert float(counts.sum()) == 0.25
        assert (
            float(counts.sel(height=height).sel({axis: at}, method="nearest")) == 0.25
        )


def test_particles_that_stay_inside_are_not_counted(footprint, particles):
    walk = particles[particles.particle <= 3].copy()
    walk["lon"] = -111.85  # in the grid, over the whole run
    walk["lat"] = 40.77
    ds = footprint.stilt.to_paris(walk)
    assert float(ds.particle_locations_n.sum() + ds.particle_locations_s.sum()) == 0
    assert float(ds.particle_locations_e.sum() + ds.particle_locations_w.sum()) == 0


def test_n_particles_given_with_particles_is_the_denominator(footprint):
    ds = footprint.stilt.to_paris(straight_lines(), n_particles=8)
    assert float(ds.particle_locations_n.sum()) == pytest.approx(1 / 8)


def test_a_footprint_without_a_layer_time_is_refused():
    with pytest.raises(ValueError, match="time coordinate"):
        layered([1, 2]).drop_vars("time").stilt.to_paris()


def test_exits_without_a_needed_column_are_refused():
    exits = exits_table([(1, "n", -112.0, 42.0, 10.0)]).drop(columns="zagl")
    with pytest.raises(ValueError, match="no column zagl"):
        layered([1, 2]).stilt.to_paris(exits=exits, n_particles=2)


def test_n_particles_must_be_positive():
    exits = exits_table([(1, "n", -112.0, 42.0, 10.0)])
    with pytest.raises(ValueError, match="at least 1"):
        layered([1, 2]).stilt.to_paris(exits=exits, n_particles=0)


def test_a_grid_of_one_cell_has_no_cell_size_to_count_exits_in():
    exits = exits_table([(1, "n", -112.0, 42.0, 10.0)])
    with pytest.raises(ValueError, match="two or more cells"):
        layered([1, 2], lat=(40.5,), lon=(-112.5,)).stilt.to_paris(
            exits=exits, n_particles=2
        )


def test_a_footprint_without_settings_cannot_find_exits(particles):
    foot = layered([1, 2, 3])
    with pytest.raises(ValueError, match="stilt_footprint"):
        foot.stilt.to_paris(particles)


def test_many_receptors_concatenate_and_write(tmp_path):
    first = layered([1, 2, 3], lat=LAT, lon=LON).stilt.to_paris()
    later = layered([1, 2, 3], lat=LAT, lon=LON).assign_coords(
        receptor="202401151900_-111.85_40.77_5",
        time=("time", RECEPTOR_TIME + pd.to_timedelta([-2, -1, 0], unit="h")),
    )
    ds = xr.concat(
        [first, later.stilt.to_paris()], dim="time", combine_attrs="drop_conflicts"
    )
    path = tmp_path / "paris.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path, decode_timedelta=False) as back:
        assert back.time.values.tolist() == [
            RECEPTOR_TIME.value,
            (RECEPTOR_TIME + pd.Timedelta("1h")).value,
        ]
        np.testing.assert_array_equal(back.srr.values, ds.srr.values)
