"""Tests for ``particles.stilt.to_cf_trajectories``."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.fixtures.particles import random_walk

RECEPTOR_TIME = pd.Timestamp("2024-01-15 18:00")


@pytest.fixture
def particles():
    """A backward particle table of 20 particles over 48 hours."""
    return random_walk()


def test_one_trajectory_per_particle_padded_at_the_end(particles):
    ds = particles.stilt.to_cf_trajectories()

    assert dict(ds.sizes) == {"trajectory": 20, "obs": 48 * 6 - 1}
    assert ds.trajectory.values.tolist() == list(range(1, 21))
    # Particle 20 stops 19 steps before particle 1, the longest.
    last = ds.sel(trajectory=20)
    assert int(last.lon.notnull().sum()) == 48 * 6 - 20
    assert last.lon.isnull()[-19:].all()


def test_values_follow_the_particle_table(particles):
    ds = particles.stilt.to_cf_trajectories()

    one = particles[particles.particle == 3].sort_values("age", ascending=False)
    row = ds.sel(trajectory=3)
    n = len(one)
    np.testing.assert_array_equal(row.lon.values[:n], one.lon.values)
    np.testing.assert_array_equal(row.zagl.values[:n], one.zagl.values)
    np.testing.assert_array_equal(row.age.values[:n], one.age.values)
    np.testing.assert_array_equal(row.foot.values[:n], one.foot.values)
    np.testing.assert_array_equal(row.time.values[:n], one.time.values)


def test_obs_counts_steps_from_the_release():
    table = pd.DataFrame(
        {
            "particle": [1, 1, 1],
            "age": [-3.0, -1.0, -2.0],
            "lon": [3.0, 1.0, 2.0],
            "lat": [0.0, 0.0, 0.0],
            "zagl": [1.0, 1.0, 1.0],
        }
    )
    ds = table.stilt.to_cf_trajectories(receptor_time=RECEPTOR_TIME)
    assert ds.lon.values.tolist() == [[1.0, 2.0, 3.0]]
    # A backward run's time decreases along obs.
    assert (np.diff(ds.time.values[0]) < np.timedelta64(0, "ns")).all()


def test_a_forward_run_runs_forward_in_time():
    table = pd.DataFrame(
        {
            "particle": [1, 1],
            "age": [10.0, 20.0],
            "lon": [0.0, 1.0],
            "lat": [0.0, 0.0],
            "zagl": [1.0, 1.0],
        }
    )
    ds = table.stilt.to_cf_trajectories(receptor_time=RECEPTOR_TIME)
    expected = np.array(
        ["2024-01-15T18:10", "2024-01-15T18:20"], dtype="datetime64[ns]"
    )
    np.testing.assert_array_equal(ds.time.values[0], expected)


def test_time_comes_from_the_receptor_time_without_a_time_column(particles):
    from_column = particles.stilt.to_cf_trajectories()
    from_receptor = particles.drop(columns="time").stilt.to_cf_trajectories(
        receptor_time=RECEPTOR_TIME
    )
    xr.testing.assert_identical(from_column.time, from_receptor.time)


def test_times_with_a_time_zone_become_utc(particles):
    utc = particles.stilt.to_cf_trajectories()
    denver = particles.assign(
        time=particles.time.dt.tz_localize("UTC").dt.tz_convert("America/Denver")
    )
    assert (
        denver.stilt.to_cf_trajectories().time.values[:, 0] == utc.time.values[:, 0]
    ).all()

    local_start = RECEPTOR_TIME.tz_localize("UTC").tz_convert("America/Denver")
    from_receptor = particles.drop(columns="time").stilt.to_cf_trajectories(
        receptor_time=local_start
    )
    xr.testing.assert_identical(from_receptor.time, utc.time)


def test_cf_metadata(particles):
    ds = particles.stilt.to_cf_trajectories()

    assert ds.attrs["Conventions"] == "CF-1.10"
    assert ds.attrs["featureType"] == "trajectory"
    assert ds.trajectory.attrs["cf_role"] == "trajectory_id"
    assert ds.time.attrs["standard_name"] == "time"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.zagl.attrs == {
        "standard_name": "height",
        "long_name": "height above ground",
        "units": "m",
        "positive": "up",
    }
    assert ds.age.attrs["units"] == "minutes"
    assert ds.foot.attrs["units"] == "ppm m2 s umol-1"
    assert ds.mlht.attrs["units"] == "m"
    assert set(ds.coords) == {"trajectory", "time", "lon", "lat", "zagl"}


def test_a_column_without_known_units_is_kept_without_units(particles):
    ds = particles.assign(mine=1.5, label="x").stilt.to_cf_trajectories()
    assert ds.mine.attrs == {}
    assert "label" not in ds


def test_a_netcdf_file_reads_back_the_same(particles, tmp_path):
    ds = particles.stilt.to_cf_trajectories()
    path = tmp_path / "trajectories.nc"
    ds.to_netcdf(path)

    # Older xarray decodes a variable in minutes as a duration; keep the numbers.
    with xr.open_dataset(path, decode_timedelta=False) as back:
        xr.testing.assert_identical(back.load(), ds)
        assert back.age.dtype == np.float64
    with xr.open_dataset(path, decode_times=False) as raw:
        # Minutes since the earliest time; a missing time is NaN.
        assert raw.time.attrs["units"].startswith("minutes since 2024-01-13")
        assert raw.time.attrs["calendar"] == "standard"
        assert raw.time.dtype == np.float64
        assert np.isnan(raw.time.values[-1, -1])


def test_cf_xarray_finds_the_coordinates(particles):
    cf_xarray = pytest.importorskip("cf_xarray")
    assert cf_xarray  # registers the .cf accessor
    ds = particles.stilt.to_cf_trajectories()
    assert ds.cf["longitude"].name == "lon"
    assert ds.cf["latitude"].name == "lat"
    assert ds.cf["time"].name == "time"
    assert ds.cf.cf_roles == {"trajectory_id": ["trajectory"]}


def test_trajan_opens_the_file(particles, tmp_path):
    pytest.importorskip("trajan")
    path = tmp_path / "trajectories.nc"
    particles.stilt.to_cf_trajectories().to_netcdf(path)

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        traj = ds.traj
        assert traj.trajectory_dim == "trajectory"
        assert traj.obs_dim == "obs"
        assert traj.time_varname == "time"
        np.testing.assert_array_equal(traj.tx.values, ds.lon.values)
        np.testing.assert_array_equal(traj.ty.values, ds.lat.values)


@pytest.mark.parametrize("column", ["particle", "age", "lon", "lat", "zagl"])
def test_a_missing_column_is_refused(particles, column):
    with pytest.raises(ValueError, match=f"no column {column}"):
        particles.drop(columns=column).stilt.to_cf_trajectories()


def test_no_time_and_no_receptor_time_is_refused(particles):
    with pytest.raises(ValueError, match="receptor_time"):
        particles.drop(columns="time").stilt.to_cf_trajectories()


def test_several_realizations_are_refused(particles):
    two = pd.concat([particles.assign(realization=0), particles.assign(realization=1)])
    with pytest.raises(ValueError, match="several realizations"):
        two.stilt.to_cf_trajectories()


def test_one_realization_is_recorded(particles):
    ds = particles.assign(realization=3).stilt.to_cf_trajectories()
    assert ds.attrs["realization"] == 3
    assert "realization" not in ds


def test_a_repeated_age_is_refused(particles):
    with pytest.raises(ValueError, match="same age"):
        pd.concat([particles, particles.iloc[:1]]).stilt.to_cf_trajectories()


def test_an_empty_table_gives_an_empty_dataset(particles):
    ds = particles.iloc[:0].stilt.to_cf_trajectories()
    assert dict(ds.sizes) == {"trajectory": 0, "obs": 0}
    assert ds.attrs["featureType"] == "trajectory"
