"""Tests for stilt.flux and Footprint.enhancement."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid
from stilt.flux import horizontal_dims, particle_enhancement, sample_flux
from stilt.footprint import Footprint


def _flux(values=None, lons=(-112.0, -111.0, -110.0), lats=(40.0, 41.0)):
    values = (
        np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
        if values is None
        else values
    )
    return xr.DataArray(
        values, dims=["lat", "lon"], coords={"lat": list(lats), "lon": list(lons)}
    )


def test_horizontal_dims_lonlat_and_projected():
    assert horizontal_dims(_flux()) == ("lat", "lon")
    projected = xr.DataArray(
        np.zeros((2, 2)), dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )
    assert horizontal_dims(projected) == ("y", "x")
    with pytest.raises(ValueError, match="Expected"):
        horizontal_dims(xr.DataArray(np.zeros((2, 2)), dims=["a", "b"]))


def test_sample_flux_nearest_cell_and_outside_is_zero():
    flux = _flux()  # rows: lat 40 -> [0,1,2], lat 41 -> [3,4,5]

    sampled = sample_flux(
        flux,
        x=[-112.0, -111.4, -110.6, -109.4, -112.6, -111.0],
        y=[40.0, 40.4, 40.6, 41.0, 40.0, 41.6],
    )

    # nearest centres: (-112,40)=0, (-111,40)=1, (-111,41)=4, outside east, outside west, outside north
    assert sampled.tolist() == [0.0, 1.0, 4.0, 0.0, 0.0, 0.0]


def test_sample_flux_edges_extend_half_a_cell():
    flux = _flux()
    inside = sample_flux(flux, x=[-112.49, -109.51], y=[39.51, 41.49])
    outside = sample_flux(flux, x=[-112.51, -109.49], y=[39.49, 41.51])
    assert inside.tolist() == [0.0, 5.0]
    assert outside.tolist() == [0.0, 0.0]


def test_sample_flux_descending_latitude_axis():
    flux = _flux(values=np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]), lats=(41.0, 40.0))
    assert sample_flux(flux, x=[-111.0, -111.0], y=[40.0, 41.0]).tolist() == [1.0, 4.0]


def test_sample_flux_nan_cells_count_as_zero():
    values = np.arange(6, dtype=float).reshape(2, 3)
    values[0, 1] = np.nan
    assert sample_flux(_flux(values), x=[-111.0], y=[40.0]).tolist() == [0.0]


def test_sample_flux_time_varying_uses_nearest_time():
    times = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 06:00"])
    flux = xr.DataArray(
        np.stack([np.full((2, 3), 1.0), np.full((2, 3), 10.0)]),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )

    sampled = sample_flux(
        flux,
        x=[-111.0, -111.0, -111.0],
        y=[40.0, 40.0, 40.0],
        times=["2023-01-01 02:00", "2023-01-01 05:00", "2023-01-02 00:00"],
    )

    assert sampled.tolist() == [1.0, 10.0, 10.0]
    with pytest.raises(ValueError, match="pass times"):
        sample_flux(flux, x=[-111.0], y=[40.0])


def test_sample_flux_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        sample_flux(_flux(), x=[0.0, 1.0], y=[0.0])


def _particles():
    # two particles; particle 2's second row is outside the flux field
    return pd.DataFrame(
        {
            "indx": [1, 1, 2, 2, 3],
            "long": [-112.0, -111.0, -110.0, -100.0, -111.0],
            "lati": [40.0, 40.0, 41.0, 41.0, 41.0],
            "foot": [2.0, 3.0, 1.0, 5.0, 0.0],
            "datetime": pd.to_datetime(["2023-01-01"] * 5),
        }
    )


def test_particle_enhancement_sums_foot_times_flux_per_particle():
    per_particle = particle_enhancement(_particles(), _flux())

    assert per_particle.index.tolist() == [1, 2, 3]
    assert per_particle.tolist() == [2.0 * 0 + 3.0 * 1, 1.0 * 5 + 0.0, 0.0]
    assert per_particle.name == "enhancement"


def test_particle_enhancement_time_varying_needs_datetime():
    flux = _flux().expand_dims(time=pd.to_datetime(["2023-01-01"]))
    assert particle_enhancement(_particles(), flux).tolist() == [3.0, 5.0, 0.0]
    with pytest.raises(ValueError, match="datetime"):
        particle_enhancement(_particles().drop(columns="datetime"), flux)


def _footprint(point_receptor, values):
    grid = Grid(xmin=-112.5, xmax=-109.5, ymin=39.5, ymax=41.5, xres=1.0, yres=1.0)
    times = pd.to_datetime(["2023-01-01 12:00", "2023-01-01 11:00"])
    data = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )
    return Footprint(point_receptor, FootprintConfig(grid=grid), data, name="test")


def test_footprint_enhancement_is_foot_times_flux_per_time_step(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), 2 * np.ones((2, 3))])

    enhancement = foot.enhancement(_flux())  # flux cells 0..5 sum to 15

    assert enhancement.dims == ("time",)
    assert enhancement.to_numpy().tolist() == [15.0, 30.0]
    assert float(enhancement.sum()) == 45.0


def test_footprint_enhancement_samples_flux_on_its_own_grid(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), np.zeros((2, 3))])
    # a single-cell axis has no spacing to bound, so it covers every point
    uniform = xr.DataArray(
        [[7.0]], dims=["lat", "lon"], coords={"lat": [40.5], "lon": [-111.0]}
    )
    assert float(foot.enhancement(uniform)[0]) == 7.0 * 6

    # two 2-degree cells reach only one degree past their centres
    narrow = xr.DataArray(
        [[7.0, 7.0]],
        dims=["lat", "lon"],
        coords={"lat": [40.5], "lon": [-114.5, -112.5]},
    )
    assert float(foot.enhancement(narrow)[0]) == 7.0 * 2  # only the -112 column


def test_footprint_enhancement_time_varying_flux(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), np.ones((2, 3))])
    flux = xr.DataArray(
        np.stack([np.ones((2, 3)), 10 * np.ones((2, 3))]),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.to_datetime(["2023-01-01 11:00", "2023-01-01 12:00"]),
            "lat": [40.0, 41.0],
            "lon": [-112.0, -111.0, -110.0],
        },
    )

    assert foot.enhancement(flux).to_numpy().tolist() == [60.0, 6.0]
