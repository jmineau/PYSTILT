"""Tests for the enhancement of footprints and particles."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from runtraj.spatial import horizontal_dims

from stilt.footprint import Grid
from stilt.footprint.config import FootprintConfig

from .fixtures.footprints import as_footprint


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


def _particles():
    # two particles; particle 2's second row is outside the flux field
    return pd.DataFrame(
        {
            "particle": [1, 1, 2, 2, 3],
            "lon": [-112.0, -111.0, -110.0, -100.0, -111.0],
            "lat": [40.0, 40.0, 41.0, 41.0, 41.0],
            "foot": [2.0, 3.0, 1.0, 5.0, 0.0],
            "time": pd.to_datetime(["2023-01-01"] * 5),
        }
    )


def test_particles_enhancement_sums_foot_times_flux_per_particle():
    per_particle = _particles().stilt.enhancement(_flux())

    assert per_particle.index.tolist() == [1, 2, 3]
    assert per_particle.tolist() == [2.0 * 0 + 3.0 * 1, 1.0 * 5 + 0.0, 0.0]
    assert per_particle.name == "enhancement"


def test_particles_enhancement_time_varying_needs_time():
    flux = _flux().expand_dims(time=pd.to_datetime(["2023-01-01"]))
    assert _particles().stilt.enhancement(flux).tolist() == [3.0, 5.0, 0.0]
    with pytest.raises(ValueError, match="time"):
        _particles().drop(columns="time").stilt.enhancement(flux)


def test_particles_enhancement_without_foot_says_how_to_get_one():
    """A positions-only model's ``sim.particles`` has no ``foot``: the footprint stage stores its own nowhere."""
    with pytest.raises(ValueError, match="no 'foot' column.*footprint_particles"):
        _particles().drop(columns="foot").stilt.enhancement(_flux())


def _footprint(point_release, values):
    grid = Grid(xmin=-112.5, xmax=-109.5, ymin=39.5, ymax=41.5, xres=1.0, yres=1.0)
    times = pd.to_datetime(["2023-01-01 12:00", "2023-01-01 11:00"])
    data = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )
    return as_footprint(data, point_release, FootprintConfig(grid=grid), "test")


def test_footprint_enhancement_is_foot_times_flux_per_time_step(point_release):
    foot = _footprint(point_release, [np.ones((2, 3)), 2 * np.ones((2, 3))])

    enhancement = foot.stilt.enhancement(_flux())  # flux cells 0..5 sum to 15

    assert enhancement.dims == ("time",)
    assert enhancement.to_numpy().tolist() == [15.0, 30.0]
    assert float(enhancement.sum()) == 45.0


def test_footprint_enhancement_does_not_depend_on_dimension_order(point_release):
    foot = _footprint(point_release, [np.ones((2, 3)), 2 * np.ones((2, 3))])
    flux = _flux()
    expected = foot.stilt.enhancement(flux)

    swapped = foot.transpose("time", "lon", "lat")
    assert swapped.stilt.enhancement(flux).to_numpy().tolist() == [15.0, 30.0]
    xr.testing.assert_equal(swapped.stilt.enhancement(flux), expected)


def test_footprint_enhancement_of_a_time_summed_footprint_is_the_total(
    point_release,
):
    foot = _footprint(point_release, [np.ones((2, 3)), 2 * np.ones((2, 3))])
    total = foot.sum("time", keep_attrs=True)

    assert float(total.stilt.enhancement(_flux())) == 45.0
    flux_in_time = _flux().expand_dims(time=pd.to_datetime(["2023-01-01 11:00"]))
    with pytest.raises(ValueError, match="no time dimension"):
        total.stilt.enhancement(flux_in_time)


def test_footprint_enhancement_samples_flux_on_its_own_grid(point_release):
    foot = _footprint(point_release, [np.ones((2, 3)), np.zeros((2, 3))])
    # a single-cell axis has no spacing to bound, so it covers every point
    uniform = xr.DataArray(
        [[7.0]], dims=["lat", "lon"], coords={"lat": [40.5], "lon": [-111.0]}
    )
    assert float(foot.stilt.enhancement(uniform)[0]) == 7.0 * 6

    # two 2-degree cells reach only one degree past their centres
    narrow = xr.DataArray(
        [[7.0, 7.0]],
        dims=["lat", "lon"],
        coords={"lat": [40.5], "lon": [-114.5, -112.5]},
    )
    assert float(foot.stilt.enhancement(narrow)[0]) == 7.0 * 2  # only the -112 column


def test_footprint_enhancement_time_varying_flux(point_release):
    foot = _footprint(point_release, [np.ones((2, 3)), np.ones((2, 3))])
    flux = xr.DataArray(
        np.stack([np.ones((2, 3)), 10 * np.ones((2, 3))]),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.to_datetime(["2023-01-01 11:00", "2023-01-01 12:00"]),
            "lat": [40.0, 41.0],
            "lon": [-112.0, -111.0, -110.0],
        },
    )

    assert foot.stilt.enhancement(flux).to_numpy().tolist() == [60.0, 6.0]


def test_footprint_enhancement_refuses_a_finer_flux(point_release):
    foot = _footprint(point_release, [np.ones((2, 3)), np.ones((2, 3))])
    lons = np.arange(-112.0, -109.9, 0.25)
    fine = _flux(np.ones((2, lons.size)), lons=tuple(lons))
    with pytest.raises(ValueError, match="finer than the footprint's"):
        foot.stilt.enhancement(fine)


def test_footprint_enhancement_warns_when_cells_of_one_size_are_offset(
    point_release, recwarn
):
    import warnings

    foot = _footprint(point_release, [np.ones((2, 3)), np.ones((2, 3))])
    with pytest.warns(UserWarning, match="0.50 of a cell"):
        foot.stilt.enhancement(_flux(lons=(-112.5, -111.5, -110.5, -109.5)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        foot.stilt.enhancement(_flux())  # aligned
        foot.stilt.enhancement(
            _flux(np.ones((1, 2)), lons=(-112.3, -110.3), lats=(40.0,))
        )  # coarser
