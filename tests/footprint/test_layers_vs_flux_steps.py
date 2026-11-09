"""A footprint layer longer than a flux step is refused: it would take one step's flux whole."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

import stilt
from stilt.footprint import FootprintConfig, jacobian

T0 = pd.Timestamp("2023-07-20 00:00")
DAY = (pd.Timestamp("2023-07-19 00:00"), pd.Timestamp("2023-07-20 00:00"))
GRID = stilt.Grid(xmin=-113, xmax=-111, ymin=39.5, ymax=41.5, xres=0.1, yres=0.1)
TOO_SHORT = "layers are 1 days 00:00:00 long but a step of the flux is 0 days 01:00:00"


def _footprint(tres: str) -> xr.DataArray:
    """Return the footprint of 10 stationary particles over 24 hours, in layers of *tres*."""
    ages = -np.arange(1, 24 * 60 + 1, dtype=float)
    particles = pd.concat(
        pd.DataFrame(
            {
                "particle": float(k),
                "age": ages,
                "lon": -112.0,
                "lat": 40.5,
                "zagl": 10.0,
                "foot": 0.01,
                "time": T0 + pd.to_timedelta(ages, unit="min"),
            }
        )
        for k in range(1, 11)
    )
    receptor = stilt.PointReceptor(
        time=T0, longitude=-112.0, latitude=40.5, altitude=10
    )
    return stilt.calc_footprint(particles, receptor, GRID, tres=tres)


def _flux(starts: pd.DatetimeIndex) -> xr.DataArray:
    """Return a flux of 10 from 12 to 18 UTC and 1 otherwise, one step from each of *starts*."""
    values = np.where((starts.hour >= 12) & (starts.hour < 18), 10.0, 1.0)
    return xr.DataArray(
        values[:, None, None] * np.ones((len(starts), 3, 3)),
        dims=("time", "lat", "lon"),
        coords={
            "time": starts,
            "lat": [39.0, 40.5, 42.0],
            "lon": [-114.0, -112.0, -110.0],
        },
    )


HOURLY = pd.date_range(*DAY, freq="1h")


def test_a_day_layer_over_an_hourly_flux_is_refused_by_enhancement():
    # on main this came out 69% low, with no warning
    with pytest.raises(ValueError, match=TOO_SHORT):
        _footprint("1D").stilt.enhancement(_flux(HOURLY))


def test_the_error_says_what_to_do():
    with pytest.raises(ValueError, match=r"tres no longer than .* average the flux"):
        _footprint("1D").stilt.enhancement(_flux(HOURLY))


def test_cf_time_bounds_give_the_step_lengths():
    bounds = pd.IntervalIndex.from_arrays(HOURLY[:-1], HOURLY[1:], closed="left")
    flux = _flux(HOURLY[:-1]).assign_coords(time=bounds)
    with pytest.raises(ValueError, match=TOO_SHORT):
        _footprint("1D").stilt.enhancement(flux)
    # a label in the middle of each step does not change the steps' lengths
    assert float(_footprint("1h").stilt.enhancement(flux).sum()) > 0


@pytest.mark.parametrize("tres", ["1h", "3h", "6h"])
def test_layers_no_longer_than_the_flux_steps_agree_with_each_other(tres):
    # 6-hour steps starting at 00, 06, 12 and 18 line up with all these layers
    flux = _flux(pd.date_range(*DAY, freq="6h"))
    total = float(_footprint(tres).stilt.enhancement(flux).sum())
    reference = float(_footprint("1h").stilt.enhancement(flux).sum())
    assert total == pytest.approx(reference)


def test_a_static_flux_is_allowed_with_any_layers():
    foot = _footprint("1D")
    steady = _flux(HOURLY).isel(time=[0])  # one time
    assert float(foot.stilt.enhancement(steady).sum()) == pytest.approx(
        float(foot.sum()) * 1.0
    )
    nothing_in_time = steady.isel(time=0, drop=True)  # no time dimension
    assert float(foot.stilt.enhancement(nothing_in_time).sum()) == pytest.approx(
        float(foot.sum()) * 1.0
    )


def test_a_flux_coarser_than_the_layers_is_allowed():
    flux = _flux(pd.date_range("2023-07-19 00:00", "2023-07-21 00:00", freq="1D"))
    assert np.isfinite(float(_footprint("6h").stilt.enhancement(flux).sum()))


def test_aggregate_refuses_time_bins_shorter_than_the_layers():
    foot = _footprint("1D")
    hourly = pd.interval_range(*DAY, freq="1h", closed="left")
    with pytest.raises(ValueError, match=r"step of the time bins is 0 days 01:00:00"):
        foot.stilt.aggregate(GRID, hourly)


def test_aggregate_takes_bins_as_long_as_the_layers():
    foot = _footprint("6h")
    bins = pd.interval_range(*DAY, freq="6h", closed="left")
    by_bin = foot.stilt.aggregate(GRID, bins)
    assert by_bin.to_numpy().sum() == pytest.approx(float(foot.sum()))
    one_day = pd.interval_range(*DAY, freq="1D", closed="left")
    assert foot.stilt.aggregate(GRID, one_day).to_numpy().sum() == pytest.approx(
        float(foot.sum())
    )


def test_jacobian_refuses_time_bins_shorter_than_the_layers():
    empty = pa.table(
        {
            "receptor": pa.array([], pa.string()),
            "hour": pa.array([], pa.int32()),
            "y": pa.array([], pa.int32()),
            "x": pa.array([], pa.int32()),
            "foot": pa.array([], pa.float32()),
        }
    )
    hourly = pd.interval_range(*DAY, freq="1h", closed="left")
    with pytest.raises(ValueError, match=r"step of the time bins is 0 days 01:00:00"):
        jacobian(empty, FootprintConfig(grid=GRID, tres="1D"), GRID, hourly, [])
