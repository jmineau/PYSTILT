"""Which flux step ``enhancement`` takes for each footprint hour or particle row."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import stilt
from stilt.sampling import sample_field

T0 = pd.Timestamp("2023-07-20 12:00")


@pytest.fixture(scope="module")
def particles():
    """Return 30 synthetic particles going 24 hours back from ``T0``."""
    rng = np.random.default_rng(0)
    ages = -np.arange(1, 24 * 60 + 1, dtype=float)
    rows = []
    for k in range(1, 31):
        lon = -112.0 + np.cumsum(rng.normal(0, 0.002, ages.size))
        lat = 40.5 + np.cumsum(rng.normal(0, 0.002, ages.size))
        rows.append(
            pd.DataFrame(
                {
                    "particle": float(k),
                    "age": ages,
                    "lon": lon,
                    "lat": lat,
                    "zagl": 10.0,
                    "foot": 0.01,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)
    p["time"] = T0 + pd.to_timedelta(p["age"], unit="min")
    return p


@pytest.fixture(scope="module")
def grid():
    return stilt.Grid(xmin=-113, xmax=-111, ymin=39.5, ymax=41.5, xres=0.02, yres=0.02)


@pytest.fixture(scope="module")
def foot(particles, grid):
    receptor = stilt.PointReceptor(
        time=T0, longitude=-112.0, latitude=40.5, altitude=10
    )
    return stilt.calc_footprint(particles, receptor, grid)


def _step_flux(starts):
    """Return a flux uniform in space whose value is the step number, from 1."""
    values = np.arange(1.0, len(starts) + 1)
    flux = xr.DataArray(
        values[:, None, None] * np.ones((len(starts), 3, 3)),
        dims=("time", "lat", "lon"),
        coords={
            "time": starts,
            "lat": [39.0, 40.5, 42.0],
            "lon": [-114.0, -112.0, -110.0],
        },
    )
    return flux, values


def _by_hand(foot, starts, values):
    """Return the sum over hours of footprint total times the value of the step holding the hour."""
    per_hour = foot.sum(["lat", "lon"]).to_pandas()
    step = (
        np.searchsorted(starts.to_numpy(), per_hour.index.to_numpy(), side="right") - 1
    )
    assert (step >= 0).all()
    return float((per_hour.to_numpy() * values[step]).sum())


STARTS = {
    "monthly": pd.DatetimeIndex(["2023-06-01", "2023-07-01", "2023-08-01"]),
    "3-hourly": pd.date_range("2023-07-19 00:00", "2023-07-21 00:00", freq="3h"),
    "hourly": pd.date_range("2023-07-19 00:00", "2023-07-21 00:00", freq="1h"),
}


@pytest.mark.parametrize("name", list(STARTS))
def test_enhancement_takes_the_step_that_starts_at_or_before_each_hour(
    foot, particles, grid, name
):
    starts = STARTS[name]
    flux, values = _step_flux(starts)
    expected = _by_hand(foot, starts, values)

    assert float(foot.stilt.enhancement(flux).sum()) == pytest.approx(expected)
    # the particles path takes the step of each row's own time
    assert float(particles.stilt.enhancement(flux).mean()) == pytest.approx(expected)


def test_a_monthly_flux_gives_july_20_the_july_flux(foot):
    flux, _ = _step_flux(STARTS["monthly"])  # June 1: 1, July 1: 2, August 1: 3
    total = float(foot.sum().item())
    assert float(foot.stilt.enhancement(flux).sum()) == pytest.approx(2.0 * total)


@pytest.mark.parametrize("name", list(STARTS))
def test_enhancement_agrees_with_aggregate_onto_the_footprint_grid(foot, grid, name):
    starts = STARTS[name]
    flux, values = _step_flux(starts)
    ends = starts[1:].append(starts[-1:] + (starts[-1] - starts[-2]))
    bins = pd.IntervalIndex.from_arrays(starts, ends, closed="left")

    aggregated = foot.stilt.aggregate(grid, bins)  # cells by bins

    from_aggregate = float((aggregated.sum(axis=0).to_numpy() * values).sum())
    assert float(foot.stilt.enhancement(flux).sum()) == pytest.approx(from_aggregate)


def test_enhancement_by_hour_follows_the_step_boundaries(foot):
    starts = STARTS["3-hourly"]
    flux, values = _step_flux(starts)

    per_hour = foot.stilt.enhancement(flux).to_series()
    totals = foot.sum(["lat", "lon"]).to_series()

    for hour, value in per_hour.items():
        step = (
            np.searchsorted(starts.to_numpy(), hour.to_datetime64(), side="right") - 1
        )
        assert value == pytest.approx(totals[hour] * values[step])


def test_an_hour_before_the_first_step_gets_zero_flux_and_one_after_the_last_takes_the_last(
    foot,
):
    # the footprint hours run from 2023-07-19 12:00 to 2023-07-20 11:00
    late, _ = _step_flux(pd.DatetimeIndex(["2023-07-20 00:00", "2023-07-20 06:00"]))
    per_hour = foot.stilt.enhancement(late).to_series()
    totals = foot.sum(["lat", "lon"]).to_series()
    before = per_hour.index < pd.Timestamp("2023-07-20 00:00")
    assert (per_hour[before] == 0).all()
    from_midnight = (per_hour.index >= pd.Timestamp("2023-07-20 00:00")) & (
        per_hour.index < pd.Timestamp("2023-07-20 06:00")
    )
    after = per_hour.index >= pd.Timestamp("2023-07-20 06:00")
    assert per_hour[from_midnight].to_numpy() == pytest.approx(
        totals[from_midnight].to_numpy() * 1.0
    )
    # no end is known for the last step, so it holds for every later hour
    assert per_hour[after].to_numpy() == pytest.approx(totals[after].to_numpy() * 2.0)


def test_labels_at_the_middle_of_a_step_are_relabeled_by_half_a_step(foot):
    starts = STARTS["3-hourly"]
    flux, _ = _step_flux(starts)
    centered = flux.assign_coords(time=starts + pd.Timedelta("90min"))
    relabeled = centered.assign_coords(time=centered.time - pd.Timedelta("90min"))

    xr.testing.assert_equal(
        foot.stilt.enhancement(relabeled), foot.stilt.enhancement(flux)
    )


def test_flux_labels_that_do_not_increase_are_refused(foot):
    flux, _ = _step_flux(STARTS["3-hourly"])
    with pytest.raises(ValueError, match="must increase"):
        foot.stilt.enhancement(flux.isel(time=slice(None, None, -1)))


def test_sample_field_reads_labels_as_step_starts_only_when_asked():
    flux, _ = _step_flux(pd.DatetimeIndex(["2023-01-01 00:00", "2023-01-01 06:00"]))
    times = [
        "2022-12-31 23:00",
        "2023-01-01 05:59",
        "2023-01-01 06:00",
        "2023-02-01 00:00",
    ]
    kwargs = dict(x=[-112.0] * 4, y=[40.5] * 4, times=times, fill_value=0.0)

    assert sample_field(flux, step_starts=True, **kwargs).tolist() == [
        0.0,
        1.0,
        2.0,
        2.0,
    ]
    # the default stays the nearest label, which other fields (met, a mole fraction) use
    assert sample_field(flux, **kwargs).tolist() == [1.0, 2.0, 2.0, 2.0]
