"""Tests for stilt.observations.winds: variograms and wind-error scales."""

import numpy as np
import pandas as pd
import pytest

from stilt.config import ErrorParams
from stilt.observations import (
    VariogramFit,
    fit_variogram,
    variogram,
    wind_error_scales,
)


def _ar1(n: int, sigma: float, length: float, step: float, rng) -> np.ndarray:
    """Series on a regular grid with exponential correlation exp(-h / length)."""
    phi = np.exp(-step / length)
    out = np.empty(n)
    out[0] = sigma * rng.standard_normal()
    noise = sigma * np.sqrt(1 - phi**2) * rng.standard_normal(n - 1)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i - 1]
    return out


def _correlated(cov: np.ndarray, n_draws: int, rng) -> np.ndarray:
    """``n_draws`` independent draws from N(0, cov), shape (n_draws, len(cov))."""
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((n_draws, cov.shape[0])) @ chol.T


# -- variogram ---------------------------------------------------------------------


def test_variogram_recovers_exponential_time_correlation():
    rng = np.random.default_rng(1)
    series = _ar1(30_000, sigma=2.0, length=300.0, step=60.0, rng=rng)
    minutes = np.arange(series.size) * 60.0
    table = variogram(series, minutes, bins=np.arange(0, 1500, 60))
    assert list(table.columns) == ["lag", "gamma", "n"]
    # regular grid: mean lag of each bin is the multiple of 60 it contains
    np.testing.assert_allclose(table["lag"], np.arange(60, 1440, 60))  # [a, b) bins
    assert table["n"].iloc[0] == series.size - 1
    fit = fit_variogram(table["lag"], table["gamma"], sigma=2.0)
    assert fit.length == pytest.approx(300.0, rel=0.15)


def test_variogram_geo_lag_uses_great_circle_km():
    # two stations 1 degree of latitude apart (111.2 km), constant difference 3
    times = pd.date_range("2024-01-01", periods=50, freq="h")
    table = variogram(
        np.tile([0.0, 3.0], 50),
        np.tile([[-112.0, 40.0], [-112.0, 41.0]], (50, 1)),
        group=np.repeat(np.arange(50), 2),
        bins=np.arange(0, 200, 10),
    )
    assert len(table) == 1
    assert table["lag"].iloc[0] == pytest.approx(111.19, abs=0.1)
    assert table["gamma"].iloc[0] == pytest.approx(0.5 * 9.0)
    assert table["n"].iloc[0] == len(times)


def test_variogram_pairs_only_within_group_and_skips_zero_lag():
    errors = np.array([0.0, 1.0, 5.0, 0.0, 2.0])
    coord = np.array([0.0, 100.0, 100.0, 0.0, 100.0])
    # group a: (0,100) and (0,100 dup at same coord -> zero-lag pair skipped)
    group = ["a", "a", "a", "b", "b"]
    table = variogram(errors, coord, group=group, bins=[0.0, 150.0])
    # pairs at lag 100: a:(0,1),(0,2) ; b:(3,4) -> 3 pairs; a:(1,2) is zero lag
    assert table["n"].iloc[0] == 3
    assert table["gamma"].iloc[0] == pytest.approx(0.5 * (1.0 + 25.0 + 4.0) / 3)


def test_variogram_ignores_nan_and_pairs_beyond_last_edge():
    errors = np.array([0.0, np.nan, 2.0, 4.0])
    coord = np.array([0.0, 10.0, 20.0, 1000.0])
    table = variogram(errors, coord, bins=[0.0, 50.0])
    assert table["n"].iloc[0] == 1  # only (0, 2); the NaN and the far point drop
    assert table["gamma"].iloc[0] == pytest.approx(2.0)


def test_variogram_validates_inputs():
    with pytest.raises(ValueError, match="increasing"):
        variogram([1.0, 2.0], [0.0, 1.0], bins=[1.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        variogram([1.0, 2.0], [0.0], bins=[0.0, 1.0])
    with pytest.raises(ValueError, match="1-D"):
        variogram([1.0, 2.0], np.zeros((2, 3)), bins=[0.0, 1.0])


# -- fit_variogram ----------------------------------------------------------------


def test_fit_variogram_exact_with_fixed_and_free_sigma():
    truth = VariogramFit(sigma=2.5, length=400.0)
    lag = np.linspace(50, 3000, 30)
    gamma = truth(lag)
    fixed = fit_variogram(lag, gamma, sigma=2.5)
    assert fixed.length == pytest.approx(400.0, rel=1e-6)
    free = fit_variogram(lag, gamma)
    assert free.sigma == pytest.approx(2.5, rel=1e-4)
    assert free.length == pytest.approx(400.0, rel=1e-4)
    np.testing.assert_allclose(free(lag), gamma, rtol=1e-4)


def test_fit_variogram_needs_points():
    with pytest.raises(ValueError):
        fit_variogram([], [], sigma=1.0)
    with pytest.raises(ValueError):
        fit_variogram([100.0], [1.0])


# -- wind_error_scales ------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_errors():
    """Upper-air and surface error tables with known scales."""
    rng = np.random.default_rng(7)
    sigma, l_z, l_t, l_x = 2.0, 500.0, 300.0, 15.0

    # upper air: 400 launches, 12 h apart, levels every 100 m to 3 km, vertical
    # correlation exp(-dz / l_z), launches independent
    heights = np.arange(0.0, 3001.0, 100.0)
    cov_z = sigma**2 * np.exp(-np.abs(heights[:, None] - heights[None, :]) / l_z)
    launches = pd.date_range("2024-01-01", periods=400, freq="12h")
    u = _correlated(cov_z, len(launches), rng)
    v = _correlated(cov_z, len(launches), rng)
    upper = pd.DataFrame(
        {
            "time": np.repeat(launches, len(heights)),
            "height": np.tile(heights, len(launches)),
            "u_err": u.ravel(),
            "v_err": v.ravel(),
        }
    )

    # surface: 16 stations on a 4x4 grid ~10 km apart, hourly for 120 days;
    # separable covariance exp(-d / l_x) * exp(-dt / l_t)
    lon = -112.2 + 0.12 * np.arange(4)
    lat = 40.5 + 0.09 * np.arange(4)
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.ravel(), lats.ravel()
    from stilt.observations.selection import _haversine_km

    dist = np.array(
        [_haversine_km(lo, la, lons, lats) for lo, la in zip(lons, lats, strict=True)]
    )
    chol_x = np.linalg.cholesky(np.exp(-dist / l_x))
    hours = pd.date_range("2024-01-01", periods=120 * 24, freq="h")

    def field():
        z = np.stack([_ar1(len(hours), sigma, l_t, 60.0, rng) for _ in lons], axis=1)
        return z @ chol_x.T  # (time, station)

    us, vs = field(), field()
    surface = pd.DataFrame(
        {
            "time": np.repeat(hours, len(lons)),
            "site": np.tile([f"S{i}" for i in range(len(lons))], len(hours)),
            "lon": np.tile(lons, len(hours)),
            "lat": np.tile(lats, len(hours)),
            "u_err": us.ravel(),
            "v_err": vs.ravel(),
        }
    )
    return upper, surface, dict(sigma=sigma, l_z=l_z, l_t=l_t, l_x=l_x)


def test_wind_error_scales_recovers_known_scales(synthetic_errors):
    upper, surface, truth = synthetic_errors
    scales = wind_error_scales(upper, surface)
    assert scales.siguverr == pytest.approx(truth["sigma"], rel=0.1)
    assert scales.zcoruverr == pytest.approx(truth["l_z"], rel=0.2)
    assert scales.tluverr == pytest.approx(truth["l_t"], rel=0.2)
    assert scales.horcoruverr == pytest.approx(truth["l_x"], rel=0.25)

    fits = scales.fits
    assert set(fits.index) == {
        (c, k) for c in ("u", "v") for k in ("height", "time", "distance")
    }
    assert fits.loc[("u", "time"), "source"] == "surface"
    assert fits.loc[("u", "height"), "source"] == "upper"
    assert abs(fits.loc[("u", "height"), "bias"]) < 0.3
    assert set(scales.variograms) == set(fits.index)
    assert {"lag", "gamma", "n"} <= set(scales.variograms[("v", "distance")].columns)

    params = ErrorParams(**scales.to_dict())
    assert params.winderrtf == 1


def test_wind_error_scales_time_from_upper_without_surface(synthetic_errors):
    upper, _, truth = synthetic_errors
    scales = wind_error_scales(upper)
    assert scales.horcoruverr is None
    assert "horcoruverr" not in scales.to_dict()
    assert scales.fits.loc[("u", "time"), "source"] == "upper"
    # launches are independent: the first resolved lag (720 min) already sits at
    # the sill, so the fitted scale is far below the resolution of the data
    assert scales.tluverr < 720.0
    assert scales.zcoruverr == pytest.approx(truth["l_z"], rel=0.2)
    with pytest.raises(ValueError, match="surface table"):
        wind_error_scales(upper, time_from="surface")


def test_wind_error_scales_height_range_and_columns(synthetic_errors):
    upper, surface, _ = synthetic_errors
    with pytest.raises(ValueError, match="No upper-air rows"):
        wind_error_scales(upper, height_range=(5000.0, 6000.0))
    with pytest.raises(ValueError, match="missing columns"):
        wind_error_scales(upper.drop(columns=["height"]))
    with pytest.raises(ValueError, match="missing columns"):
        wind_error_scales(upper, surface.drop(columns=["lat"]))
    # restricting the layer changes the pairs but keeps the layout
    scales = wind_error_scales(upper, surface, height_range=(0.0, 1000.0))
    assert (
        scales.fits.loc[("u", "height"), "n_pairs"]
        < wind_error_scales(upper, surface).fits.loc[("u", "height"), "n_pairs"]
    )
