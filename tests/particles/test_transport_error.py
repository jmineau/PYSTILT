"""Tests for stilt.particles.transport_error and sim.transport_error."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.particles import TransportError, transport_error
from stilt.transforms import AveragingKernel

FLUX = xr.DataArray(
    np.ones((3, 3)),
    dims=["lat", "lon"],
    coords={"lat": [40.0, 41.0, 42.0], "lon": [-112.0, -111.0, -110.0]},
)


def _particles(release_height, foot):
    """Particles with one row each, so a particle's enhancement on FLUX is its foot."""
    n = len(foot)
    return pd.DataFrame(
        {
            "particle": np.arange(1, n + 1),
            "age": np.zeros(n),
            "lon": np.full(n, -111.0),
            "lat": np.full(n, 41.0),
            "release_height": np.asarray(release_height, dtype=float),
            "foot": np.asarray(foot, dtype=float),
            "time": pd.to_datetime(["2023-01-01"] * n),
        }
    )


def _pair_correlation(heights, length_scale):
    """Correlation of every particle pair, the direct N x N way."""
    h = np.asarray(heights, dtype=float)
    return np.exp(-np.abs(h[:, None] - h[None, :]) / length_scale)


def _var_of_var(values):
    """Sampling variance of the variance of *values*: (m4 - var²) / n."""
    dev = np.asarray(values) - np.mean(values)
    return (np.mean(dev**4) - np.mean(dev**2) ** 2) / len(dev)


def _column(n_levels=4, per_level=200, spread=1.0, seed=0, level_spacing=500.0):
    """Column particles: each particle's foot sum ~ N(10, spread²); one row each."""
    rng = np.random.default_rng(seed)
    n = n_levels * per_level
    release_height = np.repeat(np.arange(n_levels) * level_spacing + 250.0, per_level)
    return pd.DataFrame(
        {
            "particle": np.arange(1, n + 1),
            "age": np.zeros(n),
            "lon": np.full(n, -111.0),
            "lat": np.full(n, 41.0),
            "release_height": release_height,
            "foot": 10.0 + spread * rng.standard_normal(n),
            "time": pd.to_datetime(["2023-01-01"] * n),
        }
    )


# -- transport_error ---------------------------------------------------------------


def test_no_perturbation_gives_zero_error_and_the_enhancement():
    p = _column()
    result = transport_error(p, p.copy(), FLUX)

    assert isinstance(result, TransportError)
    assert result.variance == pytest.approx(0.0)
    assert result.sd == pytest.approx(0.0)
    assert result.enhancement == pytest.approx(p["foot"].mean(), rel=1e-12)
    assert result.enhancement_perturbed == result.enhancement
    assert result.noise > 0  # a variance of 200 particles is not exact
    assert len(result.levels) == 4
    assert result.levels["n"].tolist() == [200] * 4
    assert result.levels["weight"].sum() == pytest.approx(1.0)
    assert result.levels["height"].tolist() == [250.0, 750.0, 1250.0, 1750.0]


def test_error_is_the_extra_spread_combined_over_levels():
    main = _column(spread=1.0, seed=1)
    err = _column(spread=np.sqrt(1.0 + 4.0), seed=2)  # +4 variance per level

    independent = transport_error(main, err, FLUX, length_scale=None)
    fully = transport_error(main, err, FLUX, length_scale=1e12)
    default = transport_error(main, err, FLUX)

    dvar = independent.levels["dvar"].to_numpy()
    assert dvar == pytest.approx(4.0, rel=0.3)
    # independent levels: sum w² dvar ~ 4/4; fully correlated: sum w dvar ~ 4
    assert independent.variance == pytest.approx(np.sum(0.25**2 * dvar))
    assert fully.variance == pytest.approx(np.sum(0.25 * dvar), rel=1e-6)
    assert independent.variance < default.variance < fully.variance
    assert default.length_scale == 356.0
    # a real signal (4 per level) stands well above the estimator's noise
    assert default.variance > 5 * default.noise


def test_the_column_variance_is_the_sum_over_levels_of_factor_times_dvar():
    """A known answer: the pair sum over the particles, with one level's spread falling."""
    sizes, extra = (40, 120, 60, 180), (0.6, -0.3, 0.2, 0.5)
    edges = [0.0, 300.0, 1200.0, 1500.0, 3000.0]
    rng = np.random.default_rng(0)
    heights, sign, d = [], [], []
    for lo, hi, size, d_level in zip(edges[:-1], edges[1:], sizes, extra, strict=True):
        heights.append(rng.uniform(lo + 1.0, hi, size))
        sign.append(np.tile([1.0, -1.0], size // 2))  # variance exactly 1
        d.append(np.full(size, d_level))
    heights, sign, d = (np.concatenate(a) for a in (heights, sign, d))
    main = _particles(heights, 10.0 + sign)
    err = _particles(heights, 10.0 + sign * np.sqrt(1.0 + d))

    result = transport_error(main, err, FLUX, levels=edges)

    levels = result.levels
    assert levels["dvar"].to_numpy() == pytest.approx(extra)
    assert result.variance == pytest.approx((levels["factor"] * levels["dvar"]).sum())
    # Each pair of particles counts with the mean of their two extra variances.
    corr = _pair_correlation(heights, 356.0)
    pair_mean = (d[:, None] + d[None, :]) / 2.0
    expected = (pair_mean * corr).sum() / len(d) ** 2
    assert result.variance == pytest.approx(expected, rel=1e-10)


def test_the_number_of_levels_does_not_change_the_column_variance():
    n, extra = 400, 0.5
    heights = np.linspace(0.0, 2000.0, n)
    sign = np.tile([1.0, -1.0], n // 2)  # variance exactly 1 in any even bin
    main = _particles(heights, 10.0 + sign)
    err = _particles(heights, 10.0 + sign * np.sqrt(1.0 + extra))

    two = transport_error(main, err, FLUX, levels=2)
    four = transport_error(main, err, FLUX, levels=4)

    expected = extra * _pair_correlation(heights, 356.0).sum() / n**2
    assert len(two.levels) == 2
    assert len(four.levels) == 4
    assert two.variance == pytest.approx(expected, rel=1e-10)
    assert four.variance == pytest.approx(expected, rel=1e-10)


def test_a_short_length_scale_on_a_tall_column_does_not_overflow():
    heights = np.linspace(0.0, 20000.0, 200)
    sign = np.tile([1.0, -1.0], 100)
    main = _particles(heights, 10.0 + sign)
    err = _particles(heights, 10.0 + sign * np.sqrt(1.5))

    result = transport_error(main, err, FLUX, levels=4, length_scale=0.5)

    expected = 0.5 * _pair_correlation(heights, 0.5).sum() / 200**2
    assert result.variance == pytest.approx(expected, rel=1e-10)


def test_a_small_error_in_a_column_is_not_underestimated():
    """Mean over receptors at a signal a few times the noise (#383)."""
    n, extra, receptors = 1000, 0.2, 100
    rng = np.random.default_rng(383)
    heights = rng.uniform(0.0, 3000.0, n)
    truth = extra * _pair_correlation(heights, 356.0).sum() / n**2

    variances = []
    for _ in range(receptors):
        main = _particles(heights, 10.0 + rng.standard_normal(n))
        err = _particles(heights, 10.0 + np.sqrt(1.0 + extra) * rng.standard_normal(n))
        variances.append(transport_error(main, err, FLUX).variance)

    # The mean's standard error is about 4 % of the truth. With the levels
    # combined through square roots of dvar it was 40 % low.
    assert np.mean(variances) == pytest.approx(truth, rel=0.15)


@pytest.mark.parametrize("column", [False, True], ids=["point", "column"])
def test_noise_is_the_spread_of_the_variance_with_no_perturbation(column):
    n, receptors = 1000, 150
    rng = np.random.default_rng(7)
    heights = rng.uniform(0.0, 3000.0, n) if column else np.zeros(n)

    variances, noises = [], []
    for _ in range(receptors):
        main = _particles(heights, 10.0 + rng.standard_normal(n))
        err = _particles(heights, 10.0 + rng.standard_normal(n))
        result = transport_error(main, err, FLUX)
        variances.append(result.variance)
        noises.append(result.noise)

    assert np.mean(noises) / np.std(variances, ddof=1) == pytest.approx(1.0, abs=0.15)


def test_percentile_clips_the_top_of_each_level():
    main = _column(spread=1.0, seed=3)
    err = _column(spread=1.0, seed=4)
    err.loc[err.index[0], "foot"] = 1e4  # one absurd particle (0.5%) in level 0

    clipped = transport_error(main, err, FLUX, percentile=0.99)
    raw = transport_error(main, err, FLUX)  # default keeps every particle

    assert abs(clipped.levels.loc[0, "dvar"]) < 2.0
    assert raw.levels.loc[0, "dvar"] > 1e4
    # the mean is never clipped, so the modelled enhancement is the same
    assert clipped.enhancement == raw.enhancement


def test_level_statistics_are_numpy_means_and_variances():
    main = _column(spread=1.0, seed=8)
    errs = [_column(spread=2.0, seed=9), _column(spread=3.0, seed=10)]
    errs[0] = errs[0].drop(index=errs[0].index[:30])  # level 0 loses particles

    result = transport_error(main, errs, FLUX, percentile=0.9)

    def stats(p, level):
        v = p.loc[p["release_height"] == 250.0 + 500.0 * level, "foot"].to_numpy()
        return v.mean(), v[v <= np.quantile(v, 0.9)].var()

    for level, row in result.levels.iterrows():
        mean_o, var_o = stats(main, level)
        means_e, vars_e = zip(*(stats(e, level) for e in errs), strict=True)
        assert row["mean_orig"] == mean_o
        assert row["var_orig"] == var_o
        assert row["mean_err"] == np.mean(means_e)
        assert row["var_err"] == np.mean(vars_e)


def test_point_receptor_particles_are_one_level():
    main = _column(n_levels=1, spread=1.0, seed=5).drop(columns="release_height")
    err = _column(n_levels=1, spread=2.0, seed=6).drop(columns="release_height")

    result = transport_error(main, err, FLUX)

    assert len(result.levels) == 1
    assert result.levels.loc[0, "height"] == 0.0
    assert result.levels.loc[0, "factor"] == pytest.approx(1.0)
    assert result.variance == pytest.approx(result.levels.loc[0, "dvar"])
    assert result.noise == pytest.approx(result.levels.loc[0, "noise"])
    assert result.sd == pytest.approx(np.sqrt(4.0 - 1.0), rel=0.2)


def test_continuous_release_heights_are_binned():
    main = _column(n_levels=4, per_level=100, seed=7)
    main["release_height"] = np.linspace(
        0.0, 2000.0, len(main)
    )  # every particle distinct
    err = main.copy()

    result = transport_error(main, err, FLUX, levels=5)

    assert len(result.levels) == 5
    assert result.levels["n"].sum() == len(main)
    assert result.levels["height"].is_monotonic_increasing

    explicit = transport_error(main, err, FLUX, levels=[0.0, 1000.0, 2000.0])
    assert explicit.levels["n"].tolist() == [200, 200]


def test_transforms_are_applied_to_both_tables():
    main = _column(spread=1.0, seed=8)
    err = _column(spread=np.sqrt(5.0), seed=9)
    kernel = AveragingKernel(levels=[0.0, 2000.0], values=[1.0, 0.0])  # kills the top

    plain = transport_error(main, err, FLUX, length_scale=None)
    weighted = transport_error(main, err, FLUX, transforms=[kernel], length_scale=None)

    top = weighted.levels.index[-1]
    assert weighted.levels.loc[top, "dvar"] < 0.1 * plain.levels.loc[top, "dvar"]
    assert weighted.enhancement < plain.enhancement
    assert weighted.sd < plain.sd


def test_the_receptor_reaches_the_transforms(point_receptor):
    seen = []

    class Recorder:
        def apply(self, particles, receptor=None, directory=None):
            seen.append(receptor)
            return particles

    p = _column(n_levels=1, per_level=10)
    transport_error(
        p,
        p.copy(),
        FLUX,
        transforms=[Recorder()],
        receptor=point_receptor,
    )

    assert seen == [point_receptor, point_receptor]


def test_rejects_bad_arguments():
    p = _column(n_levels=1, per_level=10)
    with pytest.raises(ValueError, match="percentile"):
        transport_error(p, p, FLUX, percentile=0.0)
    with pytest.raises(ValueError, match="length_scale"):
        transport_error(p, p, FLUX, length_scale=0.0)
    with pytest.raises(ValueError, match="levels"):
        transport_error(p, p, FLUX, levels=0)


# -- signed estimator and noise -------------------------------------


def test_less_spread_gives_a_negative_variance_and_zero_sd():
    main = _column(n_levels=1, spread=2.0, seed=10).drop(columns="release_height")
    err = _column(n_levels=1, spread=1.0, seed=11).drop(columns="release_height")

    result = transport_error(main, err, FLUX)

    assert result.variance == pytest.approx(1.0 - 4.0, rel=0.2)
    assert result.sd == 0.0
    assert result.levels.loc[0, "dvar"] < 0  # kept with its sign


def test_independent_unperturbed_runs_sit_within_the_noise():
    """Two runs that differ only by seed: |variance| should be O(noise), not O(signal)."""
    ratios = []
    for seed in range(12):
        a = _column(spread=1.0, seed=100 + seed)
        b = _column(spread=1.0, seed=200 + seed)
        r = transport_error(a, b, FLUX)
        assert r.noise > 0
        ratios.append(abs(r.variance) / r.noise)
    # |variance| is O(noise): most draws within 3 sigma, none absurdly outside
    assert np.median(ratios) < 2.0
    assert np.mean(np.asarray(ratios) < 3.0) >= 0.75
    assert max(ratios) < 8.0


def test_noise_is_the_standard_error_from_both_runs():
    main = _column(n_levels=1, spread=1.0, seed=12).drop(columns="release_height")
    err = _column(n_levels=1, spread=2.0, seed=13).drop(columns="release_height")

    result = transport_error(main, err, FLUX)

    expected = np.sqrt(_var_of_var(main["foot"]) + _var_of_var(err["foot"]))
    assert result.noise == pytest.approx(expected)
    # the perturbed run spreads more, so it carries most of the noise
    assert result.noise > np.sqrt(2.0 * _var_of_var(main["foot"]))


def test_background_field_adds_the_endpoint_spread_to_the_error():
    from stilt.particles import background

    field = xr.DataArray(
        np.tile([0.0, 1.0, 2.0], (3, 1)),
        dims=["lat", "lon"],
        coords={"lat": [40.0, 41.0, 42.0], "lon": [-112.0, -111.0, -110.0]},
    )
    main = _column(spread=1.0, seed=7)  # every endpoint at lon -111: background 1
    err = _column(spread=1.0, seed=8)
    err["lon"] = np.where(np.arange(len(err)) % 2 == 0, -112.0, -110.0)  # 0 or 2

    still = transport_error(main, main.copy(), FLUX, background=field)
    moved = transport_error(main, err, FLUX, background=field, length_scale=None)

    assert still.background == pytest.approx(1.0)
    assert still.background == pytest.approx(background(main, field).value)
    assert still.enhancement == pytest.approx(main["foot"].mean() + 1.0)
    assert still.variance == pytest.approx(0.0)
    # the perturbed endpoints alternate between 0 and 2: +1 variance per level,
    # combined uncorrelated over four equal levels -> sum(w^2) = 0.25
    assert moved.variance == pytest.approx(0.25, rel=0.4)
    assert transport_error(main, err, FLUX).background == 0.0


# -- realizations ------------------------------------------------------------------


def test_one_realization_in_a_list_matches_the_bare_table():
    main, err = _column(seed=1), _column(spread=np.sqrt(5.0), seed=2)
    single = transport_error(main, err, FLUX)
    listed = transport_error(main, [err], FLUX)

    assert listed.variance == single.variance
    assert listed.noise == single.noise
    assert listed.realizations == single.realizations == 1


def test_realizations_keep_the_variance_and_tighten_the_noise():
    main, err = _column(seed=1), _column(spread=np.sqrt(5.0), seed=2)
    single = transport_error(main, err, FLUX, length_scale=None)
    triple = transport_error(
        main, [err, err.copy(), err.copy()], FLUX, length_scale=None
    )

    assert triple.realizations == 3
    assert triple.variance == pytest.approx(single.variance)
    assert triple.enhancement_perturbed == pytest.approx(single.enhancement_perturbed)

    # Only the perturbed side averages down: the unperturbed particles are
    # the same in every realization.
    def noise(realizations):
        total = 0.0
        for height in main["release_height"].unique():
            of_main = _var_of_var(main.loc[main["release_height"] == height, "foot"])
            of_err = _var_of_var(err.loc[err["release_height"] == height, "foot"])
            total += 0.25**4 * (of_main + of_err / realizations)
        return np.sqrt(total)

    assert single.noise == pytest.approx(noise(1))
    assert triple.noise == pytest.approx(noise(3))


def test_averaging_realizations_pulls_the_estimate_toward_the_truth():
    """Independent draws of the same extra spread average toward its true value."""
    n_levels, per_level, extra = 4, 400, 4.0
    main = _column(n_levels, per_level, spread=1.0, seed=100)
    errs = [
        _column(n_levels, per_level, spread=np.sqrt(1.0 + extra), seed=200 + k)
        for k in range(24)
    ]
    # uncorrelated levels of equal weight: sum_i w_i^2 dvar_i = extra / n_levels
    truth = extra / n_levels

    singles = np.array(
        [transport_error(main, e, FLUX, length_scale=None).variance for e in errs]
    )
    pooled = transport_error(main, errs, FLUX, length_scale=None)

    assert pooled.realizations == 24
    assert abs(pooled.variance - truth) < np.median(np.abs(singles - truth))
    assert pooled.variance == pytest.approx(truth, rel=0.15)


def test_realizations_must_not_be_empty():
    p = _column()
    with pytest.raises(ValueError, match="at least one realization"):
        transport_error(p, [], FLUX)
