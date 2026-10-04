"""Tests for stilt.observations.transport_error."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.observations import TransportError, transport_error
from stilt.transforms import AveragingKernel

FLUX = xr.DataArray(
    np.ones((3, 3)),
    dims=["lat", "lon"],
    coords={"lat": [40.0, 41.0, 42.0], "lon": [-112.0, -111.0, -110.0]},
)


def _column(n_levels=4, per_level=200, spread=1.0, seed=0, level_spacing=500.0):
    """Column particles: each particle's foot sum ~ N(10, spread²); one row each."""
    rng = np.random.default_rng(seed)
    n = n_levels * per_level
    xhgt = np.repeat(np.arange(n_levels) * level_spacing + 250.0, per_level)
    return pd.DataFrame(
        {
            "indx": np.arange(1, n + 1),
            "time": np.zeros(n),
            "long": np.full(n, -111.0),
            "lati": np.full(n, 41.0),
            "xhgt": xhgt,
            "foot": 10.0 + spread * rng.standard_normal(n),
            "datetime": pd.to_datetime(["2023-01-01"] * n),
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
    assert result.noise > 0  # halves of the same ensemble still differ
    assert len(result.levels) == 4
    assert result.levels["n"].tolist() == [200] * 4
    assert result.levels["weight"].sum() == pytest.approx(1.0)
    assert result.levels["height"].tolist() == [250.0, 750.0, 1250.0, 1750.0]


def test_error_is_the_extra_spread_combined_over_levels():
    main = _column(spread=1.0, seed=1)
    err = _column(
        spread=np.sqrt(1.0 + 4.0), seed=2
    )  # +4 variance => sd_trans 2 per level

    uncorrelated = transport_error(main, err, FLUX, length_scale=None)
    fully = transport_error(main, err, FLUX, length_scale=1e12)
    default = transport_error(main, err, FLUX)

    sd_levels = uncorrelated.levels["sd_trans"].to_numpy()
    assert sd_levels == pytest.approx(2.0, rel=0.15)
    # uncorrelated: sqrt(sum w^2 sd^2) ~ 2/sqrt(4); fully correlated: sum w sd ~ 2
    assert uncorrelated.sd == pytest.approx(np.sqrt(np.sum(0.25**2 * sd_levels**2)))
    assert fully.sd == pytest.approx(np.sum(0.25 * sd_levels), rel=1e-6)
    assert uncorrelated.sd < default.sd < fully.sd
    assert default.length_scale == 356.0
    # a real signal (4 per level) stands well above the estimator's noise
    assert default.variance > 5 * default.noise


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

    result = transport_error(main, errs, FLUX, percentile=0.9, noise_splits=0)

    def stats(p, level):
        v = p.loc[p["xhgt"] == 250.0 + 500.0 * level, "foot"].to_numpy()
        return v.mean(), v[v <= np.quantile(v, 0.9)].var()

    for level, row in result.levels.iterrows():
        mean_o, var_o = stats(main, level)
        means_e, vars_e = zip(*(stats(e, level) for e in errs), strict=True)
        assert row["mean_orig"] == mean_o
        assert row["var_orig"] == var_o
        assert row["mean_err"] == np.mean(means_e)
        assert row["var_err"] == np.mean(vars_e)


def test_point_receptor_particles_are_one_level():
    main = _column(n_levels=1, spread=1.0, seed=5).drop(columns="xhgt")
    err = _column(n_levels=1, spread=2.0, seed=6).drop(columns="xhgt")

    result = transport_error(main, err, FLUX)

    assert len(result.levels) == 1
    assert result.levels.loc[0, "height"] == 0.0
    assert result.variance == pytest.approx(result.levels.loc[0, "dvar"])
    assert result.sd == pytest.approx(result.levels.loc[0, "sd_trans"])
    assert result.sd == pytest.approx(np.sqrt(4.0 - 1.0), rel=0.2)


def test_continuous_release_heights_are_binned():
    main = _column(n_levels=4, per_level=100, seed=7)
    main["xhgt"] = np.linspace(0.0, 2000.0, len(main))  # every particle distinct
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
    assert (
        weighted.levels.loc[top, "sd_trans"] < 0.3 * plain.levels.loc[top, "sd_trans"]
    )
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
    with pytest.raises(ValueError, match="noise_splits"):
        transport_error(p, p, FLUX, noise_splits=-1)


# -- signed estimator and noise floor ------------------------------


def test_less_spread_gives_a_negative_variance_and_zero_sd():
    main = _column(n_levels=1, spread=2.0, seed=10).drop(columns="xhgt")
    err = _column(n_levels=1, spread=1.0, seed=11).drop(columns="xhgt")

    result = transport_error(main, err, FLUX)

    assert result.variance == pytest.approx(1.0 - 4.0, rel=0.2)
    assert result.sd == 0.0
    assert result.levels.loc[0, "sd_trans"] < 0  # signed square root


def test_independent_unperturbed_runs_sit_within_the_noise_floor():
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


def test_noise_is_reproducible_and_optional():
    main = _column(spread=1.0, seed=12)
    err = _column(spread=2.0, seed=13)

    a = transport_error(main, err, FLUX)
    b = transport_error(main, err, FLUX)
    off = transport_error(main, err, FLUX, noise_splits=0)

    assert a.noise == b.noise
    assert np.isnan(off.noise)
    assert off.variance == a.variance


def test_background_field_adds_the_endpoint_spread_to_the_error():
    from stilt.observations import background

    field = xr.DataArray(
        np.tile([0.0, 1.0, 2.0], (3, 1)),
        dims=["lat", "lon"],
        coords={"lat": [40.0, 41.0, 42.0], "lon": [-112.0, -111.0, -110.0]},
    )
    main = _column(spread=1.0, seed=7)  # every endpoint at lon -111: background 1
    err = _column(spread=1.0, seed=8)
    err["long"] = np.where(np.arange(len(err)) % 2 == 0, -112.0, -110.0)  # 0 or 2

    still = transport_error(main, main.copy(), FLUX, background=field, noise_splits=0)
    moved = transport_error(
        main, err, FLUX, background=field, length_scale=None, noise_splits=0
    )

    assert still.background == pytest.approx(1.0)
    assert still.background == pytest.approx(background(main, field).value)
    assert still.enhancement == pytest.approx(main["foot"].mean() + 1.0)
    assert still.variance == pytest.approx(0.0)
    # the perturbed endpoints alternate between 0 and 2: +1 variance per level,
    # combined uncorrelated over four equal levels -> sum(w^2) = 0.25
    assert moved.variance == pytest.approx(0.25, rel=0.4)
    assert transport_error(main, err, FLUX, noise_splits=0).background == 0.0


# -- realizations ------------------------------------------------------------------


def test_one_realization_in_a_list_matches_the_bare_table():
    main, err = _column(seed=1), _column(spread=np.sqrt(5.0), seed=2)
    single = transport_error(main, err, FLUX)
    listed = transport_error(main, [err], FLUX)

    assert listed.variance == single.variance
    assert listed.noise == single.noise
    assert listed.realizations == single.realizations == 1


def test_identical_realizations_keep_the_variance_and_tighten_the_noise():
    main, err = _column(seed=1), _column(spread=np.sqrt(5.0), seed=2)
    single = transport_error(main, err, FLUX)
    triple = transport_error(main, [err, err.copy(), err.copy()], FLUX)

    assert triple.realizations == 3
    assert triple.variance == pytest.approx(single.variance)
    assert triple.enhancement_perturbed == pytest.approx(single.enhancement_perturbed)
    # the unperturbed side is shared, so the null spread only falls to sqrt((1+1/N)/2)
    assert triple.noise == pytest.approx(single.noise * np.sqrt((1 + 1 / 3) / 2))


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
