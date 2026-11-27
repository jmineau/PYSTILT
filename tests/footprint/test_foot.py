"""Tests for stilt.footprint.foot: foot for a model that writes none, and the near-field correction."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from runtraj.releases import ColumnRelease, PointRelease

from stilt.footprint import foot as foot_module
from stilt.footprint.foot import add_foot, correct_near_field

from ..fixtures.particles import point_at

#: A release that gives no release height of its own: a column.
_NO_POINT = ColumnRelease(
    time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, bottom=0, top=1000
)


def _particles_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [-60, -120],
            "particle": [1, 1],
            "lon": [-111.9, -112.0],
            "lat": [40.7, 40.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def test_correct_near_field_requires_columns():
    p = pd.DataFrame({"age": [-60], "particle": [1], "foot": [1e-5]})
    with pytest.raises(ValueError, match="needs the particle columns"):
        correct_near_field(p, point_at(5.0), 0.5)


def test_correct_near_field_adds_reference_column():
    out = correct_near_field(
        _particles_basic().drop(columns=["zagl"]).assign(release_height=[5.0, 5.0]),
        _NO_POINT,
        0.5,
    )
    assert "foot_no_hnf_dilution" in out.columns
    assert out["foot_no_hnf_dilution"].iloc[0] == pytest.approx(1e-5)


def test_correct_near_field_reads_footprint_depth_above_one_as_meters():
    """A footprint_depth above 1 is meters above ground, as HYSPLIT reads it (#167)."""
    p = _particles_basic().drop(columns=["zagl"]).assign(release_height=[5.0, 5.0])
    # The plume is about 32 m deep after one hour and 70 m after two.
    as_fraction = correct_near_field(p, _NO_POINT, 0.5)  # below 250 m
    as_meters = correct_near_field(p, _NO_POINT, 50.0)  # below 50 m

    assert as_meters["foot"].iloc[0] == pytest.approx(as_fraction["foot"].iloc[0])
    assert as_fraction["foot"].iloc[1] != pytest.approx(p["foot"].iloc[1])
    assert as_meters["foot"].iloc[1] == pytest.approx(p["foot"].iloc[1])


def test_correct_near_field_grows_the_plume_by_taylors_spread():
    """One minute out the plume has grown by Taylor's spread, never more than sigw t (#507)."""
    p = pd.DataFrame(
        {
            "age": [-1.0],
            "particle": [1],
            "foot": [1e-3],
            "dens": [1.2],
            "samt": [1.0],
            "sigw": [0.5],
            "tlgr": [100.0],
            "mlht": [1000.0],
            "release_height": [5.0],
        }
    )
    out = correct_near_field(p, _NO_POINT, 0.5)
    plume = 0.02897 / (out["foot"].iloc[0] * 1.2) * 60
    t, tl, sw = 60.0, 100.0, 0.5
    taylor = np.sqrt(2) * sw * np.sqrt(tl * t + tl**2 * np.exp(-t / tl) - tl**2)
    assert plume - 5.0 == pytest.approx(taylor)
    assert plume - 5.0 <= sw * t


def test_correct_near_field_grows_outward_from_release_when_forward():
    """A forward run accumulates sigma from the release point, not the far end."""
    backward = (
        _particles_basic().drop(columns=["zagl"]).assign(release_height=[5.0, 5.0])
    )
    forward = backward.assign(age=-backward["age"])

    back_out = correct_near_field(backward, _NO_POINT, 0.5)
    fwd_out = correct_near_field(forward, _NO_POINT, 0.5)

    # sigma depends on |time| only, so mirroring the track must not change foot
    assert fwd_out["foot"].to_numpy() == pytest.approx(back_out["foot"].to_numpy())
    # foot scales as 1 / plume, so it shrinks as the plume grows away from release
    assert fwd_out["foot"].iloc[0] > fwd_out["foot"].iloc[1]


def test_correct_near_field_starts_an_msl_release_at_the_ground():
    """An MSL release's plume starts at its height above ground, not above sea level (#287)."""
    p = _particles_basic().drop(columns=["zagl"]).assign(zsfc=1300.0)
    agl = PointRelease(
        time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    msl = PointRelease(
        time="2023-01-01 12:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=1305.0,
        altitude_ref="msl",
    )

    from_agl = correct_near_field(p, agl, 0.5)
    from_msl = correct_near_field(p, msl, 0.5)

    assert from_agl["foot"].iloc[0] != pytest.approx(p["foot"].iloc[0])
    assert from_msl["foot"].to_numpy() == pytest.approx(from_agl["foot"].to_numpy())


def test_correct_near_field_puts_an_msl_release_below_the_terrain_at_the_ground():
    p = _particles_basic().drop(columns=["zagl"]).assign(zsfc=1300.0)
    below = PointRelease(
        time="2023-01-01 12:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=1290.0,
        altitude_ref="msl",
    )
    foot = correct_near_field(p, below, 0.5)["foot"]
    assert np.isfinite(foot).all() and (foot > 0).all()


def test_correct_near_field_needs_zsfc_for_an_msl_release():
    p = _particles_basic().drop(columns=["zagl"]).assign(release_height=1305.0)
    msl = ColumnRelease(
        time="2023-01-01 12:00",
        longitude=-111.85,
        latitude=40.77,
        bottom=1300,
        top=2300,
        altitude_ref="msl",
    )
    with pytest.raises(ValueError, match="zsfc"):
        correct_near_field(p, msl, 0.5)


def test_correct_near_field_raises_without_a_release_height():
    """A release other than a point, and no release_height column, raises ValueError."""
    p = _particles_basic()

    with pytest.raises(ValueError, match="release height"):
        correct_near_field(p, _NO_POINT, 0.5)


def test_hnf_correction_invariants():
    """
    Mathematical invariants of the HNF near-field plume-dilution correction.

    The HNF correction replaces the HYSPLIT-raw foot with a Gaussian near-field
    value (0.02897 / (plume * dens) * samt * 60) when the plume has not yet
    grown to fill the mixing layer.  This can be larger or smaller than the raw
    value.  The invariants that must always hold:

    1. Corrected foot is always positive.
    2. When plume >= pbl_mixing, foot is left unchanged (identity path).
    3. The raw values are preserved in `foot_no_hnf_dilution`.
    """
    rng = np.random.default_rng(77)
    n = 50
    raw_foot = rng.uniform(1e-5, 1e-3, n)
    # Force some particles to have large plume (sigma >> mlht) so the identity
    # path is exercised.  Large sigw + long time → large plume.
    sigw = np.concatenate(
        [
            rng.uniform(0.01, 0.5, n // 2),  # small sigma → near-field path
            rng.uniform(5.0, 20.0, n // 2),  # large sigma → identity path
        ]
    )
    particles = pd.DataFrame(
        {
            "age": np.concatenate(
                [
                    rng.uniform(-0.5, -0.1, n // 2),  # short time → small plume
                    rng.uniform(-6.0, -5.0, n // 2),  # long time → large plume
                ]
            ),
            "particle": [float(i + 1) for i in range(n)],
            "lon": [-112.0] * n,
            "lat": [40.5] * n,
            "zagl": [5.0] * n,
            "foot": raw_foot,
            "mlht": [500.0] * n,  # pbl_mixing = 0.5 * 500 = 250 m
            "dens": [1.2] * n,
            "samt": [60.0] * n,
            "sigw": sigw,
            "tlgr": [100.0] * n,
        }
    )

    result = correct_near_field(particles.copy(), point_at(5.0), 0.5)

    # Invariant 1: corrected foot is always positive.
    assert np.all(result["foot"].values > 0), (
        "HNF-corrected foot values must be positive"
    )

    # Invariant 2: identity path — particles where plume >= pbl_mixing
    # are unchanged.  Reconstruct plume = r_zagl + sigma to find them.
    abs_time_s = np.abs(particles["age"] * 60)
    tlgr = particles["tlgr"]
    sigma = (
        particles["samt"]
        * np.sqrt(2)
        * particles["sigw"]
        * np.sqrt(tlgr * abs_time_s + tlgr**2 * np.exp(-abs_time_s / tlgr) - tlgr**2)
    )
    plume = 5.0 + sigma  # r_zagl=5 + sigma (single timestep, cumsum is sigma itself)
    pbl_mixing = 0.5 * particles["mlht"]
    identity_mask = (plume >= pbl_mixing).to_numpy()

    np.testing.assert_allclose(
        result.loc[identity_mask, "foot"].values,
        raw_foot[identity_mask],
        rtol=1e-12,
        err_msg="Particles with plume >= pbl_mixing must have foot unchanged",
    )

    # Invariant 3: raw values preserved in foot_no_hnf_dilution.
    np.testing.assert_array_equal(
        result["foot_no_hnf_dilution"].values,
        raw_foot,
        err_msg="foot_no_hnf_dilution must equal the original foot values",
    )


_ONE_MINUTE_IN_500_M = 0.02897 * 60 / 500


def _positions(**columns) -> pd.DataFrame:
    """A table of positions, two particles of two rows each, with *columns* added."""
    base = pd.DataFrame(
        {
            "particle": [1, 1, 2, 2],
            "age": [-1, -2, -1, -2],
            "lon": -112.0,
            "lat": 40.5,
            "zagl": [50.0, 80.0, 50.0, 900.0],
            "time": pd.Timestamp("2023-01-01 12:00"),
            "mlht": 1000.0,
        }
    )
    return base.assign(**columns)


@pytest.fixture
def met(monkeypatch):
    """
    Stand in for the met files: air of density ``met["density"]`` kg/m3 and a 1000 m PBLH.

    The pressure falls ρ g / 100 hPa per meter, so the air below a depth is
    ρ times the depth. A row at longitude 0 is off the grid.
    """
    met = {"density": 1.0, "calls": []}

    def sample_met(particles, files, columns, *, z_top):
        met["calls"].append((list(columns), z_top, files))
        off = particles["lon"] == 0
        if columns == ["mlht"]:
            return particles.assign(mlht=np.where(off, np.nan, 1000.0))
        pres = 1000.0 - met["density"] * 9.80665 / 100 * particles["zagl"]
        return particles.assign(pres=np.where(off, np.nan, pres))

    monkeypatch.setattr(foot_module, "sample_met", sample_met)
    return met


def test_add_foot_hands_a_dataset_to_the_met_as_it_is(met):
    dataset = xr.Dataset({"PRES": ("level", [1000.0])})
    add_foot(_positions(), dataset, footprint_depth=0.5)

    assert met["calls"]
    assert all(call[2] is dataset for call in met["calls"])


def test_add_foot_counts_a_whole_step_below_the_mixing_depth_and_none_above(met):
    foot = add_foot(_positions(), ["met"], footprint_depth=0.5)["foot"]

    np.testing.assert_allclose(foot, [_ONE_MINUTE_IN_500_M] * 3 + [0.0])
    # The pressure at the ground and at the depth, on the met's own levels.
    assert met["calls"] == [(["pres"], None, ["met"])]


def test_add_foot_reads_footprint_depth_above_1_as_meters(met):
    foot = add_foot(_positions(), ["met"], footprint_depth=800.0)["foot"]

    np.testing.assert_allclose(foot, [0.02897 * 60 / 800] * 3 + [0.0])


def test_add_foot_divides_by_the_air_below_the_depth(met):
    met["density"] = 2.0
    foot = add_foot(
        _positions(mlht=[1000.0, 2000.0, 500.0, 1e4]), ["met"], footprint_depth=0.5
    )["foot"]

    np.testing.assert_allclose(
        foot,
        np.array([1.0, 1 / 2, 2.0, 1 / 10]) * _ONE_MINUTE_IN_500_M / 2,
    )


def test_add_foot_samples_the_mets_mlht_when_the_table_has_none(met):
    out = add_foot(_positions().drop(columns="mlht"), ["met"], footprint_depth=0.5)

    assert out["mlht"].tolist() == [1000.0] * 4
    np.testing.assert_allclose(out["foot"], [_ONE_MINUTE_IN_500_M] * 3 + [0.0])
    assert [call[0] for call in met["calls"]] == [["mlht"], ["pres"]]


def test_add_foot_takes_each_step_from_the_particle_s_own_rows(met):
    """Steps follow each particle in time, whatever the row order, and a release row gets none."""
    particles = pd.DataFrame(
        {
            "particle": [2, 1, 1, 2, 1, 2],
            "age": [10, 5, 0, 0, 15, 30],
            "lon": -112.0,
            "lat": 40.5,
            "zagl": 10.0,
            "time": pd.Timestamp("2023-01-01 12:00"),
            "mlht": 1000.0,
        }
    )

    foot = (
        add_foot(particles, ["met"], footprint_depth=0.5)["foot"] / _ONE_MINUTE_IN_500_M
    )

    np.testing.assert_allclose(foot, [10.0, 5.0, 0.0, 0.0, 10.0, 20.0])


def test_add_foot_uses_the_time_below_the_layer_a_model_writes(met):
    foot = add_foot(
        _positions(samt=[0.5, 1.0, 0.0, 0.25]), ["met"], footprint_depth=0.5
    )["foot"]

    np.testing.assert_allclose(
        foot, np.array([0.5, 1.0, 0.0, 0.25]) * _ONE_MINUTE_IN_500_M
    )


def test_add_foot_gives_nan_off_the_mets_grid(met):
    foot = add_foot(_positions(lon=[0.0, -112.0, -112.0, -112.0]), ["met"], 0.5)["foot"]

    assert np.isnan(foot[0])
    assert np.isfinite(foot[1:]).all()


def test_add_foot_refuses_a_table_that_has_foot(met):
    with pytest.raises(ValueError, match="foot column already"):
        add_foot(_positions(foot=1.0), ["met"], footprint_depth=0.5)


def test_add_foot_with_a_fixed_pbl_height_replaces_mlht_and_ignores_samt(met):
    particles = _positions(samt=[0.5, 1.0, 0.0, 0.25])
    out = add_foot(particles, ["met"], footprint_depth=1.0, pbl_height=500.0)

    assert out["mlht"].tolist() == [500.0] * 4
    # samt counts the minutes below the model's own layer, so it is not used.
    np.testing.assert_allclose(out["foot"], [_ONE_MINUTE_IN_500_M] * 3 + [0.0])
    assert [call[0] for call in met["calls"]] == [["pres"]]


def test_add_foot_with_pbl_height_met_samples_pblh_over_the_models_mlht(met):
    out = add_foot(
        _positions(mlht=300.0), ["met"], footprint_depth=0.5, pbl_height="met"
    )

    assert out["mlht"].tolist() == [1000.0] * 4
    assert [call[0] for call in met["calls"]] == [["mlht"], ["pres"]]


def test_add_foot_keeps_a_height_it_finds_at_least_pbl_height_min(met):
    out = add_foot(
        _positions().drop(columns="mlht"),
        ["met"],
        footprint_depth=0.5,
        pbl_height_min=1200.0,
    )

    assert out["mlht"].tolist() == [1200.0] * 4


def test_add_foot_applies_no_minimum_to_the_models_own_mlht(met):
    out = add_foot(
        _positions(mlht=300.0), ["met"], footprint_depth=0.5, pbl_height_min=1200.0
    )

    assert out["mlht"].tolist() == [300.0] * 4


def test_add_foot_with_pbl_height_richardson_finds_it_from_the_profile(
    met, monkeypatch
):

    seen = {}

    def calc_pbl_height(particles, files, **top):
        seen.update(columns=list(particles.columns), files=files, **top)
        return np.full(len(particles), 400.0)

    monkeypatch.setattr(foot_module, "calc_pbl_height", calc_pbl_height)
    out = add_foot(_positions(), ["met"], footprint_depth=0.5, pbl_height="richardson")

    assert "mlht" not in seen["columns"]
    assert seen["files"] == ["met"]
    assert "z_top" not in seen  # calc_pbl_height's default, HYSPLIT's 25 km
    assert out["mlht"].tolist() == [400.0] * 4

    add_foot(
        _positions(), ["met"], footprint_depth=0.5, pbl_height="richardson", z_top=1e4
    )
    assert seen["z_top"] == 1e4
