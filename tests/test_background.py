"""Tests for stilt.background and sim.background."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from runtraj.transforms import FirstOrderLifetime, Oxidation, PressureWeighting, weigh

from stilt import Background, background


def _field(values=None, lons=(-112.0, -111.0, -110.0), lats=(40.0, 41.0)):
    values = (
        np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
        if values is None
        else values
    )
    return xr.DataArray(
        values, dims=["lat", "lon"], coords={"lat": list(lats), "lon": list(lons)}
    )


def _field_3d():
    """Levels 1000, 800, 600 hPa; value = 100 * level index + cell index."""
    base = np.arange(6, dtype=float).reshape(2, 3)
    return xr.DataArray(
        np.stack([base, base + 100.0, base + 200.0]),
        dims=["pres", "lat", "lon"],
        coords={
            "pres": [1000.0, 800.0, 600.0],
            "lat": [40.0, 41.0],
            "lon": [-112.0, -111.0, -110.0],
        },
    )


def _particles():
    """Three backward particles; endpoints at 1000, 650 and 820 hPa."""
    return pd.DataFrame(
        {
            "particle": [1, 1, 2, 2, 3, 3],
            "age": [0.0, -60.0, 0.0, -60.0, -30.0, -120.0],
            "lon": [-111.0, -112.0, -111.0, -110.0, -111.0, -111.0],
            "lat": [40.0, 40.0, 40.0, 41.0, 40.0, 41.0],
            "pres": [900.0, 1000.0, 900.0, 650.0, 900.0, 820.0],
            "foot": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            "time": pd.to_datetime(["2023-01-01 12:00", "2023-01-01 11:00"] * 3),
        }
    )


# endpoint values: particle 1 -> level 0 cell 0 = 0; 2 -> level 600 cell 5 = 205;
# 3 -> level 800 cell 4 = 104
ENDPOINT_VALUES = [0.0, 205.0, 104.0]


# -- endpoints ---------------------------------------------------------------------


def test_endpoints_pick_the_largest_abs_time_per_particle():
    ends = _particles().stilt.endpoints()

    assert ends["particle"].tolist() == [1, 2, 3]
    assert ends["age"].tolist() == [-60.0, -60.0, -120.0]


def test_endpoints_handle_forward_runs_and_duplicate_index():
    p = _particles()
    p["age"] = -p["age"]
    p.index = [0, 0, 1, 1, 2, 2]

    ends = p.stilt.endpoints()

    assert ends["age"].tolist() == [60.0, 60.0, 120.0]
    assert p.iloc[0:0].stilt.endpoints().empty


# -- sampling the field ---------------------------------------------------------------


def test_background_samples_each_endpoint():
    per = background(_particles(), _field_3d()).per_particle

    assert per.index.name == "particle"
    assert per.index.tolist() == [1, 2, 3]
    assert per.tolist() == ENDPOINT_VALUES
    assert per.name == "background"


def test_background_time_varying_uses_the_endpoint_time():
    times = pd.to_datetime(["2023-01-01 11:00", "2023-01-01 12:00"])
    field = xr.concat([_field(), _field() + 1000.0], dim="time").assign_coords(
        time=times
    )

    per = background(_particles(), field).per_particle  # endpoints are at 11:00

    assert per.tolist() == [0.0, 5.0, 4.0]
    with pytest.raises(ValueError, match="time"):
        background(_particles().drop(columns="time"), field)


def test_background_vertical_dimension_must_name_a_particle_column():
    with pytest.raises(ValueError, match="'level' is not a particle column"):
        background(_particles(), _field_3d().rename(pres="level"))


# -- background ----------------------------------------------------------------------


def test_background_without_transforms_is_the_mean_over_particles():
    result = background(_particles(), _field_3d())

    assert isinstance(result, Background)
    assert result.value == pytest.approx(np.mean(ENDPOINT_VALUES))
    assert result.weights.tolist() == pytest.approx([1 / 3] * 3)
    assert result.per_particle.tolist() == ENDPOINT_VALUES


def test_background_reads_each_particles_weight_at_its_endpoint():
    class DropThird:
        """A transform that writes only the weight: particle 3 counts for nothing."""

        def apply(self, particles, release=None, directory=None):
            return weigh(particles, (particles["particle"] != 3).astype(float))

    result = background(_particles(), _field_3d(), transforms=[DropThird()])

    assert result.weights.tolist() == pytest.approx([1 / 3, 1 / 3, 0.0])
    assert result.value == pytest.approx(sum(ENDPOINT_VALUES[:2]) / 3)


def test_background_accepts_values_sampled_elsewhere():
    sampled = pd.Series(ENDPOINT_VALUES, index=pd.Index([1, 2, 3], name="particle"))
    assert background(_particles(), sampled).value == pytest.approx(
        np.mean(ENDPOINT_VALUES)
    )


def test_background_lifetime_decays_at_the_endpoint_age():
    result = background(
        _particles(), _field_3d(), transforms=[FirstOrderLifetime(lifetime_hours=1.0)]
    )
    w = np.exp(-np.array([1.0, 1.0, 2.0])) / 3.0
    assert result.weights.to_numpy() == pytest.approx(w)
    assert result.value == pytest.approx(float(np.sum(w * ENDPOINT_VALUES)))


def test_background_oxidation_scales_the_inflow_by_the_survival_over_the_whole_path():
    p = _particles().assign(temz=280.0)
    oxidation = Oxidation(a=1e-4, oxidant=1.0)  # 1e-4 s⁻¹ everywhere
    result = background(p, _field_3d(), transforms=[oxidation])
    w = np.exp(-1e-4 * 60.0 * np.array([60.0, 60.0, 120.0])) / 3.0
    assert result.weights.to_numpy() == pytest.approx(w)
    assert result.value == pytest.approx(float(np.sum(w * ENDPOINT_VALUES)))


def test_background_leaves_out_particles_outside_the_field():
    p = _particles()
    p.loc[p["particle"] == 2, "lon"] = -100.0  # particle 2 ends off the field

    result = background(p, _field_3d())

    assert np.isnan(result.per_particle.loc[2])
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.value == pytest.approx((0.0 + 104.0) / 2)
    p["lon"] = -100.0
    assert np.isnan(background(p, _field_3d()).value)


def _column(n=60, p_sfc=1000.0):
    """One-row column particles released 0-3000 m on a hypsometric profile."""
    z = np.linspace(0.0, 3000.0, n)
    return pd.DataFrame(
        {
            "particle": np.arange(1, n + 1),
            "age": np.full(n, -60.0),
            "lon": np.full(n, -111.0),
            "lat": np.full(n, 40.0),
            "zagl": z,
            "release_height": z,
            "pres": p_sfc * np.exp(-z / 8000.0),
            "foot": np.ones(n),
        }
    )


def test_background_fills_missing_particles_with_the_weighted_mean_of_the_others():
    sampled = pd.Series([1.0, np.nan, 3.0], index=pd.Index([1, 2, 3], name="particle"))
    w = np.exp(-np.array([1.0, 1.0, 2.0])) / 3.0

    result = background(
        _particles(), sampled, transforms=[FirstOrderLifetime(lifetime_hours=1.0)]
    )

    mean = (w[0] * 1.0 + w[2] * 3.0) / (w[0] + w[2])
    assert result.value == pytest.approx(mean * w.sum())


def test_background_pressure_weighting_covers_the_column_mass_fraction():
    p = _column()
    field = _field(values=np.full((2, 3), 1900.0))

    result = background(p, field, transforms=[PressureWeighting()])

    covered = 1.0 - p["pres"].min() / p["pres"].max()
    assert result.weights.sum() == pytest.approx(covered, abs=0.02)
    assert result.value == pytest.approx(1900.0 * result.weights.sum())
