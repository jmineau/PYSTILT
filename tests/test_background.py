"""Tests for stilt.observations.backgrounds."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.observations import Background, background, particle_background
from stilt.observations.backgrounds import (
    endpoint_weights,
    fill_missing,
    sample_field,
    vertical_dim,
)
from stilt.trajectory import endpoint_rows
from stilt.transforms import FirstOrderLifetime, PressureWeighting


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
            "indx": [1, 1, 2, 2, 3, 3],
            "time": [0.0, -60.0, 0.0, -60.0, -30.0, -120.0],
            "long": [-111.0, -112.0, -111.0, -110.0, -111.0, -111.0],
            "lati": [40.0, 40.0, 40.0, 41.0, 40.0, 41.0],
            "pres": [900.0, 1000.0, 900.0, 650.0, 900.0, 820.0],
            "foot": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            "datetime": pd.to_datetime(["2023-01-01 12:00", "2023-01-01 11:00"] * 3),
        }
    )


# endpoint values: particle 1 -> level 0 cell 0 = 0; 2 -> level 600 cell 5 = 205;
# 3 -> level 800 cell 4 = 104
ENDPOINT_VALUES = [0.0, 205.0, 104.0]


# -- endpoint_rows -----------------------------------------------------------------


def test_endpoint_rows_picks_the_largest_abs_time_per_particle():
    ends = endpoint_rows(_particles())

    assert ends["indx"].tolist() == [1, 2, 3]
    assert ends["time"].tolist() == [-60.0, -60.0, -120.0]


def test_endpoint_rows_handles_forward_runs_and_duplicate_index():
    p = _particles()
    p["time"] = -p["time"]
    p.index = [0, 0, 1, 1, 2, 2]

    ends = endpoint_rows(p)

    assert ends["time"].tolist() == [60.0, 60.0, 120.0]
    assert endpoint_rows(p.iloc[0:0]).empty


# -- vertical_dim / sample_field -----------------------------------------------------


def test_vertical_dim_is_the_one_dimension_besides_horizontal_and_time():
    assert vertical_dim(_field()) is None
    assert vertical_dim(_field_3d()) == "pres"
    timed = _field_3d().expand_dims(time=pd.to_datetime(["2023-01-01"]))
    assert vertical_dim(timed) == "pres"
    with pytest.raises(ValueError, match="at most one vertical"):
        vertical_dim(_field_3d().expand_dims(ens=[0, 1]))


def test_sample_field_is_nearest_cell_and_nan_outside():
    sampled = sample_field(
        _field(),
        x=[-112.0, -111.4, -110.6, -109.4, -111.0],
        y=[40.0, 40.4, 40.6, 41.0, 41.6],
    )
    assert sampled[:3].tolist() == [0.0, 1.0, 4.0]
    assert np.isnan(sampled[3:]).all()


def test_sample_field_vertical_is_nearest_level_held_at_the_ends():
    sampled = sample_field(
        _field_3d(),
        x=[-112.0] * 4,
        y=[40.0] * 4,
        z=[1000.0, 650.0, 500.0, 1100.0],  # exact, nearest, below bottom, above top
    )
    assert sampled.tolist() == [0.0, 200.0, 200.0, 0.0]


def test_sample_field_wraps_longitudes_to_the_field_convention():
    east = _field(lons=(248.0, 249.0, 250.0))  # 0..360
    assert sample_field(east, x=[-111.0], y=[40.0]).tolist() == [1.0]
    assert sample_field(_field(), x=[249.0], y=[40.0]).tolist() == [1.0]


def test_sample_field_requires_z_and_times_when_the_field_has_them():
    with pytest.raises(ValueError, match="pass z"):
        sample_field(_field_3d(), x=[-112.0], y=[40.0])
    timed = _field().expand_dims(time=pd.to_datetime(["2023-01-01"]))
    with pytest.raises(ValueError, match="pass times"):
        sample_field(timed, x=[-112.0], y=[40.0])
    with pytest.raises(ValueError, match="same length"):
        sample_field(_field_3d(), x=[-112.0], y=[40.0], z=[1.0, 2.0])


# -- particle_background -------------------------------------------------------------


def test_particle_background_samples_each_endpoint():
    per = particle_background(_particles(), _field_3d())

    assert per.index.name == "indx"
    assert per.index.tolist() == [1, 2, 3]
    assert per.tolist() == ENDPOINT_VALUES
    assert per.name == "background"


def test_particle_background_time_varying_uses_the_endpoint_datetime():
    times = pd.to_datetime(["2023-01-01 11:00", "2023-01-01 12:00"])
    field = xr.concat([_field(), _field() + 1000.0], dim="time").assign_coords(
        time=times
    )

    per = particle_background(_particles(), field)  # endpoints are at 11:00

    assert per.tolist() == [0.0, 5.0, 4.0]
    with pytest.raises(ValueError, match="datetime"):
        particle_background(_particles().drop(columns="datetime"), field)


def test_particle_background_vertical_dimension_must_name_a_particle_column():
    with pytest.raises(ValueError, match="'level' is not a particle column"):
        particle_background(_particles(), _field_3d().rename(pres="level"))


# -- endpoint_weights / fill_missing --------------------------------------------------


def test_endpoint_weights_are_one_without_transforms_and_the_transform_factor_with():
    ones = endpoint_weights(_particles())
    assert ones.tolist() == [1.0, 1.0, 1.0]
    assert ones.index.name == "indx"

    decayed = endpoint_weights(_particles(), [FirstOrderLifetime(lifetime_hours=1.0)])
    assert decayed.to_numpy() == pytest.approx(np.exp(-np.array([1.0, 1.0, 2.0])))


def test_fill_missing_uses_the_weighted_mean_of_the_others():
    per = pd.Series([1.0, np.nan, 3.0], index=pd.Index([1, 2, 3], name="indx"))
    w = pd.Series([3.0, 1.0, 1.0], index=per.index)

    filled = fill_missing(per, w)

    assert filled.tolist() == [1.0, 1.5, 3.0]
    assert np.isnan(fill_missing(per * np.nan, w)).all()


# -- background ----------------------------------------------------------------------


def test_background_without_transforms_is_the_mean_over_particles():
    result = background(_particles(), _field_3d())

    assert isinstance(result, Background)
    assert result.value == pytest.approx(np.mean(ENDPOINT_VALUES))
    assert result.weights.tolist() == pytest.approx([1 / 3] * 3)
    assert result.per_particle.tolist() == ENDPOINT_VALUES


def test_background_accepts_values_sampled_elsewhere():
    sampled = pd.Series(ENDPOINT_VALUES, index=pd.Index([1, 2, 3], name="indx"))
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


def test_background_leaves_out_particles_outside_the_field():
    p = _particles()
    p.loc[p["indx"] == 2, "long"] = -100.0  # particle 2 ends off the field

    result = background(p, _field_3d())

    assert np.isnan(result.per_particle.loc[2])
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.value == pytest.approx((0.0 + 104.0) / 2)
    p["long"] = -100.0
    assert np.isnan(background(p, _field_3d()).value)


def _column(n=60, p_sfc=1000.0):
    """One-row column particles released 0-3000 m on a hypsometric profile."""
    z = np.linspace(0.0, 3000.0, n)
    return pd.DataFrame(
        {
            "indx": np.arange(1, n + 1),
            "time": np.full(n, -60.0),
            "long": np.full(n, -111.0),
            "lati": np.full(n, 40.0),
            "zagl": z,
            "xhgt": z,
            "pres": p_sfc * np.exp(-z / 8000.0),
            "foot": np.ones(n),
        }
    )


def test_background_pressure_weighting_covers_the_column_mass_fraction():
    p = _column()
    field = _field(values=np.full((2, 3), 1900.0))

    result = background(p, field, transforms=[PressureWeighting()])

    covered = 1.0 - p["pres"].min() / p["pres"].max()
    assert result.weights.sum() == pytest.approx(covered, abs=0.02)
    assert result.value == pytest.approx(1900.0 * result.weights.sum())
