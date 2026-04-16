"""Tests for the generic chemistry seam."""

import math

import pandas as pd
import pytest

from stilt.observations import (
    ChemistryContext,
    FirstOrderLifetimeChemistry,
    NoOpChemistry,
    Observation,
    apply_chemistry,
)


def _make_particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "indx": [1, 2, 3],
            "time": [0.0, -60.0, -120.0],
            "foot": [1.0, 1.0, 1.0],
        }
    )


def test_apply_chemistry_with_noop_returns_unchanged_copy():
    particles = _make_particles()

    result = apply_chemistry(particles, NoOpChemistry())

    assert result is not particles
    pd.testing.assert_frame_equal(result, particles)


def test_first_order_lifetime_chemistry_decays_by_transport_age():
    particles = _make_particles()
    observation = Observation(
        sensor="tropomi",
        species="no2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )
    context = ChemistryContext(observation=observation, species="no2")

    result = apply_chemistry(
        particles,
        FirstOrderLifetimeChemistry(lifetime_hours=1.0),
        context=context,
    )

    assert result["foot"].tolist() == pytest.approx(
        [1.0, math.exp(-1.0), math.exp(-2.0)]
    )
    assert result["foot_before_chemistry"].tolist() == [1.0, 1.0, 1.0]


def test_first_order_lifetime_chemistry_is_idempotent_on_reapply():
    particles = _make_particles()
    chemistry = FirstOrderLifetimeChemistry(lifetime_hours=1.0)

    once = apply_chemistry(particles, chemistry)
    twice = apply_chemistry(once, chemistry)

    assert twice["foot"].tolist() == pytest.approx(once["foot"].tolist())


def test_first_order_lifetime_requires_transport_time_column():
    particles = _make_particles().drop(columns=["time"])

    with pytest.raises(ValueError, match="time"):
        apply_chemistry(particles, FirstOrderLifetimeChemistry(lifetime_hours=1.0))


def test_first_order_lifetime_rejects_unknown_time_unit():
    particles = _make_particles()
    context = ChemistryContext(time_unit="fortnight")

    with pytest.raises(ValueError, match="Unsupported chemistry time unit"):
        apply_chemistry(
            particles,
            FirstOrderLifetimeChemistry(lifetime_hours=1.0),
            context=context,
        )
