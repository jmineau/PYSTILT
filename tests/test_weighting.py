"""Tests for the generic observation weighting seam."""

import pandas as pd
import pytest

from stilt.observations import (
    NoOpWeighting,
    Observation,
    VerticalOperator,
    VerticalOperatorWeighting,
    WeightingContext,
    apply_weighting,
)


def _make_particles(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "indx": list(range(1, n + 1)),
            "xhgt": [0.0, 500.0, 1000.0][:n],
            "foot": [1.0] * n,
            "time": [pd.Timestamp("2023-01-01")] * n,
        }
    )


def test_apply_weighting_with_noop_returns_unchanged_copy():
    particles = _make_particles()

    result = apply_weighting(particles, NoOpWeighting())

    assert result is not particles
    pd.testing.assert_frame_equal(result, particles)


def test_vertical_operator_weighting_uses_context_operator_and_coordinate():
    particles = _make_particles(n=2)
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )
    operator = VerticalOperator(
        mode="ak",
        levels=[0.0, 1000.0],
        values=[0.0, 1.0],
    )
    context = WeightingContext(
        observation=observation,
        operator=operator,
        coordinate="xhgt",
    )

    result = apply_weighting(
        particles,
        VerticalOperatorWeighting(),
        context=context,
    )

    assert result["foot"].tolist() == pytest.approx([0.0, 0.5])
    assert result["foot_before_weight"].tolist() == [1.0, 1.0]


def test_vertical_operator_weighting_requires_operator_somewhere():
    particles = _make_particles()

    with pytest.raises(ValueError, match="VerticalOperator"):
        apply_weighting(particles, VerticalOperatorWeighting())
