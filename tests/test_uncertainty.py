"""Tests for the generic observation uncertainty seam."""

import math

import pytest

from stilt.observations import Observation, UncertaintyBudget, UncertaintyComponent


def test_uncertainty_budget_computes_root_sum_square_total():
    budget = UncertaintyBudget(
        components=(
            UncertaintyComponent(name="measurement", sigma=2.0),
            UncertaintyComponent(name="transport", sigma=3.0),
            UncertaintyComponent(name="background", sigma=6.0),
        ),
        units="ppm",
    )

    assert budget.total == pytest.approx(7.0)
    assert budget.to_mapping() == {
        "measurement": 2.0,
        "transport": 3.0,
        "background": 6.0,
    }
    assert budget.component("transport") == UncertaintyComponent(
        name="transport", sigma=3.0
    )
    assert budget.component("missing") is None


def test_uncertainty_budget_from_mapping_preserves_total():
    budget = UncertaintyBudget.from_mapping(
        {"measurement": 1.0, "transport": 2.0},
        units="ppb",
    )

    assert budget.units == "ppb"
    assert budget.total == pytest.approx(math.sqrt(5.0))


def test_uncertainty_budget_requires_unique_component_names():
    with pytest.raises(ValueError, match="unique"):
        UncertaintyBudget(
            components=(
                UncertaintyComponent(name="measurement", sigma=1.0),
                UncertaintyComponent(name="measurement", sigma=2.0),
            )
        )


def test_uncertainty_component_rejects_negative_sigma():
    with pytest.raises(ValueError, match=">= 0"):
        UncertaintyComponent(name="measurement", sigma=-1.0)


def test_observation_derives_scalar_uncertainty_from_budget_when_missing():
    budget = UncertaintyBudget.from_mapping(
        {"measurement": 1.0, "transport": 2.0},
        units="ppm",
    )

    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        uncertainty_budget=budget,
    )

    assert observation.uncertainty_budget is budget
    assert observation.uncertainty == pytest.approx(math.sqrt(5.0))


def test_observation_keeps_explicit_scalar_uncertainty_when_budget_is_present():
    budget = UncertaintyBudget.from_mapping({"measurement": 1.0, "transport": 2.0})

    observation = Observation(
        sensor="tropomi",
        species="xch4",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        uncertainty=9.0,
        uncertainty_budget=budget,
    )

    assert observation.uncertainty == 9.0
