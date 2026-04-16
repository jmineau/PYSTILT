"""Tests for apply_vertical_operator."""

import pandas as pd
import pytest

from stilt.observations.apply import apply_vertical_operator
from stilt.observations.operators import VerticalOperator


def _make_particles(n: int = 10, heights: list[float] | None = None) -> pd.DataFrame:
    """Minimal particle DataFrame with xhgt, foot, indx columns."""
    if heights is None:
        heights = [float(i * 200) for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "indx": list(range(1, n + 1)),
            "xhgt": heights,
            "foot": [1.0] * n,
            "long": [-111.9] * n,
            "lati": [40.7] * n,
            "time": [pd.Timestamp("2023-01-01")] * n,
        }
    )


def test_mode_none_returns_original_object():
    p = _make_particles()
    operator = VerticalOperator(mode="none")
    result = apply_vertical_operator(p, operator)
    assert result is p  # no copy for mode="none"


def test_mode_uniform_returns_copy_unchanged():
    p = _make_particles()
    operator = VerticalOperator(mode="uniform")
    result = apply_vertical_operator(p, operator)
    assert result is not p
    pd.testing.assert_series_equal(result["foot"], p["foot"])
    assert "foot_before_weight" not in result.columns


def test_ak_mode_interpolates_and_weights_foot():
    # Simple operator: AK = 0 at z=0, 1 at z=1000 m.
    p = _make_particles(n=2, heights=[0.0, 1000.0])
    operator = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[0.0, 1.0])
    result = apply_vertical_operator(p, operator)

    assert result["foot_before_weight"].tolist() == [1.0, 1.0]
    assert result["foot"].tolist() == pytest.approx([0.0, 1.0])


def test_pwf_mode_scales_by_n_particles():
    n = 4
    # Uniform PWF: each particle gets 0.25, which × 4 particles = 1.0.
    p = _make_particles(n=n, heights=[100.0, 200.0, 300.0, 400.0])
    operator = VerticalOperator(
        mode="pwf",
        levels=[100.0, 200.0, 300.0, 400.0],
        values=[0.25, 0.25, 0.25, 0.25],
    )
    result = apply_vertical_operator(p, operator)

    # weight = 0.25 * 4 = 1.0, foot unchanged
    assert result["foot"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_ak_pwf_mode_scales_by_n_particles():
    n = 2
    # AK_norm × PWF: lower particle gets 0.1*0.5, upper gets 0.9*0.5.
    p = _make_particles(n=n, heights=[0.0, 1000.0])
    operator = VerticalOperator(
        mode="ak_pwf",
        levels=[0.0, 1000.0],
        values=[0.05, 0.45],  # AK*PWF pre-combined
    )
    result = apply_vertical_operator(p, operator)

    # weight = value * n_particles (=2)
    expected = [0.05 * 2, 0.45 * 2]
    assert result["foot"].tolist() == pytest.approx(expected)


def test_pressure_coordinate_sorts_ascending():
    # Profiles stored in top-to-bottom order (decreasing pressure).
    # Particles near 1000 hPa (surface) should get weight=1.0,
    # particles near 200 hPa (upper) should get weight=0.0.
    p = _make_particles(n=2, heights=[0.0, 0.0])  # heights unused
    p["pres"] = [1000.0, 200.0]
    operator = VerticalOperator(
        mode="ak",
        levels=[900.0, 300.0],  # stored high-to-low pressure
        values=[1.0, 0.0],
        pressure_levels=[900.0, 300.0],
    )
    result = apply_vertical_operator(p, operator, coordinate="pres")

    # After sorting ascending: levels=[300,900], values=[0,1].
    # Particle at 1000 hPa clamps to right edge value → 1.0.
    # Particle at 200 hPa clamps to left edge value → 0.0.
    assert result["foot"].tolist() == pytest.approx([1.0, 0.0])


def test_reweighting_restores_original_foot():
    p = _make_particles(n=2, heights=[0.0, 1000.0])
    op1 = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[0.5, 0.5])
    op2 = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[1.0, 1.0])

    after_first = apply_vertical_operator(p, op1)
    assert after_first["foot"].tolist() == pytest.approx([0.5, 0.5])

    after_second = apply_vertical_operator(after_first, op2)
    # Should re-apply to the original foot=1.0, not to 0.5.
    assert after_second["foot"].tolist() == pytest.approx([1.0, 1.0])


def test_missing_coordinate_column_raises():
    p = _make_particles()
    p = p.drop(columns=["xhgt"])
    operator = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[0.5, 0.5])
    with pytest.raises(ValueError, match="xhgt"):
        apply_vertical_operator(p, operator)


def test_mismatched_levels_values_raises():
    p = _make_particles()
    operator = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[0.5])
    with pytest.raises(ValueError, match="same length"):
        apply_vertical_operator(p, operator)


def test_empty_levels_raises():
    p = _make_particles()
    operator = VerticalOperator(mode="ak", levels=[], values=[])
    with pytest.raises(ValueError, match="non-empty"):
        apply_vertical_operator(p, operator)


def test_foot_before_weight_preserved():
    p = _make_particles(n=3, heights=[0.0, 500.0, 1000.0])
    operator = VerticalOperator(mode="ak", levels=[0.0, 1000.0], values=[0.2, 0.8])
    result = apply_vertical_operator(p, operator)

    assert "foot_before_weight" in result.columns
    assert result["foot_before_weight"].tolist() == [1.0, 1.0, 1.0]
    assert result["foot"].tolist() == pytest.approx([0.2, 0.5, 0.8])
