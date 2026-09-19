"""Tests for apply_vertical_operator."""

import numpy as np
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


def test_pressure_coordinate_uses_release_row_for_whole_trajectory():
    # ``pres`` changes along each trajectory; the weight must come from the
    # release row (smallest |time|) and be constant per particle.
    p = pd.DataFrame(
        {
            "indx": [1, 1, 1, 2, 2, 2],
            "time": [-1.0, -2.0, -3.0, -3.0, -1.0, -2.0],
            "pres": [900.0, 850.0, 800.0, 600.0, 700.0, 650.0],
            "foot": [1.0] * 6,
        }
    )
    operator = VerticalOperator(mode="ak", levels=[600.0, 900.0], values=[0.0, 1.0])
    result = apply_vertical_operator(p, operator, coordinate="pres")
    assert result["foot"].tolist() == pytest.approx([1.0] * 3 + [1.0 / 3.0] * 3)


# ---------------------------------------------------------------------------
# Particle-derived pressure weighting
#
# Reference atmosphere for these tests: isothermal, so pressure falls off
# exactly as p(z) = P_SFC * exp(-z / SCALE_HEIGHT).  That makes every expected
# weight computable in closed form.
# ---------------------------------------------------------------------------

P_SFC = 1000.0
SCALE_HEIGHT = 8000.0


def _column_particles(
    n: int,
    z_top: float = 3000.0,
    z_bottom: float = 0.0,
    rows_per_particle: int = 1,
) -> pd.DataFrame:
    """
    Particles released evenly in height through an isothermal atmosphere.

    ``rows_per_particle`` > 1 adds later trajectory steps that drift in height
    and pressure, so tests can check that only the release row is used.
    """
    z = np.linspace(z_bottom, z_top, n)
    pres = P_SFC * np.exp(-z / SCALE_HEIGHT)
    frames = [
        pd.DataFrame(
            {
                "indx": np.arange(1, n + 1),
                "time": -(step + 1.0),
                "xhgt": z,
                "zagl": z + 10.0 * step,
                "pres": pres * (1.0 - 0.001 * step),
                "foot": 1.0,
            }
        )
        for step in range(rows_per_particle)
    ]
    return pd.concat(frames, ignore_index=True)


def _expected_pwf(n: int, z_top: float = 3000.0, z_bottom: float = 0.0) -> np.ndarray:
    """
    Closed-form weights for :func:`_column_particles`.

    Each particle owns the slab centred on it: edges midway between adjacent
    release pressures, the surface closing the bottom, and the top slab
    mirroring its lower half-width.
    """
    levels = P_SFC * np.exp(-np.linspace(z_bottom, z_top, n) / SCALE_HEIGHT)
    mids = (levels[:-1] + levels[1:]) / 2.0
    lower = np.concatenate(([P_SFC], mids))
    upper = np.concatenate((mids, [2 * levels[-1] - mids[-1]]))
    return (lower - upper) / P_SFC


def test_pwf_weights_match_cell_edges():
    n = 5
    result = apply_vertical_operator(_column_particles(n), VerticalOperator(mode="pwf"))

    expected = _expected_pwf(n)
    assert result["pwf"].to_numpy() == pytest.approx(expected, rel=1e-6)
    # Every particle carries real weight, including the one at the surface.
    assert (result["pwf"].to_numpy() > 0).all()
    # weight = pwf x n_particles, cancelling Footprint.calculate's mean.
    assert result["foot"].to_numpy() == pytest.approx(expected * n, rel=1e-6)
    assert result["foot_before_weight"].tolist() == [1.0] * n


def test_pwf_release_pressure_follows_hypsometric_fit():
    n = 6
    result = apply_vertical_operator(_column_particles(n), VerticalOperator(mode="pwf"))
    expected = P_SFC * np.exp(-np.linspace(0.0, 3000.0, n) / SCALE_HEIGHT)
    assert result["xpres"].to_numpy() == pytest.approx(expected, rel=1e-6)


def test_pwf_sums_to_fraction_of_column_covered():
    # A 0-3 km column holds 1 - exp(-3000/8000) ~ 31% of the atmosphere's mass.
    result = apply_vertical_operator(
        _column_particles(200), VerticalOperator(mode="pwf")
    )
    assert result["pwf"].sum() == pytest.approx(
        1.0 - np.exp(-3000.0 / SCALE_HEIGHT), rel=1e-2
    )
    assert result["pwf"].sum() < 1.0


def test_pwf_taller_column_covers_more_mass():
    shallow = apply_vertical_operator(
        _column_particles(100, z_top=1000.0), VerticalOperator(mode="pwf")
    )
    deep = apply_vertical_operator(
        _column_particles(100, z_top=6000.0), VerticalOperator(mode="pwf")
    )
    assert deep["pwf"].sum() > shallow["pwf"].sum()


def test_pwf_weights_decrease_with_height():
    # Equal height steps span less air mass higher up.
    result = apply_vertical_operator(
        _column_particles(20), VerticalOperator(mode="pwf")
    )
    pwf = result.sort_values("xhgt")["pwf"].to_numpy()
    assert (np.diff(pwf[1:]) < 0).all()


def test_pwf_footprint_magnitude_independent_of_numpar():
    # The bug Jacob Bushey hit: weighted footprints scaled with numpar.
    small = apply_vertical_operator(
        _column_particles(100), VerticalOperator(mode="pwf")
    )
    large = apply_vertical_operator(
        _column_particles(1000), VerticalOperator(mode="pwf")
    )
    # Footprint.calculate divides by the particle count, so compare sum / N.
    # A 10x change in numpar moves the result by well under a percent; the
    # residual is the top cell's half-width, not a scaling with numpar.
    assert small["foot"].sum() / 100 == pytest.approx(
        large["foot"].sum() / 1000, rel=1e-2
    )


def test_pwf_magnitude_comparable_to_unweighted_footprint():
    # A weighted column footprint should stay the same order of magnitude as
    # an unweighted one, not be inflated or crushed by numpar.
    p = _column_particles(500)
    weighted = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    ratio = weighted["foot"].sum() / p["foot"].sum()
    assert 0.1 < ratio < 10.0


def test_pwf_uses_release_row_not_drifted_rows():
    p = _column_particles(5, rows_per_particle=3)
    result = apply_vertical_operator(p, VerticalOperator(mode="pwf"))

    # One weight per particle, broadcast along its whole trajectory.
    assert (result.groupby("indx")["foot"].nunique() == 1).all()
    release = p.loc[p["time"] == -1.0].set_index("indx")["pres"]
    got = result.drop_duplicates("indx").set_index("indx")["xpres"]
    assert got.to_numpy() == pytest.approx(release.to_numpy(), rel=1e-6)


def test_pwf_surface_pressure_override_rescales_column():
    p = _column_particles(50)
    fitted = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    pinned = apply_vertical_operator(
        p, VerticalOperator(mode="pwf", surface_pressure=900.0)
    )
    # Same column shape, referenced to the supplied surface pressure.
    assert pinned["xpres"].to_numpy() == pytest.approx(
        fitted["xpres"].to_numpy() * 0.9, rel=1e-6
    )
    assert pinned["pwf"].sum() == pytest.approx(fitted["pwf"].sum(), rel=1e-6)


def test_pwf_elevated_column_bottom_assigns_air_below_to_lowest_particle():
    # A column starting at 500 m still measures the air beneath it; the lowest
    # particle carries that sub-column.
    result = apply_vertical_operator(
        _column_particles(20, z_bottom=500.0), VerticalOperator(mode="pwf")
    )
    lowest = result.sort_values("xhgt")["pwf"].iloc[0]
    second = result.sort_values("xhgt")["pwf"].iloc[1]
    assert lowest > second


def test_ak_pwf_scales_pwf_by_averaging_kernel():
    p = _column_particles(50)
    pwf_only = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    both = apply_vertical_operator(
        p, VerticalOperator(mode="ak_pwf", levels=[0.0, 3000.0], values=[0.5, 0.5])
    )
    assert both["foot"].to_numpy() == pytest.approx(pwf_only["foot"].to_numpy() * 0.5)


def test_ak_pwf_kernel_on_height_varies_with_release_height():
    p = _column_particles(50)
    result = apply_vertical_operator(
        p, VerticalOperator(mode="ak_pwf", levels=[0.0, 3000.0], values=[1.0, 0.0])
    )
    ratio = (
        result["foot"].to_numpy()
        / apply_vertical_operator(p, VerticalOperator(mode="pwf"))["foot"].to_numpy()
    )
    # Kernel falls linearly from 1 at the surface to 0 at the column top.
    assert ratio[0] == pytest.approx(1.0, abs=1e-6)
    assert ratio[-1] == pytest.approx(0.0, abs=1e-6)


def test_ak_pwf_kernel_on_pressure_coordinate():
    p = _column_particles(50)
    # Kernel defined on pressure: 1 at 1000 hPa (surface), 0 at 500 hPa.
    result = apply_vertical_operator(
        p,
        VerticalOperator(mode="ak_pwf", levels=[500.0, 1000.0], values=[0.0, 1.0]),
        coordinate="pres",
    )
    pwf_only = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    ratio = result["foot"].to_numpy() / pwf_only["foot"].to_numpy()
    xpres = result["xpres"].to_numpy()
    assert ratio == pytest.approx(np.clip((xpres - 500.0) / 500.0, 0.0, 1.0), rel=1e-6)


def test_ak_pwf_requires_kernel_levels_and_values():
    p = _column_particles(10)
    with pytest.raises(ValueError, match="non-empty"):
        apply_vertical_operator(p, VerticalOperator(mode="ak_pwf"))


def test_pwf_ignores_levels_and_values():
    p = _column_particles(10)
    bare = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    noisy = apply_vertical_operator(
        p, VerticalOperator(mode="pwf", levels=[0.0, 1.0], values=[9.0, 9.0])
    )
    assert noisy["foot"].to_numpy() == pytest.approx(bare["foot"].to_numpy())


def test_pwf_requires_pressure_variable():
    p = _column_particles(5).drop(columns=["pres"])
    with pytest.raises(ValueError, match="'pres'"):
        apply_vertical_operator(p, VerticalOperator(mode="pwf"))


def test_pwf_requires_zagl_variable():
    p = _column_particles(5).drop(columns=["zagl"])
    with pytest.raises(ValueError, match="'zagl'"):
        apply_vertical_operator(p, VerticalOperator(mode="pwf"))


def test_pwf_rejects_single_release_height():
    p = _column_particles(5)
    p["zagl"] = 100.0
    with pytest.raises(ValueError, match="range of heights"):
        apply_vertical_operator(p, VerticalOperator(mode="pwf"))


def test_pwf_rejects_pressure_increasing_with_height():
    p = _column_particles(10)
    p["pres"] = 900.0 + p["zagl"] / 100.0
    with pytest.raises(ValueError, match="does not decrease with height"):
        apply_vertical_operator(p, VerticalOperator(mode="pwf"))


def test_pwf_reweighting_restores_original_foot():
    p = _column_particles(10)
    once = apply_vertical_operator(p, VerticalOperator(mode="pwf"))
    twice = apply_vertical_operator(once, VerticalOperator(mode="pwf"))
    assert twice["foot"].to_numpy() == pytest.approx(once["foot"].to_numpy())
    assert twice["foot_before_weight"].tolist() == [1.0] * 10


# ---------------------------------------------------------------------------
# Mode validation
#
# VerticalOperator is a plain dataclass, so its Literal annotation is not
# enforced at runtime. Without an explicit check an unknown mode used to fall
# through apply_vertical_operator leaving foot unweighted while still adding
# foot_before_weight -- a silent no-op that looked like it had worked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retired", "replacement"), [("integration", "pwf"), ("tccon", "ak_pwf")]
)
def test_retired_modes_raise_and_name_their_replacement(retired, replacement):
    with pytest.raises(ValueError, match=f"removed in 0.1.0a10.*{replacement}"):
        VerticalOperator(mode=retired, levels=[0.0, 3000.0], values=[1.0, 0.4])


def test_unknown_mode_raises_and_lists_valid_modes():
    with pytest.raises(ValueError, match="Unknown vertical-operator mode"):
        VerticalOperator(mode="nonsense")


@pytest.mark.parametrize("mode", ["none", "uniform", "ak", "pwf", "ak_pwf"])
def test_every_supported_mode_constructs(mode):
    assert (
        VerticalOperator(mode=mode, levels=[0.0, 1.0], values=[1.0, 1.0]).mode == mode
    )
