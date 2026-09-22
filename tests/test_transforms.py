"""Tests for stilt.transforms: built-in particle transforms, loading and dumping."""

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from stilt.transforms import (
    AveragingKernel,
    FirstOrderLifetime,
    ParticleTransform,
    PressureWeighting,
    TransformContext,
    UnresolvedTransform,
    apply_transforms,
    dump_transform,
    load_transform,
    transform_kind,
)

# ---------------------------------------------------------------------------
# User-defined transforms (module level so their dotted path is importable)
# ---------------------------------------------------------------------------


class ScaleFoot(BaseModel):
    """A pydantic user transform: multiply foot by a constant."""

    model_config = ConfigDict(frozen=True)

    factor: float = 1.0

    def apply(self, particles, context=None):
        out = particles.copy()
        out["foot"] = out["foot"] * self.factor
        return out


class PlainScale:
    """A plain (non-pydantic) user transform built from keyword arguments."""

    def __init__(self, factor=1.0):
        self.factor = factor

    def apply(self, particles, context=None):
        out = particles.copy()
        out["foot"] = out["foot"] * self.factor
        return out


class NotATransform:
    """Importable, but has no apply()."""


SCALE_FOOT_KIND = f"{__name__}.ScaleFoot"
PLAIN_SCALE_KIND = f"{__name__}.PlainScale"
NOT_A_TRANSFORM_KIND = f"{__name__}.NotATransform"
MISSING_KIND = "no_such_pkg_for_stilt_tests.transforms.MyKernel"


# ---------------------------------------------------------------------------
# Particle fixtures
# ---------------------------------------------------------------------------


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


def _aged_particles() -> pd.DataFrame:
    """Three particles at 0, 1 and 2 hours of transport age (minutes)."""
    return pd.DataFrame(
        {
            "indx": [1, 2, 3],
            "time": [0.0, -60.0, -120.0],
            "foot": [1.0, 1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# TransformContext / ParticleTransform protocol
# ---------------------------------------------------------------------------


def test_transform_context_defaults(point_receptor):
    ctx = TransformContext(receptor=point_receptor)
    assert ctx.receptor is point_receptor
    assert ctx.footprint_name == ""
    assert ctx.is_error is False
    assert ctx.observation is None


def test_builtins_satisfy_particle_transform_protocol():
    assert isinstance(AveragingKernel(levels=[0.0], values=[1.0]), ParticleTransform)
    assert isinstance(PressureWeighting(), ParticleTransform)
    assert isinstance(FirstOrderLifetime(lifetime_hours=1.0), ParticleTransform)
    assert isinstance(ScaleFoot(), ParticleTransform)
    assert isinstance(PlainScale(), ParticleTransform)
    assert not isinstance(NotATransform(), ParticleTransform)


# ---------------------------------------------------------------------------
# AveragingKernel
# ---------------------------------------------------------------------------


def test_ak_interpolates_and_weights_foot():
    # Simple kernel: AK = 0 at z=0, 1 at z=1000 m.
    p = _make_particles(n=2, heights=[0.0, 1000.0])
    result = AveragingKernel(levels=[0.0, 1000.0], values=[0.0, 1.0]).apply(p)

    assert result["ak_weight"].tolist() == pytest.approx([0.0, 1.0])
    assert result["foot"].tolist() == pytest.approx([0.0, 1.0])


def test_ak_interpolates_between_levels():
    p = _make_particles(n=3, heights=[0.0, 500.0, 1000.0])
    result = AveragingKernel(levels=[0.0, 1000.0], values=[0.2, 0.8]).apply(p)
    assert result["foot"].tolist() == pytest.approx([0.2, 0.5, 0.8])


def test_ak_pressure_coordinate_sorts_ascending():
    # Profiles stored in top-to-bottom order (decreasing pressure).
    # Particles near 1000 hPa (surface) should get weight=1.0,
    # particles near 200 hPa (upper) should get weight=0.0.
    p = _make_particles(n=2, heights=[0.0, 0.0])  # heights unused
    p["pres"] = [1000.0, 200.0]
    kernel = AveragingKernel(
        levels=[900.0, 300.0],  # stored high-to-low pressure
        values=[1.0, 0.0],
        coordinate="pres",
    )
    result = kernel.apply(p)

    # After sorting ascending: levels=[300,900], values=[0,1].
    # Particle at 1000 hPa clamps to right edge value -> 1.0.
    # Particle at 200 hPa clamps to left edge value -> 0.0.
    assert result["foot"].tolist() == pytest.approx([1.0, 0.0])


def test_ak_pressure_coordinate_uses_release_row_for_whole_trajectory():
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
    kernel = AveragingKernel(
        levels=[600.0, 900.0], values=[0.0, 1.0], coordinate="pres"
    )
    result = kernel.apply(p)
    assert result["foot"].tolist() == pytest.approx([1.0] * 3 + [1.0 / 3.0] * 3)


def test_ak_missing_coordinate_column_raises():
    p = _make_particles().drop(columns=["xhgt"])
    kernel = AveragingKernel(levels=[0.0, 1000.0], values=[0.5, 0.5])
    with pytest.raises(ValueError, match="xhgt"):
        kernel.apply(p)


def test_ak_mismatched_levels_values_raises_at_construction():
    with pytest.raises(ValidationError, match="same length"):
        AveragingKernel(levels=[0.0, 1000.0], values=[0.5])


def test_ak_empty_levels_raises_at_construction():
    with pytest.raises(ValidationError, match="non-empty"):
        AveragingKernel(levels=[], values=[])


# ---------------------------------------------------------------------------
# PressureWeighting: particle-derived pressure weighting
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
    result = PressureWeighting().apply(_column_particles(n))

    expected = _expected_pwf(n)
    assert result["pwf"].to_numpy() == pytest.approx(expected, rel=1e-6)
    # Every particle carries real weight, including the one at the surface.
    assert (result["pwf"].to_numpy() > 0).all()
    # weight = pwf x n_particles, cancelling Footprint.calculate's mean.
    assert result["foot"].to_numpy() == pytest.approx(expected * n, rel=1e-6)


def test_pwf_release_pressure_follows_hypsometric_fit():
    n = 6
    result = PressureWeighting().apply(_column_particles(n))
    expected = P_SFC * np.exp(-np.linspace(0.0, 3000.0, n) / SCALE_HEIGHT)
    assert result["xpres"].to_numpy() == pytest.approx(expected, rel=1e-6)


def test_pwf_sums_to_fraction_of_column_covered():
    # A 0-3 km column holds 1 - exp(-3000/8000) ~ 31% of the atmosphere's mass.
    result = PressureWeighting().apply(_column_particles(200))
    assert result["pwf"].sum() == pytest.approx(
        1.0 - np.exp(-3000.0 / SCALE_HEIGHT), rel=1e-2
    )
    assert result["pwf"].sum() < 1.0


def test_pwf_taller_column_covers_more_mass():
    shallow = PressureWeighting().apply(_column_particles(100, z_top=1000.0))
    deep = PressureWeighting().apply(_column_particles(100, z_top=6000.0))
    assert deep["pwf"].sum() > shallow["pwf"].sum()


def test_pwf_weights_decrease_with_height():
    # Equal height steps span less air mass higher up.
    result = PressureWeighting().apply(_column_particles(20))
    pwf = result.sort_values("xhgt")["pwf"].to_numpy()
    assert (np.diff(pwf[1:]) < 0).all()


def test_pwf_footprint_magnitude_independent_of_numpar():
    # The bug Jacob Bushey hit: weighted footprints scaled with numpar.
    small = PressureWeighting().apply(_column_particles(100))
    large = PressureWeighting().apply(_column_particles(1000))
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
    weighted = PressureWeighting().apply(p)
    ratio = weighted["foot"].sum() / p["foot"].sum()
    assert 0.1 < ratio < 10.0


def test_pwf_uses_release_row_not_drifted_rows():
    p = _column_particles(5, rows_per_particle=3)
    result = PressureWeighting().apply(p)

    # One weight per particle, broadcast along its whole trajectory.
    assert (result.groupby("indx")["foot"].nunique() == 1).all()
    release = p.loc[p["time"] == -1.0].set_index("indx")["pres"]
    got = result.drop_duplicates("indx").set_index("indx")["xpres"]
    assert got.to_numpy() == pytest.approx(release.to_numpy(), rel=1e-6)


def test_pwf_surface_pressure_override_rescales_column():
    p = _column_particles(50)
    fitted = PressureWeighting().apply(p)
    pinned = PressureWeighting(surface_pressure=900.0).apply(p)
    # Same column shape, referenced to the supplied surface pressure.
    assert pinned["xpres"].to_numpy() == pytest.approx(
        fitted["xpres"].to_numpy() * 0.9, rel=1e-6
    )
    assert pinned["pwf"].sum() == pytest.approx(fitted["pwf"].sum(), rel=1e-6)


def test_pwf_elevated_column_bottom_assigns_air_below_to_lowest_particle():
    # A column starting at 500 m still measures the air beneath it; the lowest
    # particle carries that sub-column.
    result = PressureWeighting().apply(_column_particles(20, z_bottom=500.0))
    lowest = result.sort_values("xhgt")["pwf"].iloc[0]
    second = result.sort_values("xhgt")["pwf"].iloc[1]
    assert lowest > second


def test_pwf_requires_pressure_variable():
    p = _column_particles(5).drop(columns=["pres"])
    with pytest.raises(ValueError, match="'pres'"):
        PressureWeighting().apply(p)


def test_pwf_requires_zagl_variable():
    p = _column_particles(5).drop(columns=["zagl"])
    with pytest.raises(ValueError, match="'zagl'"):
        PressureWeighting().apply(p)


def test_pwf_rejects_single_release_height():
    p = _column_particles(5)
    p["zagl"] = 100.0
    with pytest.raises(ValueError, match="range of heights"):
        PressureWeighting().apply(p)


def test_pwf_rejects_pressure_increasing_with_height():
    p = _column_particles(10)
    p["pres"] = 900.0 + p["zagl"] / 100.0
    with pytest.raises(ValueError, match="does not decrease with height"):
        PressureWeighting().apply(p)


def test_pwf_rejects_nonpositive_surface_pressure():
    with pytest.raises(ValidationError):
        PressureWeighting(surface_pressure=0.0)


# ---------------------------------------------------------------------------
# AveragingKernel + PressureWeighting in sequence
# ---------------------------------------------------------------------------


def test_ak_then_pwf_is_product_of_the_two_weights():
    p = _column_particles(50)
    kernel = AveragingKernel(levels=[0.0, 3000.0], values=[0.5, 0.5])
    pwf_only = PressureWeighting().apply(p)
    both = PressureWeighting().apply(kernel.apply(p))

    assert both["foot"].to_numpy() == pytest.approx(pwf_only["foot"].to_numpy() * 0.5)
    assert both["foot"].to_numpy() == pytest.approx(
        both["ak_weight"].to_numpy() * both["pwf"].to_numpy() * 50
    )


def test_ak_then_pwf_kernel_on_height_varies_with_release_height():
    p = _column_particles(50)
    kernel = AveragingKernel(levels=[0.0, 3000.0], values=[1.0, 0.0])
    result = PressureWeighting().apply(kernel.apply(p))
    ratio = result["foot"].to_numpy() / PressureWeighting().apply(p)["foot"].to_numpy()
    # Kernel falls linearly from 1 at the surface to 0 at the column top.
    assert ratio[0] == pytest.approx(1.0, abs=1e-6)
    assert ratio[-1] == pytest.approx(0.0, abs=1e-6)
    assert ratio == pytest.approx(result["ak_weight"].to_numpy(), abs=1e-6)


def test_pwf_then_ak_kernel_on_pressure_coordinate():
    p = _column_particles(50)
    # Kernel defined on pressure: 1 at 1000 hPa (surface), 0 at 500 hPa.
    kernel = AveragingKernel(
        levels=[500.0, 1000.0], values=[0.0, 1.0], coordinate="pres"
    )
    pwf_only = PressureWeighting().apply(p)
    result = kernel.apply(pwf_only)
    ratio = result["foot"].to_numpy() / pwf_only["foot"].to_numpy()
    xpres = result["xpres"].to_numpy()
    assert ratio == pytest.approx(np.clip((xpres - 500.0) / 500.0, 0.0, 1.0), rel=1e-6)


def test_order_does_not_matter_for_multiplicative_weights():
    p = _column_particles(30)
    kernel = AveragingKernel(levels=[0.0, 3000.0], values=[1.0, 0.2])
    ak_first = PressureWeighting().apply(kernel.apply(p))
    pwf_first = kernel.apply(PressureWeighting().apply(p))
    assert ak_first["foot"].to_numpy() == pytest.approx(pwf_first["foot"].to_numpy())


# ---------------------------------------------------------------------------
# Transforms apply once; re-applying compounds; inputs are never mutated
# ---------------------------------------------------------------------------


def test_reapplying_pwf_compounds():
    p = _column_particles(10)
    once = PressureWeighting().apply(p)
    twice = PressureWeighting().apply(once)
    expected = once["foot"].to_numpy() * once["pwf"].to_numpy() * 10
    assert twice["foot"].to_numpy() == pytest.approx(expected)
    assert twice["pwf"].to_numpy() == pytest.approx(once["pwf"].to_numpy())


def test_reapplying_ak_compounds():
    p = _make_particles(n=2, heights=[0.0, 1000.0])
    kernel = AveragingKernel(levels=[0.0, 1000.0], values=[0.5, 0.5])
    once = kernel.apply(p)
    twice = kernel.apply(once)
    assert once["foot"].tolist() == pytest.approx([0.5, 0.5])
    assert twice["foot"].tolist() == pytest.approx([0.25, 0.25])


@pytest.mark.parametrize(
    "transform",
    [
        AveragingKernel(levels=[0.0, 3000.0], values=[0.2, 0.8]),
        PressureWeighting(),
        FirstOrderLifetime(lifetime_hours=1.0),
    ],
    ids=["ak", "pwf", "lifetime"],
)
def test_apply_does_not_mutate_input(transform):
    p = _column_particles(10)
    before = p.copy(deep=True)
    result = transform.apply(p)
    assert result is not p
    pd.testing.assert_frame_equal(p, before)
    assert (result["foot"].to_numpy() != 1.0).any()


# ---------------------------------------------------------------------------
# FirstOrderLifetime
# ---------------------------------------------------------------------------


def test_first_order_lifetime_decays_by_transport_age():
    result = FirstOrderLifetime(lifetime_hours=1.0).apply(_aged_particles())
    assert result["foot"].tolist() == pytest.approx(
        [1.0, math.exp(-1.0), math.exp(-2.0)]
    )


def test_first_order_lifetime_honours_time_unit():
    p = _aged_particles()
    p["age_h"] = [0.0, -1.0, -2.0]
    result = FirstOrderLifetime(
        lifetime_hours=1.0, time_column="age_h", time_unit="h"
    ).apply(p)
    assert result["foot"].tolist() == pytest.approx(
        [1.0, math.exp(-1.0), math.exp(-2.0)]
    )


def test_first_order_lifetime_requires_transport_time_column():
    p = _aged_particles().drop(columns=["time"])
    with pytest.raises(ValueError, match="time"):
        FirstOrderLifetime(lifetime_hours=1.0).apply(p)


def test_first_order_lifetime_rejects_unknown_time_unit_at_construction():
    with pytest.raises(ValidationError):
        FirstOrderLifetime(lifetime_hours=1.0, time_unit="fortnight")  # type: ignore[arg-type]


def test_first_order_lifetime_rejects_nonpositive_lifetime():
    with pytest.raises(ValidationError):
        FirstOrderLifetime(lifetime_hours=0.0)


# ---------------------------------------------------------------------------
# apply_transforms
# ---------------------------------------------------------------------------


def test_apply_transforms_with_no_transforms_returns_same_object(point_receptor):
    p = _make_particles()
    ctx = TransformContext(receptor=point_receptor)
    assert apply_transforms(p, [], ctx) is p


def test_apply_transforms_runs_in_order(point_receptor):
    p = _column_particles(20)
    ctx = TransformContext(receptor=point_receptor)
    kernel = AveragingKernel(levels=[0.0, 3000.0], values=[1.0, 0.0])
    result = apply_transforms(
        p, [kernel, PressureWeighting(), ScaleFoot(factor=2.0)], ctx
    )
    expected = PressureWeighting().apply(kernel.apply(p))["foot"].to_numpy() * 2.0
    assert result["foot"].to_numpy() == pytest.approx(expected)
    assert "ak_weight" in result.columns
    assert "pwf" in result.columns


def test_apply_transforms_passes_context_through(point_receptor):
    seen = []

    class Recorder:
        def apply(self, particles, context):
            seen.append(context)
            return particles

    ctx = TransformContext(
        receptor=point_receptor, footprint_name="column", is_error=True
    )
    apply_transforms(_make_particles(), [Recorder()], ctx)
    assert seen == [ctx]


# ---------------------------------------------------------------------------
# load_transform
# ---------------------------------------------------------------------------


def test_load_transform_averaging_kernel():
    t = load_transform(
        {"kind": "averaging_kernel", "levels": [0.0, 1000.0], "values": [0.2, 0.8]}
    )
    assert isinstance(t, AveragingKernel)
    assert t.levels == [0.0, 1000.0]
    assert t.values == [0.2, 0.8]
    assert t.coordinate == "xhgt"


def test_load_transform_pressure_weighting():
    assert load_transform({"kind": "pressure_weighting"}) == PressureWeighting()
    t = load_transform({"kind": "pressure_weighting", "surface_pressure": 850.0})
    assert isinstance(t, PressureWeighting)
    assert t.surface_pressure == pytest.approx(850.0)


def test_load_transform_first_order_lifetime():
    t = load_transform({"kind": "first_order_lifetime", "lifetime_hours": 4.0})
    assert isinstance(t, FirstOrderLifetime)
    assert t.lifetime_hours == pytest.approx(4.0)
    assert t.time_column == "time"
    assert t.time_unit == "min"


def test_load_transform_passes_through_object_with_apply():
    for obj in (PressureWeighting(), ScaleFoot(factor=3.0), PlainScale(factor=3.0)):
        assert load_transform(obj) is obj


def test_load_transform_dotted_kind_pydantic_class():
    t = load_transform({"kind": SCALE_FOOT_KIND, "factor": 2.5})
    assert isinstance(t, ScaleFoot)
    assert t.factor == pytest.approx(2.5)
    result = t.apply(_make_particles(n=3))
    assert result["foot"].tolist() == pytest.approx([2.5] * 3)


def test_load_transform_dotted_kind_plain_class():
    t = load_transform({"kind": PLAIN_SCALE_KIND, "factor": 4.0})
    assert isinstance(t, PlainScale)
    assert t.factor == pytest.approx(4.0)
    assert load_transform({"kind": PLAIN_SCALE_KIND}).factor == pytest.approx(1.0)


def test_load_transform_nonexistent_module_yields_unresolved():
    t = load_transform({"kind": MISSING_KIND, "levels": [0.0, 1.0]})
    assert isinstance(t, UnresolvedTransform)
    assert t.kind == MISSING_KIND
    assert t.reason  # carries the original import error
    assert t.model_dump()["levels"] == [0.0, 1.0]  # extra keys preserved
    with pytest.raises(ImportError, match="could not be imported"):
        t.apply(_make_particles())


def test_load_transform_nonexistent_attribute_yields_unresolved():
    t = load_transform({"kind": f"{__name__}.NoSuchClass"})
    assert isinstance(t, UnresolvedTransform)
    assert "NoSuchClass" in t.reason


def test_load_transform_class_without_apply_raises():
    with pytest.raises(TypeError, match="apply"):
        load_transform({"kind": NOT_A_TRANSFORM_KIND})


def test_load_transform_rejects_non_mapping_entry():
    with pytest.raises(TypeError, match="apply"):
        load_transform(42)
    with pytest.raises(TypeError):
        load_transform("averaging_kernel")


def test_load_transform_requires_kind():
    with pytest.raises(ValueError, match="kind"):
        load_transform({"levels": [0.0], "values": [1.0]})
    with pytest.raises(ValueError, match="kind"):
        load_transform({"kind": 3})


def test_load_transform_unknown_builtin_kind_raises():
    with pytest.raises(ValidationError):
        load_transform({"kind": "nonsense"})


def test_load_transform_builtin_validates_fields():
    with pytest.raises(ValidationError, match="same length"):
        load_transform(
            {"kind": "averaging_kernel", "levels": [0.0, 1.0], "values": [1.0]}
        )


# ---------------------------------------------------------------------------
# dump_transform / transform_kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("transform", "kind"),
    [
        (
            AveragingKernel(levels=[0.0, 1000.0], values=[0.2, 0.8], coordinate="pres"),
            "averaging_kernel",
        ),
        (PressureWeighting(surface_pressure=900.0), "pressure_weighting"),
        (
            FirstOrderLifetime(lifetime_hours=2.0, time_column="age", time_unit="h"),
            "first_order_lifetime",
        ),
        (ScaleFoot(factor=1.5), SCALE_FOOT_KIND),
    ],
    ids=["ak", "pwf", "lifetime", "user_pydantic"],
)
def test_dump_transform_round_trips(transform, kind):
    dumped = dump_transform(transform)
    assert dumped["kind"] == kind
    assert next(iter(dumped)) == "kind"
    assert load_transform(dumped) == transform


def test_dump_transform_builtin_contents():
    dumped = dump_transform(AveragingKernel(levels=[0.0, 1000.0], values=[0.2, 0.8]))
    assert dumped == {
        "kind": "averaging_kernel",
        "levels": [0.0, 1000.0],
        "values": [0.2, 0.8],
        "coordinate": "xhgt",
    }
    assert dump_transform(PressureWeighting()) == {
        "kind": "pressure_weighting",
        "surface_pressure": None,
    }


def test_dump_transform_user_pydantic_uses_dotted_path():
    assert dump_transform(ScaleFoot(factor=1.5)) == {
        "kind": SCALE_FOOT_KIND,
        "factor": 1.5,
    }


def test_dump_transform_unresolved_round_trips():
    t = load_transform({"kind": MISSING_KIND, "levels": [0.0]})
    dumped = dump_transform(t)
    assert dumped["kind"] == MISSING_KIND
    assert dumped["levels"] == [0.0]
    again = load_transform(dumped)
    assert isinstance(again, UnresolvedTransform)
    assert again.kind == MISSING_KIND


def test_dump_transform_rejects_plain_object():
    with pytest.raises(TypeError, match="pydantic"):
        dump_transform(PlainScale())


def test_transform_kind():
    assert transform_kind(AveragingKernel(levels=[0.0], values=[1.0])) == (
        "averaging_kernel"
    )
    assert transform_kind(PressureWeighting()) == "pressure_weighting"
    assert transform_kind(FirstOrderLifetime(lifetime_hours=1.0)) == (
        "first_order_lifetime"
    )
    assert transform_kind(ScaleFoot()) == SCALE_FOOT_KIND
    assert transform_kind(PlainScale()) == PLAIN_SCALE_KIND
    assert transform_kind(UnresolvedTransform(kind=MISSING_KIND)) == MISSING_KIND
