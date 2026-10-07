"""Tests for stilt.footprint.gridding: calc_footprint, its kernels, and its invariants."""

import builtins

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.exceptions import EmptyFootprint
from stilt.footprint import calc_footprint
from stilt.footprint.gridding import (
    _compute_kernel_bandwidths,
    _interpolate_early_timesteps,
    _interpolation_times,
    _make_gauss_kernel,
    _project_particles_to_crs,
    _wrap_antimeridian_longitudes,
)
from stilt.receptors import PointReceptor
from stilt.spatial import Grid


def _particles_in_domain(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create synthetic particle data within [-114, -113] x [39, 40]."""
    rng = np.random.default_rng(seed)
    times = [-60] * n + [-120] * n
    indx = list(range(1, n + 1)) * 2
    return pd.DataFrame(
        {
            "time": times,
            "particle": indx,
            "lon": rng.uniform(-113.9, -113.1, n * 2),
            "lat": rng.uniform(39.1, 39.9, n * 2),
            "zagl": rng.uniform(5, 100, n * 2),
            "foot": rng.uniform(1e-6, 1e-4, n * 2),
        }
    )


def _grid(xres=0.1, yres=0.1):
    return Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=xres, yres=yres)


def _interior_particles(n: int = 40, seed: int = 55) -> pd.DataFrame:
    """Particles well inside the domain, no t=0 receptor row."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": [-60.0] * n,
            "particle": [float(i + 1) for i in range(n)],
            "lon": rng.uniform(-113.8, -113.2, n),
            "lat": rng.uniform(39.2, 39.8, n),
            "zagl": [5.0] * n,
            "foot": rng.uniform(1e-5, 1e-4, n),
        }
    )


def _interior_footprint(particles, receptor, smooth_factor: float = 0.0):
    return calc_footprint(particles, receptor, _grid(), smooth_factor=smooth_factor)


def _first_hour_particles(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Backward particles that all stay within the first hour (layer -1)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": np.repeat([-20.0, -40.0, -59.0], n),
            "particle": np.tile(np.arange(1, n + 1), 3),
            "lon": rng.uniform(-112.2, -111.7, 3 * n),
            "lat": rng.uniform(40.6, 40.9, 3 * n),
            "zagl": rng.uniform(5, 100, 3 * n),
            "foot": rng.uniform(0.0, 1e-3, 3 * n),
        }
    )


def test_make_gauss_kernel_sigma_zero():
    k = _make_gauss_kernel((0.1, 0.1), sigma=0)
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calc_footprint()

# ---------------------------------------------------------------------------


def test_calc_footprint_returns_footprint_instance(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid())
    assert foot is not None
    assert isinstance(foot, xr.DataArray)


def test_calc_footprint_dims_are_time_lat_lon(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid())
    assert foot is not None
    assert tuple(foot.dims) == ("time", "lat", "lon")


def test_calc_footprint_raises_when_particles_outside_domain(
    point_receptor,
):
    """All particles outside the domain is an EmptyFootprint, not zeros."""
    particles = _particles_in_domain()
    particles["lon"] = 0.0  # far outside [-114, -113]
    particles["lat"] = 0.0
    with pytest.raises(EmptyFootprint):
        calc_footprint(particles, point_receptor, _grid())


def test_calc_footprint_raises_when_there_are_no_particles(point_receptor):
    particles = _particles_in_domain().iloc[0:0]
    with pytest.raises(EmptyFootprint):
        calc_footprint(particles, point_receptor, _grid())


def test_calc_footprint_assigns_name(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid(), name="test")
    assert foot is not None
    assert foot.stilt.name == "test"


def test_calc_footprint_time_integrate_collapses_to_single_timestep(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid(), time_integrate=True)
    assert foot is not None
    assert len(foot.time) == 1


def test_calc_footprint_nonnegative_foot_values(point_receptor):
    """Footprint values should be non-negative."""
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid())
    assert foot is not None
    assert float(foot.values.min()) >= 0.0


def test_calc_footprint_smooth_factor_zero(point_receptor):
    """smooth_factor=0 is equivalent to no smoothing (identity kernel)."""
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid(), smooth_factor=0.0)
    assert foot is not None


def test_calc_footprint_irregular_grid_uses_complete_cells(point_receptor):
    particles = _particles_in_domain()

    foot = calc_footprint(
        particles, point_receptor, _grid(xres=0.3, yres=0.4), smooth_factor=0.0
    )

    np.testing.assert_allclose(foot.lon.values, [-113.85, -113.55, -113.25])
    np.testing.assert_allclose(foot.lat.values, [39.2, 39.6])
    assert foot.shape == (2, 2, 3)


def test_calc_footprint_non_square_resolution_is_finite(point_receptor):
    particles = _particles_in_domain()

    foot = calc_footprint(particles, point_receptor, _grid(xres=0.01, yres=0.05))

    assert foot.sizes["lon"] == 100
    assert foot.sizes["lat"] == 20
    assert np.isfinite(foot.values).all()
    assert float(foot.values.min()) >= 0.0


def test_calc_footprint_grid_property(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid())
    assert foot is not None
    assert foot.stilt.grid.xres == pytest.approx(0.1)


def test_make_gauss_kernel_normalized():
    """Kernel values sum to 1.0."""
    k = _make_gauss_kernel((0.1, 0.1), sigma=0.5)
    assert k.sum() == pytest.approx(1.0, rel=1e-6)


def test_make_gauss_kernel_odd_shape():
    """Kernel shape must be odd in both dimensions."""
    k = _make_gauss_kernel((0.1, 0.1), sigma=0.3)
    assert k.shape[0] % 2 == 1
    assert k.shape[1] % 2 == 1


def test_make_gauss_kernel_symmetric():
    """
    For equal x/y resolution, the kernel is symmetric under flips and transposition.

    An asymmetric kernel would create directional bias — footprints
    would incorrectly favour one compass direction over another.
    """
    k = _make_gauss_kernel((0.01, 0.01), sigma=0.3)
    np.testing.assert_array_equal(
        k, k.T, err_msg="kernel must be symmetric under transpose"
    )
    np.testing.assert_array_equal(
        k, k[::-1, :], err_msg="kernel must be symmetric about row axis"
    )
    np.testing.assert_array_equal(
        k, k[:, ::-1], err_msg="kernel must be symmetric about col axis"
    )


def test_interpolation_times_match_r_stilt_schedule():
    times = _interpolation_times(-1)

    assert times[0] == pytest.approx(0.0)
    assert times[np.where(np.isclose(times, -10.0))[0][0]] == pytest.approx(-10.0)
    assert times[np.where(np.isclose(times, -20.0))[0][0]] == pytest.approx(-20.0)
    assert times[-1] == pytest.approx(-100.0)
    assert len(times) == 311


def test_interpolate_early_timesteps_preserves_window_foot_sums():
    particles = pd.DataFrame(
        {
            "time": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
            "particle": [1, 1, 1, 2, 2, 2],
            "lon": [-113.0, -114.0, -115.0, -112.0, -113.5, -115.0],
            "lat": [39.0, 40.0, 41.0, 39.5, 40.5, 41.5],
            "foot": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
        }
    )
    original_atime = np.abs(particles["time"])
    original_sums = [
        particles.loc[original_atime <= 10, "foot"].sum(),
        particles.loc[(original_atime > 10) & (original_atime <= 20), "foot"].sum(),
        particles.loc[(original_atime > 20) & (original_atime <= 100), "foot"].sum(),
    ]

    interpolated = _interpolate_early_timesteps(
        particles, xres=0.01, yres=0.01, time_sign=-1
    )

    assert len(interpolated) > len(particles)
    atime = np.abs(interpolated["time"])
    interpolated_sums = [
        interpolated.loc[atime <= 10, "foot"].sum(),
        interpolated.loc[(atime > 10) & (atime <= 20), "foot"].sum(),
        interpolated.loc[(atime > 20) & (atime <= 100), "foot"].sum(),
    ]
    assert interpolated_sums == pytest.approx(original_sums)
    assert interpolated[["lon", "lat", "foot"]].isna().sum().sum() == 0


def test_interpolate_early_timesteps_matches_r_na_omit_with_extra_columns():
    particles = pd.DataFrame(
        {
            "time": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
            "particle": [1, 1, 1, 2, 2, 2],
            "lon": [-113.0, -114.0, -115.0, -112.0, -113.5, -115.0],
            "lat": [39.0, 40.0, 41.0, 39.5, 40.5, 41.5],
            "zagl": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0],
            "foot": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
        }
    )

    interpolated = _interpolate_early_timesteps(
        particles, xres=0.01, yres=0.01, time_sign=-1
    )

    expected = particles.sort_values(
        ["particle", "time"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(interpolated, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# Mathematical invariants — no R required
#
# These properties must hold from pure math regardless of STILT-R agreement.
# They are the foundation of using PYSTILT footprints in linear inversion:
#   concentration = sum(footprint * flux)
# If the footprint is not linear in the particle sensitivity values, or if
# smoothing is not mass-conservative, that inversion is invalid.

# ---------------------------------------------------------------------------


def test_calc_footprint_linearity_in_foot_values(point_receptor):
    """
    Scaling all particle foot values by a constant scales the output by as much.

    This is the foundational property of Bayesian inversion: concentration =
    integral(footprint * flux). If the footprint is not linear in the particle
    foot values, that integral is invalid.  All operations in calc_footprint
    are linear in foot (bincount, Gaussian convolution, division by n_particles),
    so the output must scale exactly.
    """
    particles = _interior_particles()
    foot_1x = _interior_footprint(particles, point_receptor, smooth_factor=1.0)

    particles_2x = particles.copy()
    particles_2x["foot"] = particles_2x["foot"] * 2.0
    foot_2x = _interior_footprint(particles_2x, point_receptor, smooth_factor=1.0)

    np.testing.assert_allclose(
        foot_2x.values,
        foot_1x.values * 2.0,
        rtol=1e-10,
        err_msg="Footprint must scale linearly with particle foot values",
    )


def test_calc_footprint_total_equals_normalized_input_sum_at_zero_smooth(
    point_receptor,
):
    """
    With smooth_factor=0, total footprint = sum(in-domain foot) / n_particles.

    STILT normalizes by particle count so that the footprint is intensive (per
    particle).  The 1×1 identity kernel does not move any sensitivity between
    cells, so the grid sum must exactly equal the un-normalized particle sum
    divided by the ensemble size.
    """
    particles = _interior_particles()
    n = particles["particle"].nunique()
    foot = _interior_footprint(particles, point_receptor, smooth_factor=0.0)

    expected = float(particles["foot"].sum()) / n
    assert float(foot.values.sum()) == pytest.approx(expected, rel=1e-10)


def test_calc_footprint_gaussian_smoothing_preserves_total_sensitivity(point_receptor):
    """
    Gaussian smoothing does not create or destroy total footprint sensitivity.

    The kernel sums to 1 (verified separately in test_make_gauss_kernel_normalized)
    and particles are placed well inside the domain so the Gaussian tails do not
    spill outside the grid boundary.  Any loss of total sensitivity from smoothing
    would silently bias flux inversion toward underestimating emissions.
    """
    particles = _interior_particles()
    foot_0 = _interior_footprint(particles, point_receptor, smooth_factor=0.0)
    foot_s = _interior_footprint(particles, point_receptor, smooth_factor=1.0)

    total_0 = float(foot_0.values.sum())
    total_s = float(foot_s.values.sum())

    assert total_s == pytest.approx(total_0, rel=1e-5), (
        f"Smoothing changed total footprint: {total_0:.6g} → {total_s:.6g} "
        f"(Δ = {abs(total_s - total_0) / total_0:.2e})"
    )


def test_calc_footprint_reproducible(point_receptor):
    """
    Calling calc_footprint twice with identical inputs returns identical arrays.

    Statefulness bugs (e.g. a mutable module-level cache that accumulates across
    calls) would cause different runs of the same simulation to diverge silently.
    """
    particles = _interior_particles()
    foot1 = _interior_footprint(particles, point_receptor, smooth_factor=1.0)
    foot2 = _interior_footprint(particles, point_receptor, smooth_factor=1.0)

    np.testing.assert_array_equal(
        foot1.values,
        foot2.values,
        err_msg="calc_footprint must be deterministic — identical inputs must produce identical outputs",
    )


def test_calc_footprint_time_integrate_equals_sum_of_time_slices(point_receptor):
    """
    time_integrate=True must equal summing the per-time-step footprint.

    If the collapsed footprint were computed differently from the sum of slices,
    daily-average footprints used in Bayesian inversion would silently differ from
    the sum of the hourly footprints researchers expect.
    """
    particles = _particles_in_domain()
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)

    foot_ti = calc_footprint(particles, point_receptor, grid, time_integrate=True)
    foot_no = calc_footprint(particles, point_receptor, grid, time_integrate=False)

    np.testing.assert_allclose(
        foot_ti.values.squeeze(),
        foot_no.sum("time").values,
        rtol=1e-10,
        err_msg="time_integrate=True must equal the sum over all individual time slices",
    )


def test_calc_footprint_smooth_zero_assigns_exact_cells(point_receptor):
    """
    With smooth_factor=0, each particle's foot goes entirely into its own cell.

    No neighbouring cells receive any spillover. This tests the 1×1 identity kernel path (sigma=0 → _make_gauss_kernel returns
    [[1.0]]).  A bug here would mean that the ``permute.f90``-equivalent scatter
    operation distributes sensitivity to wrong cells, corrupting the spatial pattern
    of all no-smooth footprints.
    """
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)

    # 10 identical particles exactly at the centre of cell (start=-113.9, centre=-113.85)
    n = 10
    foot_val = 1e-4
    particles = pd.DataFrame(
        {
            "time": [-60.0] * n,
            "particle": [float(i + 1) for i in range(n)],
            "lon": [-113.85] * n,
            "lat": [39.05] * n,
            "zagl": [5.0] * n,
            "foot": [foot_val] * n,
        }
    )

    foot = calc_footprint(particles, point_receptor, grid, smooth_factor=0.0)

    # total = sum(foot) / n_particles = (n * foot_val) / n = foot_val
    assert float(foot.values.sum()) == pytest.approx(foot_val, rel=1e-10)

    # Exactly one non-zero cell across all time layers
    nonzero_count = int((foot.values > 0).sum())
    assert nonzero_count == 1, (
        f"smooth_factor=0: expected exactly 1 non-zero cell, got {nonzero_count}"
    )


def test_concentration_reconstruction_from_known_footprint(point_receptor):
    """
    ``c = Σ foot[i,j] * q[i,j]`` recovers the analytically expected concentration.

    This is the fundamental identity that Bayesian flux inversion relies on:
    a receptor concentration enhancement equals the dot product of the footprint
    sensitivity matrix with the surface flux field.  If this identity is broken
    — by a normalization error, wrong cell assignment, or unit mismatch — every
    inferred emission estimate is wrong, silently.

    Setup (smooth_factor=0 so values are exact, no Gaussian spread):
      - Cluster A: 10 particles at cell centre (-113.85°, 39.05°), foot = 2e-4
      - Cluster B: 10 particles at cell centre (-113.35°, 39.55°), foot = 3e-4
      - 20 total unique particles (n_particles = 20)

    Analytical footprint values:
      F_A = n_A * foot_A / n_particles = 10 * 2e-4 / 20 = 1e-4  ppm/(μmol m⁻² s⁻¹)
      F_B = n_B * foot_B / n_particles = 10 * 3e-4 / 20 = 1.5e-4 ppm/(μmol m⁻² s⁻¹)

    Applied flux field (non-zero only at the two cluster cells):
      q_A = 5.0  μmol m⁻² s⁻¹   (roughly a moderate CH₄ surface source)
      q_B = 8.0  μmol m⁻² s⁻¹

    Expected concentration:
      c = F_A * q_A + F_B * q_B = 1e-4 * 5 + 1.5e-4 * 8 = 1.7e-3 ppm ≈ 1.7 ppb
    """
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)

    n_a, n_b = 10, 10
    n_total = n_a + n_b
    foot_a, foot_b = 2e-4, 3e-4

    particles = pd.DataFrame(
        {
            "time": [-60.0] * n_total,
            "particle": [float(i + 1) for i in range(n_total)],
            "lon": [-113.85] * n_a + [-113.35] * n_b,
            "lat": [39.05] * n_a + [39.55] * n_b,
            "zagl": [5.0] * n_total,
            "foot": [foot_a] * n_a + [foot_b] * n_b,
        }
    )

    foot = calc_footprint(particles, point_receptor, grid, smooth_factor=0.0)

    # Verify the footprint cell values are exactly what the formula predicts.
    expected_fa = n_a * foot_a / n_total  # 1e-4
    expected_fb = n_b * foot_b / n_total  # 1.5e-4

    f_a = float(foot.sel(lon=-113.85, lat=39.05, method="nearest").sum())
    f_b = float(foot.sel(lon=-113.35, lat=39.55, method="nearest").sum())

    assert f_a == pytest.approx(expected_fa, rel=1e-10), (
        f"Cell A footprint: expected {expected_fa:.3e}, got {f_a:.3e}"
    )
    assert f_b == pytest.approx(expected_fb, rel=1e-10), (
        f"Cell B footprint: expected {expected_fb:.3e}, got {f_b:.3e}"
    )

    # Apply flux field (non-zero at the two cluster cells only).
    q_a, q_b = 5.0, 8.0
    flux = np.zeros_like(foot.values)

    lons = foot.lon.values
    lats = foot.lat.values
    lon_a = int(np.argmin(np.abs(lons - (-113.85))))
    lat_a = int(np.argmin(np.abs(lats - 39.05)))
    lon_b = int(np.argmin(np.abs(lons - (-113.35))))
    lat_b = int(np.argmin(np.abs(lats - 39.55)))

    flux[:, lat_a, lon_a] = q_a
    flux[:, lat_b, lon_b] = q_b

    # c = F_A * q_A + F_B * q_B
    c_computed = float((foot.values * flux).sum())
    c_expected = expected_fa * q_a + expected_fb * q_b  # 1.7e-3 ppm

    assert c_computed == pytest.approx(c_expected, rel=1e-10), (
        f"Concentration reconstruction: c = {c_computed:.4g} ppm, "
        f"expected {c_expected:.4g} ppm  (~{c_expected * 1e3:.2f} ppb)"
    )


# ---------------------------------------------------------------------------
# Python-only helper tests: branch coverage for paths the live-R fidelity
# tests cannot reliably target. These run in CI on every Python version
# without needing R, Rscript, or HYSPLIT.

# ---------------------------------------------------------------------------


def test_wrap_antimeridian_longitudes_global_branch():
    """``xdist == 0`` (global 360° grid) anchors to [-180, 180] without wrapping."""
    p = pd.DataFrame({"lon": [-179.0, 0.0, 179.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=-180.0, xmax=180.0)
    assert xmin == -180.0
    assert xmax == 180.0
    assert wrapped is False
    # Particle longitudes must be unchanged in the global branch.
    np.testing.assert_array_equal(out["lon"].values, p["lon"].values)


def test_wrap_antimeridian_longitudes_crossing_branch():
    """``xmax < xmin`` (dateline crossing) rotates longitudes into [0, 360)."""
    p = pd.DataFrame({"lon": [179.0, -179.0, 170.0, -170.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=170.0, xmax=-170.0)
    assert wrapped is True
    # Bounds wrap to 170, 190 in [0, 360) space.
    assert xmin == pytest.approx(170.0)
    assert xmax == pytest.approx(190.0)
    expected = np.array([179.0, 181.0, 170.0, 190.0])
    np.testing.assert_allclose(out["lon"].values, expected)


def test_wrap_antimeridian_longitudes_partial_wrap_branch():
    """``xmax > 180`` (partial wrap, e.g. xmin=170, xmax=200) also rotates."""
    p = pd.DataFrame({"lon": [175.0, -175.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=170.0, xmax=200.0)
    assert wrapped is True
    assert xmin == pytest.approx(170.0)
    assert xmax == pytest.approx(200.0)
    np.testing.assert_allclose(out["lon"].values, np.array([175.0, 185.0]))


def test_wrap_antimeridian_longitudes_no_wrap_branch():
    """Standard CONUS domain (xmin=-113, xmax=-111) returns particles unchanged."""
    p = pd.DataFrame({"lon": [-112.5, -111.5]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(
        p, xmin=-113.0, xmax=-111.0
    )
    assert wrapped is False
    assert xmin == -113.0
    assert xmax == -111.0
    np.testing.assert_array_equal(out["lon"].values, p["lon"].values)


def test_project_particles_to_crs_raises_on_missing_pyproj(monkeypatch):
    """Non-longlat path surfaces a clear ImportError when pyproj is absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("pyproj missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = pd.DataFrame({"lon": [-112.0], "lat": [40.0]})
    with pytest.raises(ImportError, match="pyproj"):
        _project_particles_to_crs(
            p,
            crs="+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs",
            xmin=-113.0,
            xmax=-111.0,
            ymin=39.0,
            ymax=41.0,
        )


def test_project_particles_to_crs_rejects_invalid_proj_string():
    """An unparseable proj4 string surfaces from pyproj as a parse error."""
    from pyproj.exceptions import CRSError

    p = pd.DataFrame({"lon": [-112.0], "lat": [40.0]})
    with pytest.raises(CRSError):
        _project_particles_to_crs(
            p,
            crs="+proj=nope-not-a-projection",
            xmin=-113.0,
            xmax=-111.0,
            ymin=39.0,
            ymax=41.0,
        )


def test_compute_kernel_bandwidths_single_particle_returns_zero_sigma():
    """
    A one-particle trajectory produces a valid (degenerate) footprint.

    A single particle has var(long)=var(lati)=NaN, which R's na.omit() drops.
    PYSTILT's helper returns w=0 (identity kernel) instead of crashing.
    """
    p = pd.DataFrame(
        {
            "particle": [1.0, 1.0],
            "rtime": [-1.0, -2.0],
            "time": [-1.0, -2.0],
            "lon": [-112.0, -112.0],
            "lat": [40.5, 40.5],
            "foot": [1e-3, 1e-3],
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, is_longlat=True)
    np.testing.assert_array_equal(w, np.zeros_like(w))
    assert len(kernel_df) == len(w)


def test_compute_kernel_bandwidths_two_coincident_particles_returns_zero_sigma():
    """Two particles at identical positions have varsum=0 ⇒ w=0."""
    p = pd.DataFrame(
        {
            "particle": [1.0, 1.0, 2.0, 2.0],
            "rtime": [-1.0, -2.0, -1.0, -2.0],
            "time": [-1.0, -2.0, -1.0, -2.0],
            "lon": [-112.0] * 4,
            "lat": [40.5] * 4,
            "foot": [1e-3] * 4,
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, is_longlat=True)
    np.testing.assert_array_equal(w, np.zeros_like(w))


def test_calc_footprint_returns_a_named_dataarray_with_its_receptor(point_receptor):
    particles = _particles_in_domain()
    foot = calc_footprint(particles, point_receptor, _grid(), name="hrrr")
    assert isinstance(foot, xr.DataArray)
    assert foot.name == "foot"
    assert foot["receptor"].item() == str(point_receptor.id)
    assert foot.stilt.receptor == point_receptor
    assert foot.stilt.name == "hrrr"
    assert foot.attrs["units"] == "ppm m2 s umol-1"


def test_one_layer_footprint_is_stamped_at_its_hour():
    """A backward footprint with only layer -1 is stamped an hour before the receptor, as in STILT-R."""
    receptor = PointReceptor(
        time="2023-01-01 12:00", longitude=-111.95, latitude=40.75, altitude=5.0
    )
    grid = Grid(xmin=-112.3, xmax=-111.6, ymin=40.5, ymax=41.0, xres=0.01, yres=0.01)

    foot = calc_footprint(_first_hour_particles(), receptor, grid)
    integrated = calc_footprint(
        _first_hour_particles(), receptor, grid, time_integrate=True
    )

    assert list(pd.DatetimeIndex(foot["time"].values)) == [
        pd.Timestamp("2023-01-01 11:00")
    ]
    assert list(pd.DatetimeIndex(integrated["time"].values)) == [
        pd.Timestamp("2023-01-01 12:00")
    ]


def test_calculated_coordinates_match_the_grid_axes():
    """A calculated footprint has the coordinates a stored one is read back with."""
    receptor = PointReceptor(
        time="2023-01-01 12:00", longitude=-111.95, latitude=40.75, altitude=5.0
    )
    grid = Grid(xmin=-112.3, xmax=-111.6, ymin=40.5, ymax=41.0, xres=0.01, yres=0.01)

    foot = calc_footprint(_first_hour_particles(), receptor, grid)

    x, y = grid.axes
    np.testing.assert_array_equal(foot["lon"].values, x)
    np.testing.assert_array_equal(foot["lat"].values, y)
