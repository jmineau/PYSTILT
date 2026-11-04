"""Tests for stilt.footprint.gridding: calc_footprint, its kernels, and its invariants."""

import builtins
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.exceptions import EmptyFootprint
from stilt.footprint import calc_footprint, gridding
from stilt.footprint.gridding import (
    _compute_kernel_bandwidths,
    _fold_padding_around_the_globe,
    _grid_units_per_degree_longitude,
    _interpolate_early_timesteps,
    _interpolate_longitudes,
    _interpolation_times,
    _make_gauss_kernel,
    _onto_padded_axis,
    _project_particles_to_crs,
    _unwrap_around_cloud,
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
            "age": times,
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
            "age": [-60.0] * n,
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
            "age": np.repeat([-20.0, -40.0, -59.0], n),
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


@pytest.mark.parametrize("max_pairs", [1, 50, 2**20])
def test_stamp_kernel_equals_convolution_at_the_edges(monkeypatch, max_pairs):
    """
    Stamping the kernel at each cell equals a convolution with zeros beyond the grid.

    Cells sit on and next to every edge, so kernel mass falls outside the
    grid and must be dropped, as ``scipy.ndimage.convolve(mode="constant")``
    drops it. The kernel is asymmetric, so a flipped kernel would fail, and
    small ``_MAX_STAMP_PAIRS`` budgets split the cells into chunks.
    """
    from scipy.ndimage import convolve

    from stilt.footprint import gridding

    monkeypatch.setattr(gridding, "_MAX_STAMP_PAIRS", max_pairs)
    rng = np.random.default_rng(0)
    shape = (12, 9)
    kernel = rng.uniform(size=(5, 7))
    rows = np.array([0, 0, 11, 11, 5, 1, 10, 6, 3])
    cols = np.array([0, 8, 0, 8, 4, 2, 7, 0, 8])
    values = rng.uniform(size=len(rows))

    out = np.zeros(shape)
    gridding._stamp_kernel(out, rows, cols, values, kernel)

    dense = np.zeros(shape)
    dense[rows, cols] = values
    expected = convolve(dense, kernel, mode="constant", cval=0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)


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


def test_interpolation_times_are_built_as_r_builds_them():
    """R's seq(0, 10, by = 0.1) is 0 + (0:100) * 0.1, which is not i / 10 (#288)."""
    times = _interpolation_times(1)

    assert times[29] == 29 * 0.1  # 2.9000000000000004, as in R
    assert times[29] != 2.9
    assert times[100] == 10.0 and times[-1] == 100.0


def test_calc_footprint_interpolates_a_full_particle_table():
    """
    Columns beyond the footprint's own do not stop the early-time interpolation (#288).

    One particle moves 0.05 degrees a minute east on a 0.01-degree grid, so
    its output steps land in every fifth cell. Interpolated to 0.1-minute
    steps it crosses each cell between them. STILT-R's simulation_step passes
    calc_footprint only time, indx, long, lati and foot, so the inserted rows
    are kept there; with zagl or any other column they used to be dropped.
    """
    age = -np.arange(1.0, 121.0)
    particles = pd.DataFrame(
        {
            "particle": 1.0,
            "age": age,
            "lon": -112.5 + 0.05 * np.abs(age),
            "lat": 40.505,
            "zagl": 5.0,
            "mlht": 500.0,
            "foot": 1e-3,
        }
    )
    receptor = PointReceptor(
        time="2023-07-01 12:00", longitude=-112.5, latitude=40.505, altitude=5.0
    )
    grid = Grid(xmin=-113.0, xmax=-106.0, ymin=40.0, ymax=41.0, xres=0.01, yres=0.01)

    foot = calc_footprint(particles, receptor, grid, smooth_factor=0.0)
    row = foot.sum("time").sel(lat=40.505, method="nearest")

    # Every cell the particle crosses in its first ten minutes has influence,
    # not only the cells of its whole-minute steps.
    crossed = row.sel(lon=slice(-112.45, -112.0)).to_numpy()
    assert (crossed > 0).all()
    assert float(foot.sum()) == pytest.approx(float(particles["foot"].sum()))


def test_interpolate_early_timesteps_preserves_window_foot_sums():
    particles = pd.DataFrame(
        {
            "age": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
            "particle": [1, 1, 1, 2, 2, 2],
            "lon": [-113.0, -114.0, -115.0, -112.0, -113.5, -115.0],
            "lat": [39.0, 40.0, 41.0, 39.5, 40.5, 41.5],
            "foot": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
        }
    )
    original_atime = np.abs(particles["age"])
    original_sums = [
        particles.loc[original_atime <= 10, "foot"].sum(),
        particles.loc[(original_atime > 10) & (original_atime <= 20), "foot"].sum(),
        particles.loc[(original_atime > 20) & (original_atime <= 100), "foot"].sum(),
    ]

    interpolated = _interpolate_early_timesteps(
        particles, xres=0.01, yres=0.01, time_sign=-1
    )

    assert len(interpolated) > len(particles)
    atime = np.abs(interpolated["age"])
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
            "age": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
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
        ["particle", "age"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(interpolated, expected, check_dtype=False)


def _northward_particles(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """
    Particles leaving (-112, 40.5) north at about 7 m/s for 100 minutes (#307).

    Each 1-minute step is about 0.005 degrees of latitude, 555 m, and almost
    nothing east to west.
    """
    rng = np.random.default_rng(seed)
    ages = np.arange(-1.0, -101.0, -1.0)
    frames = []
    for k in range(n):
        speed = 0.005 * (1 + 0.1 * rng.normal())
        frames.append(
            pd.DataFrame(
                {
                    "particle": k + 1,
                    "age": ages,
                    "lon": -112.0 + 0.0005 * rng.normal(),
                    "lat": 40.5 + speed * np.abs(ages),
                    "foot": 1e-3,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _interpolates(p: pd.DataFrame, **grid) -> bool:
    """Whether _interpolate_early_timesteps adds rows to *p* on this grid."""
    return len(_interpolate_early_timesteps(p, time_sign=-1, **grid)) > len(p)


def test_projected_grid_interpolates_steps_longer_than_a_cell():
    """
    On 100 m UTM cells, 555 m steps are interpolated, as on 0.001 degree cells (#307).

    The step used to be measured in degrees and compared with ``xres`` in
    meters, so a grid in meters never interpolated.
    """
    pytest.importorskip("pyproj")
    p = _northward_particles()
    longlat = _interpolate_early_timesteps(p, xres=0.001, yres=0.001, time_sign=-1)
    utm = _interpolate_early_timesteps(
        p, xres=100.0, yres=100.0, time_sign=-1, crs=_UTM12
    )

    assert len(longlat) > len(p)
    pd.testing.assert_frame_equal(utm, longlat)


def test_projected_grid_does_not_interpolate_steps_shorter_than_a_cell():
    """On 1 km UTM cells the same 555 m steps are left as they are."""
    pytest.importorskip("pyproj")
    p = _northward_particles()
    assert not _interpolates(p, xres=1000.0, yres=1000.0, crs=_UTM12)
    assert not _interpolates(p, xres=0.01, yres=0.01)


def test_projected_step_is_measured_in_the_grid_units():
    """A UTM grid in kilometers decides as the same grid in meters."""
    pytest.importorskip("pyproj")
    p = _northward_particles()
    km = _UTM12.replace("+units=m", "+units=km")
    assert _interpolates(p, xres=0.1, yres=0.1, crs=km)
    assert not _interpolates(p, xres=1.0, yres=1.0, crs=km)


def test_projected_step_is_compared_with_the_cell_along_each_axis():
    """
    A north step is compared with ``yres``, an east step with ``xres``.

    The particles move 555 m north and a few meters east per step, so cells
    500 m tall interpolate and cells 500 m wide but 1 km tall do not.
    """
    pytest.importorskip("pyproj")
    p = _northward_particles()
    assert _interpolates(p, xres=1000.0, yres=500.0, crs=_UTM12)
    assert not _interpolates(p, xres=500.0, yres=1000.0, crs=_UTM12)


def test_projected_step_follows_a_turned_grid():
    """
    On a grid turned by 45 degrees, a 555 m north step is 393 m along each axis.

    So 450 m cells interpolate on the unturned oblique Mercator grid and not
    on the turned one: the step is measured along the grid's own axes.
    """
    pytest.importorskip("pyproj")
    p = _northward_particles()
    omerc = "+proj=omerc +lat_0=40.5 +lonc=-112 +gamma=0 +k_0=1 +datum=WGS84 +units=m"
    assert _interpolates(p, xres=450.0, yres=450.0, crs=f"{omerc} +alpha=0.0001")
    assert not _interpolates(p, xres=450.0, yres=450.0, crs=f"{omerc} +alpha=45")
    assert _interpolates(p, xres=350.0, yres=350.0, crs=f"{omerc} +alpha=45")


def test_projected_footprint_fills_the_cells_between_steps_like_the_longlat_one(
    point_receptor,
):
    """
    On 100 m UTM cells the footprint covers about as many cells as on 0.001 degree cells.

    Without the interpolation the UTM footprint was the particles' 1-minute
    positions, a few cells apart: a fifth fewer cells than the lon/lat one
    (#307). The 0.001 degree cells are 85 m by 111 m here, so the two grids
    do not have the same cells; both keep the same total.
    """
    pytest.importorskip("pyproj")
    p = _northward_particles().assign(zagl=5.0)
    bounds = {"xmin": -112.05, "xmax": -111.95, "ymin": 40.45, "ymax": 41.05}
    feet = {
        name: calc_footprint(
            p, point_receptor, grid, time_integrate=True, smooth_factor=0.0
        ).to_numpy()
        for name, grid in {
            "longlat": Grid(**bounds, xres=0.001, yres=0.001),
            "utm": Grid(**bounds, xres=100.0, yres=100.0, crs=_UTM12),
        }.items()
    }
    cells = {name: np.count_nonzero(np.nan_to_num(a)) for name, a in feet.items()}

    assert cells["utm"] > 0.9 * cells["longlat"]
    np.testing.assert_allclose(
        np.nansum(feet["utm"]), np.nansum(feet["longlat"]), rtol=1e-3
    )


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
            "age": [-60.0] * n,
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
            "age": [-60.0] * n_total,
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
            "age": [-1.0, -2.0],
            "lon": [-112.0, -112.0],
            "lat": [40.5, 40.5],
            "foot": [1e-3, 1e-3],
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0)
    np.testing.assert_array_equal(w, np.zeros_like(w))
    assert len(kernel_df) == len(w)


def test_compute_kernel_bandwidths_two_coincident_particles_returns_zero_sigma():
    """Two particles at identical positions have varsum=0 ⇒ w=0."""
    p = pd.DataFrame(
        {
            "particle": [1.0, 1.0, 2.0, 2.0],
            "rtime": [-1.0, -2.0, -1.0, -2.0],
            "age": [-1.0, -2.0, -1.0, -2.0],
            "lon": [-112.0] * 4,
            "lat": [40.5] * 4,
            "foot": [1e-3] * 4,
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0)
    np.testing.assert_array_equal(w, np.zeros_like(w))


_UTM12 = "+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs"


def _spread_particles(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """
    Particles at (-112, 40.5) at release and 0.03 degrees apart a day later.

    A day after release the kernel is about 2 km wide, two cells of 1 km.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "particle": np.tile(np.arange(1, n + 1), 2),
            "age": np.repeat([-1.0, -1441.0], n),
            "lon": np.concatenate([np.full(n, -112.0), rng.normal(-112.0, 0.03, n)]),
            "lat": np.concatenate([np.full(n, 40.5), rng.normal(40.5, 0.03, n)]),
            "zagl": np.full(2 * n, 5.0),
            "foot": np.repeat([0.0, 1e-3], n),
        }
    )


def _with_rtime(p: pd.DataFrame) -> pd.DataFrame:
    """Add ``rtime`` as calc_footprint does for a backward run."""
    p = p.copy()
    p["rtime"] = p["age"] + p["age"].abs().groupby(p["particle"]).transform("min")
    return p


def test_projected_kernel_is_as_wide_east_to_west_as_the_longlat_kernel():
    """
    On a projected grid the kernel width is the longitude/latitude grid's, converted to meters.

    The longitude/latitude width is in degrees of longitude, so in meters it
    is that times the length of a degree of longitude along the parallel (the
    geodesic on WGS84). The UTM width may differ from it only by UTM's scale
    factor, within 0.1 % in a zone. STILT-R's formula on the projected
    coordinates gives a kernel tens of meters wide here.
    """
    pyproj = pytest.importorskip("pyproj")
    p = _with_rtime(_spread_particles())
    kernel_ll, w_ll = _compute_kernel_bandwidths(p, smooth_factor=1.0)
    _, w_utm = _compute_kernel_bandwidths(p, smooth_factor=1.0, crs=_UTM12)

    lon = kernel_ll["lon_mean"].to_numpy()
    lat = kernel_ll["lat_mean"].to_numpy()
    _, _, meters_per_degree = pyproj.Geod(ellps="WGS84").inv(
        lon - 0.5, lat, lon + 0.5, lat
    )
    assert w_utm.max() > 1000  # wider than a cell of 1 km
    np.testing.assert_allclose(w_utm, w_ll * meters_per_degree, rtol=1e-3)


def test_projected_kernel_width_follows_the_grid_units():
    """The same particles get a kernel 1000 times smaller in kilometers than in meters."""
    pytest.importorskip("pyproj")
    p = _with_rtime(_spread_particles())
    _, w_m = _compute_kernel_bandwidths(p, smooth_factor=1.0, crs=_UTM12)
    _, w_km = _compute_kernel_bandwidths(
        p, smooth_factor=1.0, crs=_UTM12.replace("+units=m", "+units=km")
    )
    np.testing.assert_allclose(w_km * 1000, w_m, rtol=1e-9)


def test_projected_kernel_width_does_not_depend_on_how_the_axes_are_turned():
    """An oblique Mercator grid turned by 45 degrees gets the kernel of the unturned one."""
    pytest.importorskip("pyproj")
    p = _with_rtime(_spread_particles())
    omerc = "+proj=omerc +lat_0=40.5 +lonc=-112 +gamma=0 +k_0=1 +datum=WGS84 +units=m"
    _, w_north = _compute_kernel_bandwidths(
        p, smooth_factor=1.0, crs=f"{omerc} +alpha=0.0001"
    )
    _, w_turned = _compute_kernel_bandwidths(
        p, smooth_factor=1.0, crs=f"{omerc} +alpha=45"
    )
    np.testing.assert_allclose(w_turned, w_north, rtol=1e-6)


def _cloud_at(lon0: float, lat0: float = 50.0) -> pd.DataFrame:
    """
    ``_spread_particles`` moved to (lon0, lat0), 0.3 degrees apart a day later.

    Longitudes are written in [-180, 180), as HYSPLIT writes them, so a
    cloud at 179.9 has particles on both sides of 180.
    """
    p = _spread_particles()
    rng = np.random.default_rng(11)
    late = p["age"] < -1
    p.loc[late, "lon"] = rng.normal(0.0, 0.3, int(late.sum()))
    p.loc[late, "lat"] = rng.normal(0.0, 0.3, int(late.sum()))
    p.loc[~late, ["lon", "lat"]] = 0.0
    p["lon"] = ((p["lon"] + lon0 + 180.0) % 360.0) - 180.0
    p["lat"] = p["lat"] + lat0
    return p


def _lambert_at(lon0: float, lat0: float = 50.0) -> str:
    return f"+proj=lcc +lat_1=45 +lat_2=55 +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"


def test_projected_kernel_of_a_cloud_across_180_degrees_is_the_cloud_at_0_degrees():
    """
    A cloud on both sides of 180 degrees gets the kernel of the same cloud at 0 degrees.

    Each is on a Lambert grid centered on it, so the two are the same on the
    ground. Without unwrapping, the longitudes near +180 and -180 give a
    variance of thousands of square degrees and a kernel about 20 times too
    wide.
    """
    pytest.importorskip("pyproj")
    across = _with_rtime(_cloud_at(179.9))
    assert across["lon"].min() < -179 and across["lon"].max() > 179
    at_zero = _with_rtime(_cloud_at(0.0))
    _, w_across = _compute_kernel_bandwidths(
        across, smooth_factor=1.0, crs=_lambert_at(179.9)
    )
    _, w_zero = _compute_kernel_bandwidths(
        at_zero, smooth_factor=1.0, crs=_lambert_at(0.0)
    )
    assert w_zero.max() > 3000  # a kernel of a few km, wider than a cell
    np.testing.assert_allclose(w_across, w_zero, rtol=1e-6)


def test_projected_footprint_of_a_cloud_across_180_degrees_keeps_its_total(
    point_receptor,
):
    """On a 3 km Lambert grid across 180 degrees the footprint is the one at 0 degrees."""
    pytest.importorskip("pyproj")
    feet = []
    for lon0, west, east in ((179.9, 177.9, -178.1), (0.0, -2.0, 2.0)):
        grid = Grid(
            xmin=west,
            xmax=east,
            ymin=48.0,
            ymax=52.0,
            xres=3000,
            yres=3000,
            crs=_lambert_at(lon0),
        )
        feet.append(
            calc_footprint(_cloud_at(lon0), point_receptor, grid, time_integrate=True)
        )
    across, at_zero = (f.to_numpy() for f in feet)
    np.testing.assert_allclose(across.sum(), at_zero.sum(), rtol=1e-6)
    np.testing.assert_allclose(
        np.count_nonzero(across), np.count_nonzero(at_zero), rtol=0.02
    )


#: A Lambert conformal grid centered on a receptor in Utah. Its scale depends
#: on latitude only, and it is cut at lon_0 + 180, the receptor's antipode.
_UTAH = (-111.85, 40.77)
_LAMBERT_UTAH = (
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40.77 +lon_0=-111.85 +datum=WGS84 +units=m"
)


def _drifting_cloud(n_steps: int, deg_per_hour: float) -> pd.DataFrame:
    """300 particles leaving Utah, spreading, and drifting west by *deg_per_hour*."""
    rng = np.random.default_rng(5)
    n = 300
    dlon, dlat = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    rows = []
    for j in range(n_steps):
        spread = 0.02 + 0.3 * j / max(n_steps - 1, 1)
        lon = _UTAH[0] - deg_per_hour * j + spread * dlon
        rows.append(
            pd.DataFrame(
                {
                    "particle": np.arange(1, n + 1),
                    "age": -1.0 - 60.0 * j,
                    "lon": ((lon + 180.0) % 360.0) - 180.0,
                    "lat": _UTAH[1] + spread * dlat,
                    "zagl": 5.0,
                    "foot": 1e-3,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _far_cloud(lon0: float) -> pd.DataFrame:
    """Particles released in Utah and, ten days back, a cloud at (lon0, 45 N)."""
    p = _cloud_at(lon0, lat0=45.0)
    p.loc[p["age"] > -2, ["lon", "lat"]] = _UTAH
    p.loc[p["age"] < -2, "age"] = -14401.0
    return p


def test_projected_kernel_has_no_seam_at_the_release_antipode():
    """
    Ten days back, a cloud on the far side of the globe gets the same kernel wherever it is.

    68.15 E is the antipode of the Utah release. Longitudes taken within 180
    degrees of the release split a cloud there in two, and its kernel came
    out over 20 times too wide.
    """
    pytest.importorskip("pyproj")
    w = {}
    for lon0 in (60.0, 68.15, 68.3):
        _, widths = _compute_kernel_bandwidths(
            _with_rtime(_far_cloud(lon0)), smooth_factor=1.0, crs=_LAMBERT_UTAH
        )
        w[lon0] = widths.max()
    assert w[60.0] > 5000
    np.testing.assert_allclose([w[68.15], w[68.3]], w[60.0], rtol=1e-6)


def test_projected_kernel_of_a_cloud_circling_the_globe_is_the_standing_cloud():
    """
    A cloud drifting 360 degrees west in ten days gets, at each step, the kernel of the same cloud standing still.

    On its way it crosses 180 degrees, the release's antipode, and the
    Lambert grid's cut, and its kernel must not jump at any of them.
    """
    pytest.importorskip("pyproj")
    _, circling = _compute_kernel_bandwidths(
        _with_rtime(_drifting_cloud(241, 1.5)), smooth_factor=1.0, crs=_LAMBERT_UTAH
    )
    _, standing = _compute_kernel_bandwidths(
        _with_rtime(_drifting_cloud(241, 0.0)), smooth_factor=1.0, crs=_LAMBERT_UTAH
    )
    np.testing.assert_allclose(circling, standing, rtol=1e-6, atol=1e-9)


def test_grid_units_per_degree_longitude_ignores_a_projection_cut():
    """At the Lambert grid's cut (lon_0 + 180) a degree is as long as one degree away."""
    pytest.importorskip("pyproj")
    lon = np.array([67.15, 68.145, 68.15, 68.155, 69.15])
    lengths = _grid_units_per_degree_longitude(
        _LAMBERT_UTAH, lon=lon, lat=np.full(lon.size, 45.0)
    )
    np.testing.assert_allclose(lengths, lengths[0], rtol=1e-6)


def test_unwrap_around_cloud_keeps_a_cloud_away_from_any_seam_bit_for_bit():
    """Longitudes already within 180 degrees of their step's mean come back unchanged."""
    p = _with_rtime(_drifting_cloud(5, 0.0))
    out = _unwrap_around_cloud(p)
    assert out.to_numpy().tobytes() == p["lon"].to_numpy(dtype=float).tobytes()


def test_unwrap_around_cloud_joins_a_cloud_across_180_degrees():
    """A cloud with longitudes near +180 and -180 comes back on one side, as one cloud."""
    p = _with_rtime(_cloud_at(179.9))
    late = p["age"] < -1
    out = _unwrap_around_cloud(p)[late]
    assert out.max() - out.min() < 5
    np.testing.assert_allclose(out.mean(), 179.9, atol=0.1)


_UTM48 = "EPSG:32648"
_HIMAWARI = "+proj=geos +h=35785831 +lon_0=140.7 +sweep=y +datum=WGS84 +units=m"


def _tropical_run(far_lon: float = 20.0) -> pd.DataFrame:
    """
    Particles released at Singapore (103.85 E, 1.35 N), and ten days back a cloud at (far_lon, 2 N).

    At 20 E the far cloud is 85 degrees from UTM 48N's meridian, where the
    transverse Mercator has no value, and off Himawari's disk.
    """
    rng = np.random.default_rng(3)
    n = 500

    def step(age, lon, lat, sd, foot):
        return pd.DataFrame(
            {
                "particle": np.arange(1, n + 1),
                "age": age,
                "lon": rng.normal(lon, sd, n),
                "lat": rng.normal(lat, sd, n),
                "zagl": 10.0,
                "foot": foot,
            }
        )

    return pd.concat(
        [
            step(-1.0, 103.85, 1.35, 0.001, 1e-3),
            step(-600.0, 103.0, 1.6, 0.3, 1e-3),
            step(-14400.0, far_lon, 2.0, 1.0, 1e-9),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize("crs", [_UTM48, _HIMAWARI], ids=["utm48n", "himawari"])
def test_projected_footprint_is_made_when_a_far_step_is_where_the_projection_has_no_value(
    crs,
):
    """
    A step centered where the projection has no value is left out; the rest makes the footprint.

    Its particles cannot reach the grid, so the footprint is the one made
    without them. Before, that step's width was NaN and calc_footprint raised.
    """
    pytest.importorskip("pyproj")
    p = _tropical_run()
    receptor = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12), longitude=103.85, latitude=1.35, altitude=5.0
    )
    grid = Grid(
        xmin=103.3, xmax=104.4, ymin=1.0, ymax=1.8, xres=1000, yres=1000, crs=crs
    )
    foot = calc_footprint(p, receptor, grid, time_integrate=True)
    without_far = calc_footprint(
        p[p["age"] > -14400], receptor, grid, time_integrate=True
    )
    assert float(foot.sum()) > 0
    np.testing.assert_array_equal(foot.to_numpy(), without_far.to_numpy())


def test_projected_kernel_leaves_out_only_the_steps_with_no_width():
    """Of three steps, the one at 20 E on UTM 48N is left out and the others keep their widths."""
    pytest.importorskip("pyproj")
    p = _with_rtime(_tropical_run())
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, crs=_UTM48)
    near = _with_rtime(p[p["age"] > -14400])
    near_df, w_near = _compute_kernel_bandwidths(near, smooth_factor=1.0, crs=_UTM48)
    assert np.isfinite(w).all()
    assert -14399.0 not in set(kernel_df["rtime"])
    np.testing.assert_array_equal(kernel_df["rtime"], near_df["rtime"])
    np.testing.assert_array_equal(w, w_near)


def test_projected_kernel_is_zero_when_no_step_has_a_width():
    """When every step is where the projection has no value, nothing is smoothed."""
    pytest.importorskip("pyproj")
    p = _tropical_run()
    # Every step's cloud centered at 20 E, 85 degrees from the meridian.
    centered = p["lon"] - p.groupby("age")["lon"].transform("mean") + 20.0
    p = _with_rtime(p.assign(lon=centered, lat=p["lat"] - p["lat"].mean() + 2.0))
    _, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, crs=_UTM48)
    np.testing.assert_array_equal(w, np.zeros(len(w)))


def test_longlat_kernel_of_a_cloud_across_180_degrees_is_the_cloud_at_0_degrees():
    """
    A cloud on both sides of 180 degrees gets the kernel of the same cloud at 0 degrees.

    On a global grid nothing shifts the longitudes, which HYSPLIT writes in
    -180 to 180, so the cloud's values near -180 and +180 gave a variance of
    thousands of square degrees and a kernel more than ten times too wide.
    """
    across = _with_rtime(_cloud_at(179.9))
    assert across["lon"].min() < -179 and across["lon"].max() > 179
    _, w_across = _compute_kernel_bandwidths(across, smooth_factor=1.0)
    _, w_zero = _compute_kernel_bandwidths(
        _with_rtime(_cloud_at(0.0)), smooth_factor=1.0
    )
    assert 0.03 < w_zero.max() < 0.3  # degrees of longitude
    np.testing.assert_allclose(w_across, w_zero, rtol=1e-6)


def test_longlat_kernel_of_a_cloud_across_0_degrees_on_a_grid_in_0_to_360():
    """
    A cloud across 0 degrees on a wide grid that crosses 180 gets its own kernel.

    A grid from -100 to 200 degrees has longitudes shifted to 0 to 360,
    which cuts a cloud around 0 in two.
    """
    cloud = _with_rtime(_cloud_at(0.0))
    shifted = cloud.assign(lon=cloud["lon"] % 360)
    assert shifted["lon"].min() < 1 and shifted["lon"].max() > 359
    _, w = _compute_kernel_bandwidths(shifted, smooth_factor=1.0)
    _, w_ref = _compute_kernel_bandwidths(cloud, smooth_factor=1.0)
    np.testing.assert_allclose(w, w_ref, rtol=1e-6)


def test_longlat_kernel_leaves_a_cloud_that_is_not_cut_alone_to_the_last_bit():
    """Longitudes of a cloud that does not meet the seam are not changed, to the last bit."""
    p = _with_rtime(_spread_particles())
    reference = p.groupby("rtime")["lon"].var().to_numpy()
    kernel_df, _ = _compute_kernel_bandwidths(p, smooth_factor=1.0)
    np.testing.assert_array_equal(kernel_df["long_var"].to_numpy(), reference)


def _cloud_from(
    release_lon: float, cloud_lon: float, *, spread: float = 0.3, lat0: float = 45.0
) -> pd.DataFrame:
    """
    Particles released at *release_lon* that are *spread* degrees apart ten days later at *cloud_lon*.

    Longitudes are written in [-180, 180), as HYSPLIT writes them.
    """
    n = 300
    rng = np.random.default_rng(7)
    dlon, dlat = rng.normal(0.0, spread, n), rng.normal(0.0, spread, n)
    lon = np.r_[np.full(n, release_lon), cloud_lon + dlon]
    p = pd.DataFrame(
        {
            "particle": np.tile(np.arange(1, n + 1), 2),
            "age": np.repeat([-1.0, -14401.0], n),
            "lon": ((lon + 180.0) % 360.0) - 180.0,
            "lat": np.r_[np.full(n, 40.77), lat0 + dlat],
            "zagl": 5.0,
            "foot": np.repeat([0.0, 1e-3], n),
        }
    )
    return p


@pytest.mark.parametrize("cloud_lon", [60.0, 68.15, 68.3, 75.0, 179.9, -111.85])
@pytest.mark.parametrize("spread", [0.3, 10.0])
def test_longlat_kernel_does_not_depend_on_where_the_release_was(cloud_lon, spread):
    """
    A cloud has the same kernel wherever its release was, even when it is on the far side of the globe.

    Release at -111.85 degrees has its antipode at 68.15 degrees. A window
    around the release put its seam there and cut a cloud that had drifted
    that far (a kernel 21 times too wide). The window is around the cloud,
    so the seam is on the far side of the globe from the cloud.
    """
    here = _with_rtime(_cloud_from(-111.85, cloud_lon, spread=spread))
    there = _with_rtime(_cloud_from(cloud_lon, cloud_lon, spread=spread))
    k_here, w_here = _compute_kernel_bandwidths(here, smooth_factor=1.0)
    k_there, w_there = _compute_kernel_bandwidths(there, smooth_factor=1.0)
    np.testing.assert_allclose(w_here, w_there, rtol=1e-9)
    np.testing.assert_allclose(k_here["varsum"], k_there["varsum"], rtol=1e-9)


def test_longlat_kernel_keeps_a_real_spread_of_10_degrees_across_180_degrees():
    """A cloud the winds spread by 10 degrees is measured as 10 degrees, not folded and not cut."""
    across = _with_rtime(_cloud_from(179.9, 179.9, spread=10.0))
    assert across["lon"].min() < -170 and across["lon"].max() > 170
    k_across, w_across = _compute_kernel_bandwidths(across, smooth_factor=1.0)
    k_zero, w_zero = _compute_kernel_bandwidths(
        _with_rtime(_cloud_from(0.0, 0.0, spread=10.0)), smooth_factor=1.0
    )
    late = k_across["rtime"] < -1
    assert 80 < k_across.loc[late, "long_var"].iloc[0] < 125  # 10 degrees squared
    np.testing.assert_allclose(w_across, w_zero, rtol=1e-9)


def test_longlat_kernel_of_particles_that_circled_the_globe_is_the_kernel_of_their_positions():
    """
    A particle that went round the earth for days is where it is, not 720 degrees away.

    Longitudes written unwrapped (here up to three turns either way) give
    the kernel of the same positions written in -180 to 180.
    """
    p = _with_rtime(_cloud_from(-111.85, 179.9))
    turns = np.random.default_rng(3).integers(-3, 4, len(p))
    circled = p.assign(lon=p["lon"] + 360.0 * turns)
    assert circled["lon"].abs().max() > 540
    k_wrapped, w_wrapped = _compute_kernel_bandwidths(p, smooth_factor=1.0)
    k_circled, w_circled = _compute_kernel_bandwidths(circled, smooth_factor=1.0)
    np.testing.assert_allclose(w_circled, w_wrapped, rtol=1e-9)


@pytest.mark.parametrize(
    "grid_bounds",
    [(40.0, 100.0), (-30.0, 30.0), (-180.0, 180.0)],
    ids=["asia", "around_0", "global"],
)
@pytest.mark.parametrize("cloud_lon", [68.15, 68.3, 0.2])
def test_footprint_does_not_depend_on_where_the_release_was(
    point_receptor, grid_bounds, cloud_lon
):
    """
    Whether the cloud is on the grid or not, its footprint does not depend on its release longitude.

    On a regional grid nowhere near 180 degrees, the cloud at the antipode of
    the release (68.15 degrees for a release at -111.85) had dozens of times
    the nonzero cells it has at 60 degrees.
    """
    west, east = grid_bounds
    if not west - 5 < cloud_lon < east + 5:
        pytest.skip("a cloud this far from the grid leaves no footprint")
    xres = 0.25 if east - west > 100 else 0.1
    grid = Grid(xmin=west, xmax=east, ymin=20.0, ymax=70.0, xres=xres, yres=xres)
    feet = []
    for release_lon in (-111.85, 10.0):
        receptor = point_receptor.model_copy(update={"longitude": release_lon})
        particles = _cloud_from(release_lon, cloud_lon)
        feet.append(
            calc_footprint(particles, receptor, grid, time_integrate=True)
            .isel(time=0)
            .to_numpy()
        )
    np.testing.assert_allclose(feet[0], feet[1], atol=1e-14, rtol=0)
    assert 100 < np.count_nonzero(feet[0]) < 2000


@pytest.mark.parametrize("cloud_lon", [60.0, 68.15, 68.3, 75.0, 179.9, -111.85])
@pytest.mark.parametrize("spread", [0.3, 10.0])
def test_projected_kernel_does_not_depend_on_where_the_release_was(cloud_lon, spread):
    """
    On a projected grid too, a cloud has the same kernel wherever its release was.

    Release at -111.85 degrees has its antipode at 68.15 degrees. A window
    around the release put its seam there and cut a cloud that had drifted
    that far (a kernel about 20 times too wide). The window is around the
    cloud, so the seam is on the far side of the globe from the cloud.
    """
    pytest.importorskip("pyproj")
    crs = _lambert_at(cloud_lon, 45.0)
    here = _with_rtime(_cloud_from(-111.85, cloud_lon, spread=spread))
    there = _with_rtime(_cloud_from(cloud_lon, cloud_lon, spread=spread))
    k_here, w_here = _compute_kernel_bandwidths(here, smooth_factor=1.0, crs=crs)
    k_there, w_there = _compute_kernel_bandwidths(there, smooth_factor=1.0, crs=crs)
    np.testing.assert_allclose(w_here, w_there, rtol=1e-9)
    np.testing.assert_allclose(k_here["varsum"], k_there["varsum"], rtol=1e-9)


@pytest.mark.parametrize("cloud_lon", [60.0, 68.15, 68.3])
def test_projected_footprint_does_not_depend_on_where_the_release_was(
    point_receptor, cloud_lon
):
    """
    On a 5 km Lambert grid the cloud at the antipode of the release is the cloud at 60 degrees.

    With the window around the release, the cloud at 68.15 degrees (release
    at -111.85) had a kernel about 20 times too wide and 40 times the
    nonzero cells.
    """
    pytest.importorskip("pyproj")
    grid = Grid(
        xmin=cloud_lon - 8,
        xmax=cloud_lon + 8,
        ymin=40.0,
        ymax=50.0,
        xres=5000,
        yres=5000,
        crs=_lambert_at(cloud_lon, 45.0),
    )
    feet = []
    for release_lon in (-111.85, cloud_lon):
        receptor = point_receptor.model_copy(update={"longitude": release_lon})
        particles = _cloud_from(release_lon, cloud_lon)
        feet.append(
            calc_footprint(particles, receptor, grid, time_integrate=True)
            .isel(time=0)
            .to_numpy()
        )
    np.testing.assert_allclose(feet[0], feet[1], atol=1e-14, rtol=0)
    assert 500 < np.count_nonzero(feet[0]) < 3000


_GLOBAL = {"xmin": -180.0, "xmax": 180.0, "ymin": 48.0, "ymax": 52.0}


def _global_footprint(lon0: float, receptor: PointReceptor, **grid_kw) -> np.ndarray:
    """The time-integrated footprint of ``_cloud_at(lon0)`` on a global 0.05 degree grid."""
    grid = Grid(**{**_GLOBAL, "xres": 0.05, "yres": 0.05, **grid_kw})
    receptor = receptor.model_copy(
        update={"longitude": float(((lon0 + 180.0) % 360.0) - 180.0), "latitude": 50.0}
    )
    foot = calc_footprint(_cloud_at(lon0), receptor, grid, time_integrate=True)
    return foot.isel(time=0).to_numpy()


@pytest.mark.parametrize("lon0", [179.9, -179.95, 175.0, 180.0])
def test_global_footprint_of_a_cloud_near_180_degrees_is_the_cloud_at_0_degrees(
    point_receptor, lon0
):
    """
    On a global grid a cloud near 180 degrees keeps its total and its shape.

    The footprint of the cloud, shifted by 180 degrees, is the footprint of
    the cloud at ``lon0 - 180`` rolled by half the cells. Before, a cloud
    across 180 degrees was smoothed over tens of times as many cells, and
    what spilled past the grid's edges was lost (42 % of the footprint at
    179.9 degrees).
    """
    near = _global_footprint(lon0, point_receptor)
    away = _global_footprint(lon0 - 180.0, point_receptor)
    np.testing.assert_allclose(near.sum(), 1e-3, rtol=1e-9)
    np.testing.assert_allclose(near, np.roll(away, 3600, axis=1), atol=1e-14, rtol=0)


def test_global_footprint_wraps_smoothing_across_the_seam(point_receptor):
    """Smoothing that spills past the east edge of a global grid returns at the west edge."""
    near = _global_footprint(179.9, point_receptor)
    assert near[:, :20].sum() > 0 and near[:, -20:].sum() > 0
    unsmoothed = calc_footprint(
        _cloud_at(179.9),
        point_receptor.model_copy(update={"longitude": 179.9, "latitude": 50.0}),
        Grid(**_GLOBAL, xres=0.05, yres=0.05),
        time_integrate=True,
        smooth_factor=0,
    )
    # The mass past the seam is in the cells smoothing added, not lost.
    np.testing.assert_allclose(near.sum(), float(unsmoothed.sum()), rtol=1e-9)


@pytest.mark.parametrize("bounds", [(0.0, 360.0), (-90.0, 270.0)])
def test_global_footprint_is_the_same_on_a_grid_written_in_0_to_360(
    point_receptor, bounds
):
    """A grid from 0 to 360, or any 360 degrees, is anchored to -180 to 180, as STILT-R does."""
    reference = _global_footprint(179.9, point_receptor)
    other = _global_footprint(179.9, point_receptor, xmin=bounds[0], xmax=bounds[1])
    np.testing.assert_array_equal(other, reference)


def test_global_grid_with_cells_that_do_not_fit_360_degrees_is_not_wrapped(
    point_receptor,
):
    """Cells of 0.7 degrees leave a gap of 0.2 degrees, so the east edge does not meet the west."""
    foot = _global_footprint(179.9, point_receptor, xres=0.7, yres=0.5)
    assert foot.shape[1] == 514
    assert 0 < foot.sum() <= 1e-3 + 1e-12


def test_footprint_of_a_cloud_across_180_degrees_on_a_grid_that_crosses_it(
    point_receptor,
):
    """A regional grid from 177.9 to -178.1 degrees gives the footprint of the cloud at 0 degrees."""
    feet = []
    for lon0, west, east in ((179.9, 177.9, -178.1), (-0.1, -2.1, 1.9)):
        grid = Grid(xmin=west, xmax=east, ymin=48.0, ymax=52.0, xres=0.05, yres=0.05)
        receptor = point_receptor.model_copy(
            update={
                "longitude": float(((lon0 + 180.0) % 360.0) - 180.0),
                "latitude": 50.0,
            }
        )
        feet.append(
            calc_footprint(_cloud_at(lon0), receptor, grid, time_integrate=True)
        )
    across, at_zero = feet
    # The grid across 180 degrees is stored west to east from -180 degrees:
    # the 1.9 degrees east of 180 first (38 cells), then the 2.1 degrees west.
    np.testing.assert_allclose(
        np.roll(across.isel(time=0).to_numpy(), 42, axis=1),
        at_zero.isel(time=0).to_numpy(),
        atol=1e-14,
        rtol=0,
    )


def _edge_cloud(lon0: float, sd_lon: float = 0.03) -> pd.DataFrame:
    """
    400 particles at (lon0, 50) at release and ten days later ``sd_lon`` by 0.3 degrees apart.

    Longitudes are written in [-180, 180), as HYSPLIT writes them.
    """
    n = 400
    rng = np.random.default_rng(3)
    lon = np.concatenate([np.full(n, lon0), lon0 + rng.normal(0.0, sd_lon, n)])
    return pd.DataFrame(
        {
            "particle": np.tile(np.arange(1, n + 1), 2),
            "age": np.repeat([-1.0, -14401.0], n),
            "lon": ((lon + 180.0) % 360.0) - 180.0,
            "lat": np.concatenate([np.full(n, 50.0), rng.normal(50.0, 0.3, n)]),
            "zagl": np.full(2 * n, 5.0),
            "foot": np.repeat([0.0, 1e-3], n),
        }
    )


def _edge_footprint(
    receptor: PointReceptor, lon0: float, xmin: float, xmax: float, sd_lon: float
) -> np.ndarray:
    """The time-integrated footprint of ``_edge_cloud`` on 0.1 degree cells."""
    grid = Grid(xmin=xmin, xmax=xmax, ymin=45.0, ymax=55.0, xres=0.1, yres=0.1)
    receptor = receptor.model_copy(
        update={"longitude": float(((lon0 + 180.0) % 360.0) - 180.0), "latitude": 50.0}
    )
    foot = calc_footprint(
        _edge_cloud(lon0, sd_lon), receptor, grid, time_integrate=True
    )
    return foot.isel(time=0).to_numpy()


@pytest.mark.parametrize(
    ("lon0", "xmin", "xmax", "sd_lon"),
    [
        (179.85, -180.0, -170.0, 0.03),  # just across the west edge, all east of 180
        (-179.85, 170.0, 180.0, 0.03),  # just across the east edge, all west of 180
        (-179.6, -180.0, -170.0, 0.3),  # on both sides of 180
        (179.85, -179.9, -170.0, 0.03),  # an edge short of 180, its padding past it
    ],
)
def test_footprint_beside_a_grid_edge_near_180_degrees_is_the_footprint_away_from_it(
    point_receptor, lon0, xmin, xmax, sd_lon
):
    """
    Particles across 180 degrees from a grid's edge are smoothed onto it, as at any other edge.

    The grid and the cloud turned by 180 degrees, away from 180, give the
    same footprint. Before, a grid that ends at or near 180 degrees dropped
    the particles on the other side: a cloud only there gave
    ``EmptyFootprint``.
    """
    near = _edge_footprint(point_receptor, lon0, xmin, xmax, sd_lon)
    turn = 180.0 if xmin < 0 else -180.0
    away = _edge_footprint(
        point_receptor, lon0 + turn, xmin + turn, xmax + turn, sd_lon
    )
    assert near.sum() > 1e-5
    np.testing.assert_allclose(near, away, atol=1e-14, rtol=0)


@pytest.mark.parametrize(
    ("lon0", "bounds"),
    [
        (-112.0, (-113.0, -111.0)),  # a regional grid
        (-179.5, (-179.0, -170.0)),  # near 180, its padding short of it
        (179.9, (177.9, -178.1)),  # across 180, written 0 to 360
        (179.9, (-180.0, 180.0)),  # all the way around
    ],
)
def test_footprint_whose_padding_does_not_pass_180_degrees_is_unchanged(
    point_receptor, monkeypatch, lon0, bounds
):
    """Where the padded grid does not pass 180 degrees, no longitude moves, and the footprint is the same to the last bit."""
    grid = Grid(
        xmin=bounds[0], xmax=bounds[1], ymin=48.0, ymax=52.0, xres=0.05, yres=0.05
    )
    receptor = point_receptor.model_copy(
        update={"longitude": float(((lon0 + 180.0) % 360.0) - 180.0), "latitude": 50.0}
    )
    p = _cloud_at(lon0)
    new = calc_footprint(p, receptor, grid, time_integrate=True).to_numpy()
    monkeypatch.setattr(gridding, "_onto_padded_axis", lambda lon, **_: lon)
    old = calc_footprint(p, receptor, grid, time_integrate=True).to_numpy()
    assert new.sum() > 0
    np.testing.assert_array_equal(new, old)


def test_onto_padded_axis_moves_only_longitudes_off_the_axis():
    """A longitude moves a whole turn only when it is off the axis and the turn puts it on."""
    lon = pd.Series([179.95, -179.95, 179.0, -175.0, 0.0, 180.0])
    out = _onto_padded_axis(lon, west=-180.2, east=-169.8)
    np.testing.assert_allclose(out, [-180.05, -179.95, 179.0, -175.0, 0.0, -180.0])
    # Longitudes already on the axis are the same objects' values, bit for bit.
    assert out.iloc[1] == lon.iloc[1] and out.iloc[3] == lon.iloc[3]


def _by_meridian(foot: xr.DataArray, turn: float = 0.0) -> pd.Series:
    """The nonzero cells of a time-integrated footprint, by longitude + *turn* modulo 360 and latitude."""
    cells = foot.isel(time=0).to_series()
    cells = cells[cells != 0]
    index = cells.index.to_frame(index=False)
    index["lon"] = np.round((index["lon"] + turn) % 360.0, 6)
    return pd.Series(
        cells.to_numpy(), index=pd.MultiIndex.from_frame(index)
    ).sort_index()


def _crossing_cloud(lon0: float, n: int = 40) -> pd.DataFrame:
    """
    *n* particles moving east 0.03 degree a minute from *lon0* for their first 100 minutes.

    The median step is above a 0.01 degree cell, so the early-time
    interpolation runs. Longitudes are written in [-180, 180), as HYSPLIT
    writes them, so a cloud from 179 jumps from 179.99 to -179.98.
    """
    rng = np.random.default_rng(5)
    frames = []
    for k in range(1, n + 1):
        lon = lon0 + 0.03 * np.arange(100) + rng.normal(0.0, 0.002)
        frames.append(
            pd.DataFrame(
                {
                    "particle": k,
                    "age": -np.arange(1, 101.0),
                    "lon": ((lon + 180.0) % 360.0) - 180.0,
                    "lat": 50.0 + 0.001 * np.arange(100),
                    "zagl": 5.0,
                    "foot": 1e-5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _crossing_footprint(
    point_receptor: PointReceptor, lon0: float, bounds: tuple[float, float], crs: str
) -> xr.DataArray:
    """The unsmoothed, time-integrated footprint of ``_crossing_cloud(lon0)``."""
    res = 0.01 if crs == "+proj=longlat" else 500.0
    grid = Grid(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=49.5,
        ymax=50.5,
        xres=res,
        yres=res,
        crs=crs,
    )
    receptor = point_receptor.model_copy(
        update={"longitude": float(((lon0 + 180.0) % 360.0) - 180.0), "latitude": 50.0}
    )
    return calc_footprint(
        _crossing_cloud(lon0), receptor, grid, time_integrate=True, smooth_factor=0
    )


@pytest.mark.parametrize(
    ("bounds", "turned_bounds", "crs", "turned_crs"),
    [
        ((-180.0, 180.0), (-180.0, 180.0), "+proj=longlat", "+proj=longlat"),
        ((-180.0, -170.0), (0.0, 10.0), "+proj=longlat", "+proj=longlat"),
        (
            (178.0, 182.0),
            (-2.0, 2.0),
            "+proj=laea +lat_0=50 +lon_0=180 +units=m",
            "+proj=laea +lat_0=50 +lon_0=0 +units=m",
        ),
    ],
)
def test_early_interpolation_across_180_is_that_of_the_same_cloud_away_from_it(
    point_receptor, bounds, turned_bounds, crs, turned_crs
):
    """
    Positions interpolated across 180 degrees stay near the particles (#316).

    The cloud and grid turned by 180 degrees, so the particles cross 0
    instead, give the same footprint. Before, a step from 179.99 to -179.98
    was interpolated through 0, which put 0.5 % of the footprint on a global
    grid on the far side of the earth, and lost it on the other grids.
    """
    if crs != "+proj=longlat":
        pytest.importorskip("pyproj")
    near = _crossing_footprint(point_receptor, 179.0037, bounds, crs)
    away = _crossing_footprint(point_receptor, -0.9963, turned_bounds, turned_crs)
    assert float(near.sum()) > 1e-4
    if crs == "+proj=longlat":
        a, b = _by_meridian(near), _by_meridian(away, turn=180.0)
        pd.testing.assert_index_equal(a.index, b.index)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    else:
        np.testing.assert_allclose(near, away, rtol=1e-12, atol=0)


def test_early_interpolation_across_0_puts_nothing_on_a_grid_across_180(
    point_receptor,
):
    """On a grid shifted to 0 to 360, particles crossing 0 put nothing at 180 degrees."""
    with pytest.raises(EmptyFootprint):
        _crossing_footprint(point_receptor, -0.9963, (170.0, 190.0), "+proj=longlat")


def test_interpolate_longitudes_takes_the_short_way_round():
    """A step across the seam passes the seam, and the positions stay in the frame written."""
    time = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 0.5, 1.0, 1.25, 1.75, 2.0])
    lon = np.array([179.8, 179.9, -179.9])
    out = _interpolate_longitudes(target, time, lon, west=-180.0)
    np.testing.assert_allclose(out, [179.8, 179.85, 179.9, 179.95, -179.95, -179.9])
    np.testing.assert_array_equal(out[[0, 2, 5]], lon)

    # On a grid shifted to 0 to 360 the seam is at 0.
    out = _interpolate_longitudes(target, time, lon % 360.0, west=0.0)
    np.testing.assert_allclose(
        out, np.array([179.8, 179.85, 179.9, 179.95, 180.05, 180.1])
    )
    lon = np.array([359.8, 359.9, 0.1])
    out = _interpolate_longitudes(target, time, lon, west=0.0)
    np.testing.assert_allclose(out, [359.8, 359.85, 359.9, 359.95, 0.05, 0.1])


def test_early_step_across_180_is_measured_the_short_way_round():
    """Steps of 0.003 degree across 180 are shorter than a 0.01 degree cell, so nothing is interpolated."""
    p = pd.DataFrame(
        {
            "particle": [1, 1, 1, 2, 2, 2],
            "age": [-1.0, -2.0, -3.0] * 2,
            "lon": [179.998, -179.999, -179.996, 179.997, -180.0, -179.997],
            "lat": [50.0] * 6,
            "foot": [1e-5] * 6,
        }
    )
    assert not _interpolates(p, xres=0.01, yres=0.01)


def test_interpolate_longitudes_of_a_track_that_does_not_cross_is_unchanged():
    """A track with no step over 180 degrees is interpolated as ``numpy.interp`` does, bit for bit."""
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(-100.0, 0.0, 20))
    target = np.unique(np.concatenate([time, np.linspace(-100.0, 0.0, 301)]))
    lon = -112.0 + np.cumsum(rng.normal(0.0, 0.05, 20))
    np.testing.assert_array_equal(
        _interpolate_longitudes(target, time, lon, west=-180.0),
        np.interp(target, time, lon, left=np.nan, right=np.nan),
    )


def test_fold_padding_around_the_globe_adds_each_side_to_the_other():
    """Padding past the east edge goes to the first cells, and past the west edge to the last."""
    n_lon, pad = 6, 2
    arr = np.arange(1.0, n_lon + 2 * pad + 1).reshape(-1, 1, 1)  # 1..10
    folded = _fold_padding_around_the_globe(arr, pad=pad, n_lon=n_lon)[:, 0, 0]
    # cells 3..8 are the grid; 1, 2 lie west of it (cells 5 and 6 again),
    # 9, 10 lie east of it (cells 3 and 4 again).
    np.testing.assert_array_equal(folded, [3 + 9, 4 + 10, 5, 6, 7 + 1, 8 + 2])
    assert folded.sum() == arr.sum()


def test_fold_padding_around_the_globe_when_the_padding_is_wider_than_the_grid():
    """A kernel wider than the grid wraps more than once, and no mass is lost."""
    n_lon, pad = 3, 5
    arr = np.ones((n_lon + 2 * pad, 2, 1))
    folded = _fold_padding_around_the_globe(arr, pad=pad, n_lon=n_lon)
    assert folded.sum() == arr.sum()
    assert folded.shape == (n_lon, 2, 1)


def _smoothing_variance_km2(foot, unsmoothed, x_km, y_km) -> tuple[float, float]:
    """Return the variance smoothing adds to a footprint along x and y, in km**2."""

    def variances(f):
        a = f.isel(time=0).to_numpy()
        xx, yy = np.meshgrid(x_km, y_km)
        mx, my = (a * xx).sum() / a.sum(), (a * yy).sum() / a.sum()
        return (
            (a * (xx - mx) ** 2).sum() / a.sum(),
            (a * (yy - my) ** 2).sum() / a.sum(),
        )

    (vx, vy), (vx0, vy0) = variances(foot), variances(unsmoothed)
    return vx - vx0, vy - vy0


def test_projected_footprint_is_smoothed_like_the_longlat_footprint(point_receptor):
    """
    Smoothing spreads a 1 km UTM footprint east to west as much as a 0.01 degree one.

    The variance the kernel adds is its width squared, about (1.3 km)**2
    here. On the longitude/latitude grid the kernel is one width in degrees
    on both axes, so it is ``1 / cos(lat)`` (1.31) times wider north to
    south; the UTM kernel is round. With STILT-R's width on the projected
    coordinates the UTM footprint was hardly smoothed at all.
    """
    pyproj = pytest.importorskip("pyproj")
    p = _spread_particles()
    bounds = {"xmin": -112.5, "xmax": -111.5, "ymin": 40.0, "ymax": 41.0}
    geod = pyproj.Geod(ellps="WGS84")
    km_per_lon = geod.inv(-112.5, 40.5, -111.5, 40.5)[2] / 1000
    km_per_lat = geod.inv(-112.0, 40.0, -112.0, 41.0)[2] / 1000

    added = {}
    for name, grid in {
        "lonlat": Grid(**bounds, xres=0.01, yres=0.01),
        "utm": Grid(**bounds, xres=1000, yres=1000, crs=_UTM12),
    }.items():
        foot = calc_footprint(p, point_receptor, grid, time_integrate=True)
        unsmoothed = calc_footprint(
            p, point_receptor, grid, time_integrate=True, smooth_factor=0
        )
        if name == "lonlat":
            x_km, y_km = foot.lon.values * km_per_lon, foot.lat.values * km_per_lat
        else:
            x_km, y_km = foot.x.values / 1000, foot.y.values / 1000
        added[name] = _smoothing_variance_km2(foot, unsmoothed, x_km, y_km)

    (ll_x, ll_y), (utm_x, utm_y) = added["lonlat"], added["utm"]
    assert utm_x > 1.0  # a kernel over 1 km wide
    np.testing.assert_allclose(utm_x, ll_x, rtol=0.1)
    np.testing.assert_allclose(utm_y, utm_x, rtol=0.1)
    np.testing.assert_allclose(ll_y, ll_x / np.cos(np.radians(40.5)) ** 2, rtol=0.1)


def test_projected_footprint_on_cells_that_are_not_square_keeps_its_total(
    point_receptor,
):
    """With 1000 m by 500 m cells, smoothing spreads the footprint and loses none of it."""
    pytest.importorskip("pyproj")
    p = _spread_particles()
    grid = Grid(
        xmin=-112.5, xmax=-111.5, ymin=40.0, ymax=41.0, xres=1000, yres=500, crs=_UTM12
    )
    smoothed = calc_footprint(p, point_receptor, grid, time_integrate=True)
    unsmoothed = calc_footprint(
        p, point_receptor, grid, time_integrate=True, smooth_factor=0
    )
    assert np.count_nonzero(smoothed.to_numpy()) > 2 * np.count_nonzero(
        unsmoothed.to_numpy()
    )
    np.testing.assert_allclose(
        float(smoothed.sum()), float(unsmoothed.sum()), rtol=1e-12
    )


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
