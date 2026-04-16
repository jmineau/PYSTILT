"""Fidelity test: Python vs R-STILT numerical agreement.

Tests in this module verify two things:

1. **Trajectory fixture integrity** — the committed parquet has the expected
   shape, columns, particle count, and time range produced by uataq/stilt.

2. **Footprint numerical agreement** — ``Footprint.calculate()`` applied to
   the reference trajectory produces a field that agrees cell-by-cell with
   the reference NetCDF from R-STILT.

It requires neither HYSPLIT, met files, nor an R installation — just the
static reference fixtures in ``tests/fixtures/r_ref/``:

- ``r_traj.parquet`` — particle trajectory from uataq/stilt (n_hours=-6, numpar=1000, WBB receptor), converted from the
  native ``_traj.rds`` format by ``tests/fixtures/generate_r_ref.sh``.
- ``r_foot.nc`` — reference footprint NetCDF produced by R-STILT for the
  same simulation with the standard 0.01° WBB domain.

To regenerate reference fixtures run from the PYSTILT repo root::

    bash tests/fixtures/generate_r_ref.sh

then commit the updated files in ``tests/fixtures/r_ref/``.

Tolerance philosophy
--------------------
R and Python use the same binning algorithm but differ in floating-point
accumulation order.  Empirically the max cell-level relative error is
O(1e-6).  We assert:

- ``np.testing.assert_allclose(rtol=1e-4)`` — strict enough to catch
  algorithmic regressions, tolerant of benign FP reordering.
- Total sum relative error < 1e-5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid
from stilt.footprint import Footprint
from stilt.receptor import Receptor

# ---------------------------------------------------------------------------
# Paths to committed reference fixtures
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "r_ref"
_R_TRAJ = _FIXTURES / "r_traj.parquet"
_R_FOOT = _FIXTURES / "r_foot.nc"

# ---------------------------------------------------------------------------
# Reference simulation constants (uataq/stilt 01-wbb, n_hours=-6, numpar=1000)
# ---------------------------------------------------------------------------

_RECEPTOR = Receptor(
    time="2015-12-10 00:00:00",
    longitude=-112.0,
    latitude=40.5,
    altitude=5.0,
)
_FOOT_CONFIG = FootprintConfig(
    grid=Grid(xmin=-113.0, xmax=-111.0, ymin=39.5, ymax=41.5, xres=0.01, yres=0.01),
    smooth_factor=1.0,
    time_integrate=False,
)

# Expected trajectory properties (from the uataq/stilt $particle data frame)
_EXPECTED_TRAJ_COLS = {
    "time",
    "indx",
    "long",
    "lati",
    "zagl",
    "foot",
    "mlht",
    "dens",
    "samt",
    "sigw",
    "tlgr",
    "foot_no_hnf_dilution",
}
_EXPECTED_N_PARTICLES = 1000
_EXPECTED_N_HOURS = 6  # |n_hours|
_EXPECTED_TIME_STEP_MIN = 1.0

# ---------------------------------------------------------------------------
# Module-scoped fixtures (loaded once for all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_traj() -> pd.DataFrame:
    """Raw particle DataFrame from the committed R-STILT reference trajectory."""
    return pd.read_parquet(_R_TRAJ)


@pytest.fixture(scope="module")
def r_foot_ds() -> xr.Dataset:
    """R-STILT reference footprint, opened from the committed NetCDF."""
    return xr.open_dataset(_R_FOOT)


@pytest.fixture(scope="module")
def py_foot(r_traj) -> Footprint:
    """Python footprint computed from the R-STILT reference trajectory."""
    result = Footprint.calculate(
        particles=r_traj,
        receptor=_RECEPTOR,
        config=_FOOT_CONFIG,
    )
    if result is None:
        pytest.fail("Footprint.calculate() returned None for the reference trajectory")
    return result


# ---------------------------------------------------------------------------
# Trajectory fixture integrity tests
# ---------------------------------------------------------------------------


def test_trajectory_columns(r_traj: pd.DataFrame) -> None:
    """Reference parquet contains all columns required by Footprint.calculate()."""
    missing = _EXPECTED_TRAJ_COLS - set(r_traj.columns)
    assert not missing, f"Missing columns in reference trajectory: {missing}"


def test_trajectory_particle_count(r_traj: pd.DataFrame) -> None:
    """Reference parquet has exactly the expected number of unique particles."""
    n = r_traj["indx"].nunique()
    assert n == _EXPECTED_N_PARTICLES, (
        f"Expected {_EXPECTED_N_PARTICLES} particles, got {n}"
    )


def test_trajectory_time_range(r_traj: pd.DataFrame) -> None:
    """Particle time column spans the expected backward simulation window."""
    # time is in minutes; backward runs have negative values
    t_min = r_traj["time"].min()
    t_max = r_traj["time"].max()
    expected_min = -_EXPECTED_N_HOURS * 60
    assert t_min >= expected_min, (
        f"Time extends beyond n_hours={_EXPECTED_N_HOURS}: min={t_min}"
    )
    assert t_max < 0, f"Latest time step should be negative (backward run), got {t_max}"


def test_trajectory_foot_nonnegative(r_traj: pd.DataFrame) -> None:
    """All particle footprint influence values are non-negative."""
    neg = (r_traj["foot"] < 0).sum()
    assert neg == 0, f"{neg} negative 'foot' values in reference trajectory"


# ---------------------------------------------------------------------------
# Footprint numerical agreement tests
# ---------------------------------------------------------------------------


def test_footprint_grid_matches_r(py_foot: Footprint, r_foot_ds: xr.Dataset) -> None:
    """Python and R footprints share the same lat/lon grid and time dimension."""
    assert py_foot.data.sizes["time"] == r_foot_ds.sizes["time"], (
        f"Time steps differ: Python={py_foot.data.sizes['time']}, "
        f"R={r_foot_ds.sizes['time']}"
    )
    np.testing.assert_allclose(
        py_foot.data.lat.values,
        r_foot_ds.lat.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Latitude coordinates differ from R reference",
    )
    np.testing.assert_allclose(
        py_foot.data.lon.values,
        r_foot_ds.lon.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Longitude coordinates differ from R reference",
    )


def test_footprint_total_sum_matches_r(
    py_foot: Footprint, r_foot_ds: xr.Dataset
) -> None:
    """Total footprint sum agrees with R to within 1e-5 relative error."""
    py_sum = float(py_foot.data.sum())
    r_sum = float(r_foot_ds.foot.sum())

    assert r_sum > 0, "R reference footprint has zero total — check the reference file"
    rel_err = abs(py_sum - r_sum) / r_sum
    assert rel_err < 1e-5, (
        f"Total footprint sum mismatch: Python={py_sum:.8f}, R={r_sum:.8f}, "
        f"relative error={rel_err:.2e} (threshold=1e-5)"
    )


def test_footprint_cell_values_match_r(
    py_foot: Footprint, r_foot_ds: xr.Dataset
) -> None:
    """Per-cell values agree with R to rtol=1e-4 across all (time, lat, lon) cells.

    Tolerance covers floating-point accumulation order differences between R
    and Python without permitting algorithmic regressions.
    """
    py_aligned = py_foot.data.reindex_like(
        r_foot_ds.foot, method="nearest", tolerance=1e-6
    )
    np.testing.assert_allclose(
        py_aligned.values,
        r_foot_ds.foot.values.astype(np.float64),
        rtol=1e-4,
        atol=0,
        err_msg=(
            "Per-cell footprint values differ from R reference beyond rtol=1e-4. "
            "This likely indicates a regression in Footprint.calculate()."
        ),
    )
