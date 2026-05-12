"""
Integration fidelity tests: seeded PYSTILT vs R-STILT.

Each scenario in ``ALL_SCENARIOS`` runs as a parametrized test triple:

1. ``test_setup_cfg_pins_krand_and_seed`` — SETUP.CFG written by PYSTILT
   contains the expected RNG controls for the scenario.
2. ``test_trajectory_matches_r`` — PYSTILT particle table matches R-STILT's
   at rtol=1e-7 when both run the same hycs_std binary.
3. ``test_footprint_matches_r`` — Feed the PYSTILT trajectory (HNF-corrected
   particles) to R-STILT's ``calc_footprint`` helper and compare the resulting
   footprint grid at rtol=1e-7.

R is run directly for each test; no pre-committed fixtures are required and
tests cannot silently pass against stale outputs.

Skips automatically when:
  * Met files are absent (``met_dir`` fixture handles this).
  * R is unavailable: ``STILT_R_DIR`` not set or ``Rscript`` not on PATH
    (``rscript``/``r_stilt_dir`` fixtures from ``tests/conftest.py``).
"""

from __future__ import annotations

import subprocess
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.hysplit.driver import _bundled_exe_dir

from ..conftest import integration
from ..fixtures.r_stilt_reference import (
    ReferenceScenario,
)

pytestmark = [pytest.mark.fidelity]

_R_HELPERS = Path(__file__).parents[1] / "fixtures" / "r_helpers"


# ---------------------------------------------------------------------------
# Deep footprint comparison
# ---------------------------------------------------------------------------


def _assert_footprint_deep(
    py_ds: xr.Dataset,
    r_ds: xr.Dataset,
    scenario: ReferenceScenario,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-8,
) -> None:
    """
    Multi-level footprint comparison: coordinate exactness, scalar invariants, per-cell.

    atol=1e-8 is justified by cross-language summation-order drift of
    O(N_particles * ε * max_foot).  For standard runs (max_foot ~ 1e-2,
    N=1000) this is ~2e-15, negligible.  For low-AGL stress scenarios
    (max_foot ~ 0.1) it reaches ~2e-14.  1e-8 gives a 10^6 safety margin
    while remaining tight enough to catch any real implementation error.

    Scalar invariants tested independently of per-cell tolerance:
      - Total sum agrees at rtol=1e-6 (summation errors cancel; systematic
        errors such as a wrong normalisation factor do not).
      - Peak value agrees at the per-cell tolerance.
      - Nonzero cell count agrees (guards against phantom cells or dropped
        particles that an all-close with generous atol might miss).
    """
    py_foot = py_ds.foot.values.astype(np.float64)
    r_foot = r_ds.foot.values.astype(np.float64)

    np.testing.assert_allclose(
        py_ds.lat.values,
        r_ds.lat.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"[{scenario.name}] lat coordinates differ.",
    )
    np.testing.assert_allclose(
        py_ds.lon.values,
        r_ds.lon.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"[{scenario.name}] lon coordinates differ.",
    )

    py_sum = float(py_foot.sum())
    r_sum = float(r_foot.sum())
    np.testing.assert_allclose(
        py_sum,
        r_sum,
        rtol=1e-6,
        atol=0,
        err_msg=(
            f"[{scenario.name}] Total footprint sum differs: "
            f"PYSTILT={py_sum:.6e}, R={r_sum:.6e}"
        ),
    )

    py_peak = float(py_foot.max())
    r_peak = float(r_foot.max())
    np.testing.assert_allclose(
        py_peak,
        r_peak,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"[{scenario.name}] Peak footprint value differs: "
            f"PYSTILT={py_peak:.6e}, R={r_peak:.6e}"
        ),
    )

    py_nz = int((py_foot > 0).sum())
    r_nz = int((r_foot > 0).sum())
    assert py_nz == r_nz, (
        f"[{scenario.name}] Nonzero cell count differs: PYSTILT={py_nz}, R={r_nz}"
    )

    np.testing.assert_allclose(
        py_foot,
        r_foot,
        rtol=rtol,
        atol=atol,
        err_msg=f"[{scenario.name}] Per-cell footprint values differ from R-STILT.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_hysplit_binary_matches_r(r_stilt_dir: Path) -> None:
    py_hycs = _bundled_exe_dir() / "hycs_std"
    r_hycs = r_stilt_dir / "exe" / "hycs_std"
    assert py_hycs.exists(), f"PYSTILT hycs_std not found: {py_hycs}"
    assert r_hycs.exists(), f"R-STILT hycs_std not found: {r_hycs}"
    py_hash = _sha256_file(py_hycs)
    r_hash = _sha256_file(r_hycs)
    assert py_hash == r_hash, (
        "PYSTILT and R-STILT must use the same hycs_std binary for trajectory "
        f"fidelity tests.\nPYSTILT: {py_hycs} {py_hash}\n"
        f"R-STILT: {r_hycs} {r_hash}"
    )


def _csv(values: float | tuple[float, ...]) -> str:
    if isinstance(values, tuple):
        return ",".join(f"{v:g}" for v in values)
    return f"{values:g}"


def _sorted_trajectory(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return (
        df.loc[:, list(columns)].sort_values(by=["indx", "time"]).reset_index(drop=True)
    )


def _trajectory_compare_columns(scenario: ReferenceScenario) -> tuple[str, ...]:
    if scenario.receptor_type == "multipoint" and scenario.hnf_plume:
        # R-STILT has no direct PYSTILT-style multipoint receptor; when given
        # multiple heights in one run, calc_trajectory treats them as a
        # column/line and assigns xhgt across the particle index range.  The
        # raw HYSPLIT trajectory and uncorrected foot still match exactly, but
        # HNF-corrected foot intentionally differs because PYSTILT maps
        # particles to their actual multipoint release altitudes.
        return tuple(c for c in scenario.compare_columns if c != "foot")
    return scenario.compare_columns


def _r_footprint_from_traj(
    tmp_path: Path,
    rscript: str,
    r_stilt_dir: Path,
    traj_path: Path,
    scenario: ReferenceScenario,
) -> xr.Dataset:
    """
    Run R-STILT calc_footprint on a PYSTILT trajectory parquet and return the
    resulting footprint as an xarray Dataset.

    Feeds the HNF-corrected ``foot`` column directly to R so the comparison
    is: does PYSTILT's footprint calculation agree with R's given the same
    particle positions and foot values?
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    nc_path = tmp_path / "r_foot.nc"

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_footprint.r"),
            str(traj_path),
            str(nc_path),
            str(scenario.xmin),
            str(scenario.xmax),
            str(scenario.ymin),
            str(scenario.ymax),
            str(scenario.xres),
            str(scenario.yres),
            str(scenario.smooth_factor),
            str(scenario.time_integrate).upper(),
            str(r_stilt_dir),
        ],
        capture_output=True,
        text=True,
    )
    print(f"\n[PROFILE] {scenario.name} R footprint: {time.perf_counter() - t0:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(
            f"[{scenario.name}] calc_footprint.r failed "
            f"(exit {result.returncode}):\n{result.stderr}"
        )

    if not nc_path.exists():
        # calc_footprint returned NULL — all particles outside domain.
        # Use cell centres (+ 0.5*res) to match PYSTILT's coordinate convention.
        n_lat = round((scenario.ymax - scenario.ymin) / scenario.yres)
        n_lon = round((scenario.xmax - scenario.xmin) / scenario.xres)
        lat = scenario.ymin + (np.arange(n_lat) + 0.5) * scenario.yres
        lon = scenario.xmin + (np.arange(n_lon) + 0.5) * scenario.xres
        return xr.Dataset(
            {
                "foot": (
                    ["time", "lat", "lon"],
                    np.zeros((1, n_lat, n_lon), dtype=np.float32),
                )
            },
            coords={"lat": lat, "lon": lon, "time": [0.0]},
        )

    ds = xr.open_dataset(nc_path)
    ds.load()
    ds.close()
    return ds


# ---------------------------------------------------------------------------
# Fidelity tests (parametrized via scenario_outputs fixture in conftest.py)
# ---------------------------------------------------------------------------


@integration
def test_setup_cfg_pins_krand_and_seed(scenario_outputs: dict) -> None:
    """SETUP.CFG written by PYSTILT contains the expected RNG controls."""
    s: ReferenceScenario = scenario_outputs["scenario"]
    content = scenario_outputs["setup"].read_text().lower()

    assert f"krand={s.krand}" in content, (
        f"[{s.name}] krand={s.krand} not found in SETUP.CFG"
    )
    assert f"seed={s.seed}" in content, (
        f"[{s.name}] seed={s.seed} not found in SETUP.CFG"
    )


@integration
def test_hysplit_binary_matches_r(r_stilt_dir: Path) -> None:
    """Trajectory parity is only meaningful when both tools run the same hycs_std."""
    _assert_hysplit_binary_matches_r(r_stilt_dir)


@integration
def test_trajectory_matches_r(
    scenario_outputs: dict,
    r_stilt_dir: Path,
) -> None:
    """
    PYSTILT trajectory table matches R-STILT when both run the same hycs_std.

    This compares particle positions and footprint sensitivity columns after
    R-STILT's trajectory read and HNF correction, before either implementation
    performs footprint gridding.

    R trajectory is pre-computed in the scenario_outputs fixture to avoid
    paying a second HYSPLIT invocation.
    """
    s: ReferenceScenario = scenario_outputs["scenario"]
    if scenario_outputs["r_traj"] is None:
        pytest.skip(
            f"[{s.name}] trajectory identical to '{s.shares_trajectory_with}'; "
            "covered by that scenario's test_trajectory_matches_r"
        )
    _assert_hysplit_binary_matches_r(r_stilt_dir)
    compare_columns = _trajectory_compare_columns(s)

    py_traj = pd.read_parquet(scenario_outputs["traj"])
    r_traj = scenario_outputs["r_traj"]

    assert len(py_traj) == len(r_traj), (
        f"[{s.name}] trajectory row counts differ: "
        f"PYSTILT={len(py_traj)}, R-STILT={len(r_traj)}"
    )
    assert set(compare_columns) <= set(py_traj.columns), (
        f"[{s.name}] PYSTILT trajectory missing comparison columns: "
        f"{sorted(set(compare_columns) - set(py_traj.columns))}"
    )
    assert set(compare_columns) <= set(r_traj.columns), (
        f"[{s.name}] R-STILT trajectory missing comparison columns: "
        f"{sorted(set(compare_columns) - set(r_traj.columns))}"
    )

    py_sorted = _sorted_trajectory(py_traj, compare_columns)
    r_sorted = _sorted_trajectory(r_traj, compare_columns)

    np.testing.assert_array_equal(
        py_sorted["indx"].to_numpy(),
        r_sorted["indx"].to_numpy(),
        err_msg=f"[{s.name}] particle indices differ.",
    )
    np.testing.assert_array_equal(
        py_sorted["time"].to_numpy(),
        r_sorted["time"].to_numpy(),
        err_msg=f"[{s.name}] trajectory times differ.",
    )
    np.testing.assert_allclose(
        py_sorted.drop(columns=["indx", "time"]).to_numpy(dtype=float),
        r_sorted.drop(columns=["indx", "time"]).to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-10,
        err_msg=f"[{s.name}] trajectory values differ from R-STILT live output.",
    )


@integration
def test_footprint_matches_r(
    scenario_outputs: dict,
    rscript: str,
    r_stilt_dir: Path,
    tmp_path: Path,
) -> None:
    """
    PYSTILT footprint matches R-STILT output when given the same particles.

    Feeds the PYSTILT trajectory parquet (HNF-corrected foot column) directly
    to R's calc_footprint helper.  Uses a deep comparison that checks coordinate
    exactness, total-mass conservation, peak agreement, nonzero cell count, and
    per-cell values — catching failures that a single all-close might miss.
    """
    s: ReferenceScenario = scenario_outputs["scenario"]

    py_ds = xr.open_dataset(scenario_outputs["foot"])
    py_ds.load()
    py_ds.close()

    r_ds = _r_footprint_from_traj(
        tmp_path / s.name,
        rscript,
        r_stilt_dir,
        scenario_outputs["traj"],
        s,
    )

    _assert_footprint_deep(py_ds, r_ds, s)
