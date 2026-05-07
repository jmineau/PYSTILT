"""
Integration fidelity tests: seeded PYSTILT vs R-STILT run live.

Each scenario in ``ALL_SCENARIOS`` runs as a parametrized test pair:

1. ``test_setup_cfg_pins_krand_and_seed`` — SETUP.CFG written by PYSTILT
   contains the expected RNG controls for the scenario.
2. ``test_footprint_matches_r_live`` — Feed the PYSTILT trajectory (HNF-
   corrected particles) to R-STILT's ``calc_footprint`` helper and compare
   the resulting footprint grid at rtol=1e-7.

This replaces the old pre-committed-fixture approach. R is run live, which
means the test is self-contained: no committed R outputs are required, and
the test cannot silently pass against a stale fixture.

Skips automatically when:
  * Met files are absent (``met_dir`` fixture handles this).
  * R is unavailable: ``STILT_R_DIR`` not set or ``Rscript`` not on PATH
    (``rscript``/``r_stilt_dir`` fixtures from ``tests/conftest.py``).
"""

from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.hysplit.driver import _bundled_exe_dir
from stilt.model import Model

from ...fixtures.r_stilt_reference import (
    ALL_SCENARIOS,
    REFERENCE_MET_FILE_FORMAT,
    REFERENCE_MET_FILE_INTERVAL_HOURS,
    REFERENCE_TIME,
    ReferenceScenario,
)
from ..conftest import integration

_R_HELPERS = Path(__file__).parents[2] / "fixtures" / "r_helpers"


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


def _r_trajectory_live(
    tmp_path: Path,
    rscript: str,
    r_stilt_dir: Path,
    met_dir: Path,
    scenario: ReferenceScenario,
) -> pd.DataFrame:
    """Run R-STILT calc_trajectory live and return the particle table."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_path = tmp_path / "r_traj.parquet"
    work_dir = tmp_path / "r_run"

    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_trajectory.r"),
            str(out_path),
            str(work_dir),
            str(r_stilt_dir),
            str(met_dir),
            REFERENCE_MET_FILE_FORMAT,
            f"{REFERENCE_MET_FILE_INTERVAL_HOURS} hours",
            REFERENCE_TIME.strftime("%Y-%m-%dT%H:%M:%S"),
            _csv(scenario.longitude),
            _csv(scenario.latitude),
            _csv(scenario.altitude),
            str(scenario.n_hours),
            str(scenario.numpar),
            str(scenario.krand),
            str(scenario.seed),
            str(scenario.hnf_plume).upper(),
            "TRUE",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[{scenario.name}] calc_trajectory.r failed "
            f"(exit {result.returncode}):\nSTDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )
    return pd.read_parquet(out_path)


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
    if result.returncode != 0:
        raise RuntimeError(
            f"[{scenario.name}] calc_footprint.r failed "
            f"(exit {result.returncode}):\n{result.stderr}"
        )

    if not nc_path.exists():
        # calc_footprint returned NULL — all particles outside domain.
        n_lat = round((scenario.ymax - scenario.ymin) / scenario.yres)
        n_lon = round((scenario.xmax - scenario.xmin) / scenario.xres)
        lat = scenario.ymin + np.arange(n_lat) * scenario.yres
        lon = scenario.xmin + np.arange(n_lon) * scenario.xres
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


def _scenario_id(s: ReferenceScenario) -> str:
    return s.name


# ---------------------------------------------------------------------------
# Per-scenario PYSTILT run fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=ALL_SCENARIOS,
    ids=_scenario_id,
)
def scenario_outputs(request, met_dir, tmp_path_factory) -> dict:
    """Run one seeded PYSTILT simulation per scenario and return output paths."""
    scenario: ReferenceScenario = request.param

    project_dir = tmp_path_factory.mktemp(f"fidelity_{scenario.name}")
    receptor = scenario.make_receptor()
    config = scenario.make_model_config(met_dir)
    model = Model(project=project_dir, config=config, receptors=[receptor])
    model.run()

    sim_id = scenario.py_sim_id()
    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sim_id

    traj_files = list(sim_dir.glob("*_traj.parquet"))
    foot_files = list(sim_dir.glob("*_foot.nc"))

    if not traj_files:
        pytest.fail(f"[{scenario.name}] No trajectory parquet found in {sim_dir}")
    if not foot_files:
        pytest.fail(f"[{scenario.name}] No footprint NetCDF found in {sim_dir}")

    return {
        "scenario": scenario,
        "traj": traj_files[0],
        "foot": foot_files[0],
        "setup": sim_dir / "SETUP.CFG",
    }


# ---------------------------------------------------------------------------
# Fidelity tests (parametrized via scenario_outputs)
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
def test_trajectory_matches_r_live(
    scenario_outputs: dict,
    rscript: str,
    r_stilt_dir: Path,
    met_dir: Path,
    tmp_path: Path,
) -> None:
    """
    PYSTILT trajectory table matches R-STILT when both run the same hycs_std.

    This compares particle positions and footprint sensitivity columns after
    R-STILT's trajectory read and HNF correction, before either implementation
    performs footprint gridding.
    """
    s: ReferenceScenario = scenario_outputs["scenario"]
    _assert_hysplit_binary_matches_r(r_stilt_dir)
    compare_columns = _trajectory_compare_columns(s)

    py_traj = pd.read_parquet(scenario_outputs["traj"])
    r_traj = _r_trajectory_live(tmp_path / s.name, rscript, r_stilt_dir, met_dir, s)

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
def test_footprint_matches_r_live(
    scenario_outputs: dict,
    rscript: str,
    r_stilt_dir: Path,
    tmp_path: Path,
) -> None:
    """
    PYSTILT footprint matches R-STILT output when given the same particles.

    Feeds the PYSTILT trajectory parquet (HNF-corrected foot column) directly
    to R's calc_footprint helper and compares the resulting grid values at
    rtol=1e-7.  This is self-contained — no pre-committed R outputs needed.
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

    np.testing.assert_allclose(
        py_ds.lat.values,
        r_ds.lat.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"[{s.name}] Footprint latitude coordinates differ.",
    )
    np.testing.assert_allclose(
        py_ds.lon.values,
        r_ds.lon.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"[{s.name}] Footprint longitude coordinates differ.",
    )
    np.testing.assert_allclose(
        py_ds.foot.values.astype(np.float64),
        r_ds.foot.values.astype(np.float64),
        rtol=1e-7,
        atol=1e-12,
        err_msg=f"[{s.name}] Footprint values differ from R-STILT live output.",
    )
