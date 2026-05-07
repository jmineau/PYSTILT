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
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from stilt.model import Model

from ...fixtures.r_stilt_reference import ALL_SCENARIOS, ReferenceScenario
from ..conftest import integration

_R_HELPERS = Path(__file__).parents[2] / "fixtures" / "r_helpers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
