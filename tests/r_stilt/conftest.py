"""Shared fixtures for r_stilt integration tests."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from stilt.model import Model

from ..fixtures.r_stilt_reference import (
    ALL_SCENARIOS,
    ReferenceScenario,
)

_R_HELPERS = Path(__file__).parents[1] / "fixtures" / "r_helpers"


def _scenario_id(s: ReferenceScenario) -> str:
    return s.name


def _csv(values: float | tuple[float, ...]) -> str:
    if isinstance(values, tuple):
        return ",".join(f"{v:g}" for v in values)
    return f"{values:g}"


@pytest.fixture(
    scope="session",
    params=ALL_SCENARIOS,
    ids=_scenario_id,
)
def scenario_outputs(request, met_dir, rscript, r_stilt_dir, tmp_path_factory) -> dict:
    """
    Run one seeded PYSTILT simulation per scenario and return output paths.

    Also runs R-STILT calc_trajectory.r once per scenario so the result can be
    shared by test_trajectory_matches_r and test_footprint_matches_r without
    paying a second HYSPLIT invocation.
    """
    scenario: ReferenceScenario = request.param

    project_dir = tmp_path_factory.mktemp(f"fidelity_{scenario.name}")
    receptor = scenario.make_receptor()
    config = scenario.make_model_config(met_dir)
    model = Model(project=project_dir, config=config, receptors=[receptor])
    t0 = time.perf_counter()
    model.run()
    print(f"\n[PROFILE] {scenario.name} PYSTILT sim: {time.perf_counter() - t0:.1f}s")

    sim_id = scenario.py_sim_id()
    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sim_id

    traj_files = list(sim_dir.glob("*_traj.parquet"))
    foot_files = list(sim_dir.glob("*_foot.nc"))

    if not traj_files:
        pytest.fail(f"[{scenario.name}] No trajectory parquet found in {sim_dir}")
    if not foot_files:
        pytest.fail(f"[{scenario.name}] No footprint NetCDF found in {sim_dir}")

    # Skip R trajectory run for scenarios whose transport is identical to another.
    # test_trajectory_matches_r will pytest.skip() when r_traj is None.
    if scenario.shares_trajectory_with is not None:
        print(
            f"\n[PROFILE] {scenario.name} R trajectory: skipped"
            f" (shares trajectory with '{scenario.shares_trajectory_with}')"
        )
        return {
            "scenario": scenario,
            "traj": traj_files[0],
            "foot": foot_files[0],
            "setup": sim_dir / "SETUP.CFG",
            "r_traj": None,
        }

    # Run R trajectory once here so test_trajectory_matches_r can reuse it
    # instead of paying a second HYSPLIT invocation.
    r_traj_path = project_dir / "r_traj.parquet"
    r_work_dir = project_dir / "r_run"
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_trajectory.r"),
            str(r_traj_path),
            str(r_work_dir),
            str(r_stilt_dir),
            str(met_dir),
            scenario.met_file_format,
            f"{int(scenario.met_file_tres[:-1])} hours",
            scenario.time.strftime("%Y-%m-%dT%H:%M:%S"),
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
    print(f"\n[PROFILE] {scenario.name} R trajectory: {time.perf_counter() - t0:.1f}s")
    if result.returncode != 0:
        pytest.fail(
            f"[{scenario.name}] calc_trajectory.r failed "
            f"(exit {result.returncode}):\nSTDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )
    r_traj = pd.read_parquet(r_traj_path)

    return {
        "scenario": scenario,
        "traj": traj_files[0],
        "foot": foot_files[0],
        "setup": sim_dir / "SETUP.CFG",
        "r_traj": r_traj,
    }
