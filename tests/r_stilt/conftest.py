"""Shared fixtures for r_stilt integration tests."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from stilt.project import Project

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

    Also runs STILT-R calc_trajectory.r once per scenario so the result can be
    shared by test_trajectory_matches_r and test_footprint_matches_r without
    paying a second HYSPLIT invocation.
    """
    scenario: ReferenceScenario = request.param

    project_dir = tmp_path_factory.mktemp(f"fidelity_{scenario.name}")
    receptor = scenario.make_receptor()
    config = scenario.make_model_config(met_dir)
    project = Project.init(project_dir, config=config, receptors=[receptor])
    t0 = time.perf_counter()
    project.run()
    print(f"\n[PROFILE] {scenario.name} PYSTILT sim: {time.perf_counter() - t0:.1f}s")

    sim = project.simulation(*scenario.py_sim_id())
    if not sim.has_particles:
        pytest.fail(f"[{scenario.name}] No particles written for {sim.id}")
    if not sim.has_footprint:
        pytest.fail(f"[{scenario.name}] No footprint written for {sim.id}")
    traj_file = sim.particles_path
    assert traj_file is not None
    # The footprint is stored sparse in float32; compare the footprint as
    # computed, in float64, by remaking it from the stored particles.
    foot = sim.calc_footprint()
    assert foot is not None, f"[{scenario.name}] footprint is empty"
    foot_file = project_dir / "py_foot.nc"
    foot.stilt.to_netcdf(foot_file)
    assert sim.kept_workdir is not None
    setup_file = sim.kept_workdir / "SETUP.CFG"

    error_traj_path = None
    if scenario.error_variant is not None:
        err = project.simulation(*scenario.py_sim_id(scenario.error_variant))
        error_traj_path = err.particles_path if err.has_particles else None

    # Skip R trajectory run for scenarios whose transport is identical to another.
    # test_trajectory_matches_r will pytest.skip() when r_traj is None.
    if scenario.shares_trajectory_with is not None:
        print(
            f"\n[PROFILE] {scenario.name} R trajectory: skipped"
            f" (shares trajectory with '{scenario.shares_trajectory_with}')"
        )
        return {
            "scenario": scenario,
            "traj": traj_file,
            "foot": foot_file,
            "setup": setup_file,
            "r_traj": None,
            "r_error_traj": None,
            "error_traj": None,
        }

    # Run R trajectory once here so test_trajectory_matches_r can reuse it
    # instead of paying a second HYSPLIT invocation.
    r_traj_path = project_dir / "r_traj.parquet"
    r_work_dir = project_dir / "r_run"

    # Derive kmsl from the scenario's altitude_ref.
    kmsl = 0 if scenario.altitude_ref == "agl" else 1

    # WINDERR args: pass "" when unused so calc_trajectory.r can detect absence.
    winderr_args = [
        str(scenario.siguverr) if scenario.siguverr is not None else "",
        str(scenario.tluverr) if scenario.tluverr is not None else "",
        str(scenario.zcoruverr) if scenario.zcoruverr is not None else "",
        str(scenario.horcoruverr) if scenario.horcoruverr is not None else "",
    ]

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
            str(kmsl),
            *winderr_args,
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

    # Load the R error trajectory if WINDERR produced one.
    r_error_traj = None
    if scenario.siguverr is not None:
        r_error_path = r_traj_path.parent / (
            r_traj_path.stem + "_error" + r_traj_path.suffix
        )
        if r_error_path.exists():
            r_error_traj = pd.read_parquet(r_error_path)

    return {
        "scenario": scenario,
        "traj": traj_file,
        "foot": foot_file,
        "setup": setup_file,
        "r_traj": r_traj,
        "r_error_traj": r_error_traj,
        "error_traj": error_traj_path,
    }
