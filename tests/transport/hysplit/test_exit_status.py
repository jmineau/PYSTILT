"""A ``hycs_std`` that ends badly is a failed simulation, whatever its particle file holds (#318)."""

import datetime as dt
import stat
from pathlib import Path

import pytest

from stilt.config import ProjectConfig
from stilt.exceptions import SimulationError
from stilt.meteorology import run_window
from stilt.project import Project
from stilt.receptors import PointReceptor
from stilt.transport.hysplit import FailureReason, HysplitConfig, HysplitModel

from ...fixtures.factories import make_met_config, make_met_files

VARS = ["time", "indx", "long", "lati", "zagl", "foot"]
N_HOURS = -24
RECEPTOR = PointReceptor(
    time=dt.datetime(2023, 1, 1, 12, 0), longitude=-111.85, latitude=40.77, altitude=5.0
)


def _particle_dat(path: Path, last_minute: int) -> Path:
    """Write a PARTICLE_STILT.DAT of one particle, hourly, whose last row is *last_minute* before release."""
    times = [-1, *range(-60, -last_minute - 1, -60)]
    lines = ["header"] + [f"{t} 1 -111.9 40.7 10.0 0.0" for t in times]
    path.write_text("\n".join(lines) + "\n")
    return path


def _fake_hycs_std(directory: Path, body: str) -> Path:
    """Return a build directory whose ``hycs_std`` runs the shell commands in *body*."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "version").write_text("fake-1.0\n")  # a custom build names its version
    exe = directory / "hycs_std"
    exe.write_text("#!/usr/bin/env bash\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return directory


def _model_run(tmp_path: Path, build: Path, outdt: int):
    """Run the HYSPLIT model on fake met files with the fake ``hycs_std`` in *build*."""
    config = HysplitConfig(
        n_hours=N_HOURS,
        numpar=10,
        hnf_plume=False,
        varsiwant=VARS,
        outdt=outdt,
        exe_dir=build,
    )
    make_met_files(tmp_path / "met", RECEPTOR.time, N_HOURS)
    workdir = tmp_path / "run"
    workdir.mkdir()
    return HysplitModel().run(
        RECEPTOR,
        config,
        make_met_config(tmp_path / "met"),
        run_window(RECEPTOR.time, N_HOURS),
        workdir,
        timeout=30,
    )


@pytest.mark.parametrize("outdt", [0, 30])
def test_a_hycs_std_killed_part_way_fails_with_its_signal(tmp_path, outdt):
    """Its partial file is not read as a finished run, with ``outdt`` 0 or above."""
    partial = _particle_dat(tmp_path / "partial.dat", 120)
    build = _fake_hycs_std(
        tmp_path / "build", f"cp {partial} PARTICLE_STILT.DAT\nkill -KILL $$\n"
    )

    with pytest.raises(SimulationError) as caught:
        _model_run(tmp_path, build, outdt)

    assert caught.value.reason == FailureReason.EXIT_STATUS
    assert "SIGKILL" in str(caught.value)


@pytest.mark.parametrize("outdt", [0, 30])
def test_a_hycs_std_that_exits_1_part_way_fails_with_its_status(tmp_path, outdt):
    partial = _particle_dat(tmp_path / "partial.dat", 120)
    build = _fake_hycs_std(
        tmp_path / "build", f"cp {partial} PARTICLE_STILT.DAT\nexit 1\n"
    )

    with pytest.raises(SimulationError) as caught:
        _model_run(tmp_path, build, outdt)

    assert caught.value.reason == FailureReason.EXIT_STATUS
    assert "status 1" in str(caught.value)


def test_a_hycs_std_that_exits_0_with_the_full_run_is_complete(tmp_path):
    full = _particle_dat(tmp_path / "full.dat", abs(N_HOURS) * 60)
    build = _fake_hycs_std(tmp_path / "build", f"cp {full} PARTICLE_STILT.DAT\n")

    run = _model_run(tmp_path, build, outdt=0)

    assert run.particles["age"].min() == N_HOURS * 60


def test_a_hycs_std_that_exits_0_without_output_still_fails_for_that(tmp_path):
    build = _fake_hycs_std(tmp_path / "build", "echo 'nothing to do'\n")

    with pytest.raises(SimulationError) as caught:
        _model_run(tmp_path, build, outdt=0)

    assert caught.value.reason == FailureReason.NO_PARTICLE_DATA


def test_a_killed_simulation_stores_no_particles_and_a_rerun_runs_it_again(tmp_path):
    """The failure is recorded with its reason, and the next run does not skip it (#318)."""
    full = _particle_dat(tmp_path / "full.dat", abs(N_HOURS) * 60)
    partial = _particle_dat(tmp_path / "partial.dat", 120)
    flag = tmp_path / "kill-it"
    flag.touch()
    build = _fake_hycs_std(
        tmp_path / "build",
        f"if [ -e {flag} ]; then cp {partial} PARTICLE_STILT.DAT; kill -KILL $$; fi\n"
        f"cp {full} PARTICLE_STILT.DAT\n",
    )
    make_met_files(tmp_path / "met", RECEPTOR.time, N_HOURS)
    config = ProjectConfig(
        mets={"hrrr": make_met_config(tmp_path / "met")},
        n_hours=N_HOURS,
        numpar=10,
        hnf_plume=False,
        varsiwant=VARS,
        outdt=30,
        exe_dir=build,
        variants={"hrrr": {}},
    )
    project = Project.init(tmp_path / "project", config=config, receptors=[RECEPTOR])

    project.run(compute_root=tmp_path / "work")

    [row] = project.status().to_dict("records")
    assert (row["state"], row["reason"]) == ("failed", "EXIT_STATUS")
    assert "SIGKILL" in row["message"]
    assert not list((tmp_path / "project").glob("output/particles/**/*.parquet"))

    flag.unlink()
    project.run(compute_root=tmp_path / "work")

    [row] = project.status().to_dict("records")
    assert row["state"] == "complete"
    [path] = (tmp_path / "project").glob("output/particles/**/*.parquet")
    assert path.exists()
