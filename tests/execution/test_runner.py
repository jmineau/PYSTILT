"""Tests for runtraj.execution.runner: shares of the releases, the job array script, and submission."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from runtraj.execution import runner
from runtraj.execution.config import (
    ExecutionConfig,
)
from runtraj.project import Project

from ..fixtures.factories import make_project_config

# ---------------------------------------------------------------------------
# The job array script
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Submitting to Slurm
# ---------------------------------------------------------------------------


@pytest.fixture
def pending(monkeypatch):
    """Set which releases the runner finds incomplete."""
    ids: list[str] = []
    monkeypatch.setattr(runner, "pending", lambda project, **_: list(ids))
    return ids


@pytest.fixture
def commands(monkeypatch):
    """Record the commands the runner runs; ``answers`` maps a program to its outputs in turn."""
    ran: list[list[str]] = []
    answers: dict[str, list[SimpleNamespace]] = {}

    def fake_run(args, **kwargs):
        ran.append(list(args))
        queue = answers.get(args[0], [])
        return (
            queue.pop(0)
            if queue
            else SimpleNamespace(returncode=0, stdout="", stderr="")
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Run inside a Slurm job, the tests would see that job's cluster.
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    return SimpleNamespace(ran=ran, answers=answers)


def _out(stdout: str = "", returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# A task that is told to stop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Time limits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A share of the releases: --releases and --task
# ---------------------------------------------------------------------------


def _hourly_project(tmp_path, n: int) -> Project:
    import datetime as dt

    from runtraj.releases import PointRelease

    config = make_project_config(tmp_path)
    releases = [
        PointRelease(
            time=dt.datetime(2023, 1, 1, h),
            longitude=-111.85,
            latitude=40.77,
            altitude=5.0,
        )
        for h in range(n)
    ]
    return Project.init(tmp_path / "hourly", config=config, releases=releases)


# ---------------------------------------------------------------------------
# The fetch before the releases run (#436)
# ---------------------------------------------------------------------------


def _downloading_project(tmp_path, n: int) -> Project:
    """A project like :func:`_hourly_project` whose met downloads HRRR."""
    import datetime as dt

    from runtraj.models.hysplit import HysplitMetConfig
    from runtraj.releases import PointRelease

    config = make_project_config(
        tmp_path,
        mets={"hrrr": HysplitMetConfig(directory=tmp_path / "met", download="hrrr")},
    )
    releases = [
        PointRelease(
            time=dt.datetime(2023, 1, 1, h),
            longitude=-111.85,
            latitude=40.77,
            altitude=5.0,
        )
        for h in range(n)
    ]
    return Project.init(tmp_path / "hourly", config=config, releases=releases)


def test_no_fetch_job_when_the_particles_are_there(
    pending, commands, tmp_path, monkeypatch
):
    """Footprints left to make need no met."""
    import pandas as pd

    project = _downloading_project(tmp_path, 2)
    pending.extend(project.releases["release"])
    monkeypatch.setattr(
        Project,
        "_unrun",
        lambda self, sel=None, **_: pd.DataFrame(columns=["release"]),
    )
    commands.answers["sbatch"] = [_out("42\n")]

    runner.submit(
        project, execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"})
    )

    # One sbatch, the array; runtraj also asks scontrol for MaxArraySize.
    assert [args[0] for args in commands.ran].count("sbatch") == 1
