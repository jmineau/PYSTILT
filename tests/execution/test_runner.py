"""Tests for stilt.execution.runner: shares of the receptors, the job array script, and submission."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from stilt.execution import runner
from stilt.execution.config import ExecutionConfig
from stilt.execution.runner import _project_slug, job_script, task_share
from stilt.project import Project

# ---------------------------------------------------------------------------
# The job array script
# ---------------------------------------------------------------------------


def _sbatch(script: str) -> dict[str, str | bool]:
    """The #SBATCH options of a script, a bare flag as True."""
    options: dict[str, str | bool] = {}
    for line in script.splitlines():
        if line.startswith("#SBATCH --"):
            key, _, value = line[len("#SBATCH --") :].partition("=")
            options[key] = value if value else True
    return options


def test_job_script_asks_for_what_the_execution_settings_say(tmp_path):
    project = Project(tmp_path / "My_Project")
    folder = tmp_path / "My_Project" / "_slurm" / "stamp"
    execution = ExecutionConfig(
        backend="slurm",
        n_workers=50,
        cpus=4,
        time="02:00:00",
        mem="8G",
        partition="lin-np",
        account="lin-np",
        qos="lin-np",
        array_parallelism=10,
        setup=["module load gcc", "export OMP_NUM_THREADS=1"],
        slurm={"exclude": "notch345", "signal": "B:USR1@300", "exclusive": True},
    )

    script = job_script(project, execution, folder, 7)

    assert script.startswith("#!/bin/bash\n")
    assert _sbatch(script) == {
        "job-name": "pystilt-my-project",
        "array": "0-6%10",
        "cpus-per-task": "4",
        "time": "120",
        "mem": "8G",
        "partition": "lin-np",
        "account": "lin-np",
        "qos": "lin-np",
        "output": str(folder / "%a.log"),
        "open-mode": "append",
        "requeue": True,
        "signal": "B:USR1@300",  # the slurm options replace the defaults
        "exclude": "notch345",
        "exclusive": True,
    }
    lines = script.splitlines()
    # The log says where and when the task ran, before any setup can fail.
    header = lines.index("module load gcc") - 1
    assert lines[header].startswith('echo "$(date -u +%FT%TZ) $SLURMD_NODENAME job ')
    assert lines.index("module load gcc") < lines.index("export OMP_NUM_THREADS=1")
    last = lines[-1]
    assert last.startswith("exec ")
    assert f"-m stilt run {project.directory} " in last
    assert f"--receptors {folder / 'receptors.txt'}" in last
    assert '--task "$SLURM_ARRAY_TASK_ID/7"' in last
    assert f"--execution {folder / 'execution.yaml'}" in last
    assert "--no-skip" not in script
    assert "--compute-root" not in script  # each node uses its own scratch


def test_job_script_leaves_out_what_was_not_set(tmp_path):
    script = job_script(
        Project(tmp_path), ExecutionConfig(backend="slurm"), tmp_path, 1
    )
    options = _sbatch(script)
    assert options["array"] == "0-0"
    assert options["signal"] == f"B:USR1@{runner.NOTICE_SECONDS}"
    for unset in ("time", "mem", "partition", "account", "qos"):
        assert unset not in options


def test_job_script_runs_again_only_on_the_first_start_and_names_a_compute_root(
    tmp_path,
):
    script = job_script(
        Project(tmp_path),
        ExecutionConfig(backend="slurm"),
        tmp_path,
        2,
        skip_existing=False,
        compute_root=tmp_path / "scratch",
    )
    assert '[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && skip=""' in script
    assert script.splitlines()[-1].endswith(
        f"--compute-root {(tmp_path / 'scratch').resolve()} $skip"
    )


def test_job_script_is_valid_bash(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(
        job_script(
            Project(tmp_path / "with space"),
            ExecutionConfig(backend="slurm", setup=["echo ready"]),
            tmp_path / "with space" / "_slurm",
            3,
            skip_existing=False,
        )
    )
    subprocess.run(["bash", "-n", str(script)], check=True)


# ---------------------------------------------------------------------------
# Submitting to Slurm
# ---------------------------------------------------------------------------


@pytest.fixture
def pending(monkeypatch):
    """Set which receptors the runner finds incomplete."""
    ids: list[str] = []
    monkeypatch.setattr(
        runner, "_pending", lambda project, skip_existing, *_: list(ids)
    )
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

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    return SimpleNamespace(ran=ran, answers=answers)


def _out(stdout: str = "", returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_submit_writes_a_submission_and_hands_it_to_sbatch(pending, commands, tmp_path):
    pending.extend(["a", "b", "c"])
    project = Project(tmp_path / "my_project")
    execution = ExecutionConfig(backend="slurm", n_workers=2, cpus=4, timeout=600)
    commands.answers["sbatch"] = [_out("777;cluster\n")]

    job_id = runner.submit(project, execution=execution, skip_existing=False)

    assert job_id == "777"
    [(sbatch, parsable, script)] = commands.ran
    assert (sbatch, parsable) == ("sbatch", "--parsable")
    folder = Path(script).parent
    assert folder.parent == project.directory / "_slurm"
    assert (folder / "receptors.txt").read_text().split() == ["a", "b", "c"]
    stored = yaml.safe_load((folder / "execution.yaml").read_text())
    assert ExecutionConfig.model_validate(stored) == execution
    text = Path(script).read_text()
    assert _sbatch(text)["array"] == "0-1"
    assert "--no-skip" in text


def test_submit_uses_no_more_tasks_than_receptors(pending, commands, tmp_path):
    pending.append("a")
    commands.answers["sbatch"] = [_out("5\n")]
    runner.submit(
        Project(tmp_path), execution=ExecutionConfig(backend="slurm", n_workers=8)
    )
    assert _sbatch(Path(commands.ran[0][2]).read_text())["array"] == "0-0"


def test_submit_with_nothing_to_do_submits_nothing(pending, commands, tmp_path):
    assert (
        runner.submit(Project(tmp_path), execution=ExecutionConfig(backend="slurm"))
        is None
    )
    assert commands.ran == []


def test_submit_says_why_sbatch_refused(pending, commands, tmp_path):
    pending.append("a")
    commands.answers["sbatch"] = [_out(returncode=1, stderr="Invalid account")]
    with pytest.raises(RuntimeError, match="Invalid account"):
        runner.submit(Project(tmp_path), execution=ExecutionConfig(backend="slurm"))


def test_waiting_polls_sacct_until_every_task_ends(commands, caplog):
    commands.answers["sacct"] = [
        _out(returncode=1),  # the accounting database has not heard of it yet
        _out("9_0|RUNNING|0:0\n9_[1-3]|PENDING|0:0\n"),
        _out("9_0|COMPLETED|0:0\n9_1|REQUEUED|0:0\n9_2|RUNNING|0:0\n9_3|RUNNING|0:0\n"),
        _out(
            "9_0|COMPLETED|0:0\n9_1|FAILED|1:0\n9_2|OUT_OF_MEMORY|0:125\n"
            "9_3|CANCELLED by 123|0:15\n"
        ),
    ]
    with caplog.at_level("WARNING", logger="stilt.execution.runner"):
        runner._wait("9", poll=0)

    assert [c[0] for c in commands.ran] == ["sacct"] * 4
    warned = caplog.text
    assert "9_2 ended OUT_OF_MEMORY" in warned
    assert "9_3 ended CANCELLED" in warned
    assert "9_1" not in warned  # failed simulations are the status table's to report
    assert "9_0" not in warned


# ---------------------------------------------------------------------------
# A task that is told to stop
# ---------------------------------------------------------------------------


def test_a_task_told_to_stop_requeues_itself(tmp_path, monkeypatch, commands):
    from stilt.execution import worker

    project = _hourly_project(tmp_path, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)

    def stopped(*args, **kwargs):
        signal.raise_signal(signal.SIGUSR1)

    def told_twice(*args, **kwargs):
        try:
            stopped()
        finally:
            signal.raise_signal(signal.SIGUSR1)  # a second one only adds to the record

    monkeypatch.setattr(worker, "run_receptors", told_twice)
    table = runner.run(project, task=(0, 1))

    assert commands.ran == [["scontrol", "requeue", "4321"]]
    assert set(table["state"]) == {"pending"}
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL


def test_a_task_requeues_itself_only_so_many_times(tmp_path, monkeypatch, commands):
    from stilt.execution import worker

    project = _hourly_project(tmp_path, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.setenv("SLURM_RESTART_COUNT", str(runner.MAX_REQUEUES))
    monkeypatch.setattr(
        worker, "run_receptors", lambda *a, **k: signal.raise_signal(signal.SIGUSR1)
    )
    runner.run(project, task=(0, 1))
    assert commands.ran == []


def test_a_preempted_task_requeues_itself_and_a_cancelled_one_does_not(
    tmp_path, monkeypatch, commands
):
    """A preempted task gets SIGTERM, and maybe no SIGUSR1 before its grace time ends."""
    from stilt.execution import worker

    project = _hourly_project(tmp_path, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    # run_receptors stops on SIGTERM and returns with the receptor unfinished.
    monkeypatch.setattr(worker, "run_receptors", lambda *a, **k: None)

    commands.answers["scontrol"] = [
        _out("JobId=4321 PreemptTime=2026-10-06T12:00:00 Requeue=1")
    ]
    runner.run(project, task=(0, 1))
    assert commands.ran == [
        ["scontrol", "show", "job", "4321"],
        ["scontrol", "requeue", "4321"],
    ]

    commands.ran.clear()
    commands.answers["scontrol"] = [_out("JobId=4321 PreemptTime=None Requeue=1")]
    runner.run(project, task=(0, 1))  # scancel: not preempted
    assert commands.ran == [["scontrol", "show", "job", "4321"]]


def test_a_task_that_finished_is_not_requeued(tmp_path, monkeypatch, commands):
    import pandas as pd

    from stilt.execution import worker

    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.setattr(
        worker, "run_receptors", lambda *a, **k: signal.raise_signal(signal.SIGUSR1)
    )
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": ["complete"]})
    )
    runner.run(_hourly_project(tmp_path, 1), task=(0, 1))
    assert commands.ran == []


def test_an_interrupt_that_is_not_the_notice_is_not_swallowed(
    tmp_path, monkeypatch, commands
):
    from stilt.execution import worker

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_receptors", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(_hourly_project(tmp_path, 1), task=(0, 1))
    assert commands.ran == []


# ---------------------------------------------------------------------------
# Time limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time", "minutes"),
    [
        (90, 90),
        ("90", 90),
        ("30:00", 30),
        ("02:00:00", 120),
        ("00:10:30", 11),  # rounded up
        ("1-00", 1440),
        ("1-12:30", 2190),
        ("2-00:00:00", 2880),
    ],
)
def test_time_limits_are_read_the_way_sbatch_reads_them(time, minutes):
    assert ExecutionConfig(time=time).time_minutes == minutes


@pytest.mark.parametrize("time", ["soon", "1:2:3:4", "-5", "0", "00:00:00", "1-"])
def test_a_time_limit_sbatch_would_not_take_is_an_error(time):
    with pytest.raises(ValueError, match="time limit"):
        ExecutionConfig(time=time)


def test_no_time_limit_is_left_to_the_partition():
    assert ExecutionConfig().time_minutes is None


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("/data/projects/My_Project", "my-project"),
        ("/data/projects/My_Project/", "my-project"),
        ("/data/Weird  Name!!", "weird-name"),
        ("", "project"),
    ],
)
def test_project_slug_names_the_slurm_job(directory, expected):
    assert _project_slug(directory) == expected


# ---------------------------------------------------------------------------
# A share of the receptors: --receptors and --task
# ---------------------------------------------------------------------------


def _hourly_project(tmp_path, n: int) -> Project:
    import datetime as dt

    from stilt.config import ProjectConfig
    from stilt.receptors import PointReceptor

    config = ProjectConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
        variants={"hrrr": {}},
    )
    receptors = [
        PointReceptor(
            time=dt.datetime(2023, 1, 1, h),
            longitude=-111.85,
            latitude=40.77,
            altitude=5.0,
        )
        for h in range(n)
    ]
    return Project.init(tmp_path / "hourly", config=config, receptors=receptors)


def test_task_share_is_every_nth_receptor():
    ids = list("abcdefg")
    assert [task_share(ids, i, 3) for i in range(3)] == [
        ["a", "d", "g"],
        ["b", "e"],
        ["c", "f"],
    ]
    assert task_share(ids[:1], 2, 3) == []
    for task, n in [(3, 3), (-1, 3), (0, 0)]:
        with pytest.raises(ValueError, match="0 to n_tasks - 1"):
            task_share(ids, task, n)


def test_tasks_split_the_receptors_the_same_way_whenever_each_starts(
    tmp_path, monkeypatch
):
    """
    A share is taken before the complete receptors are dropped. Taken after,
    a task that starts late would shift onto another task's receptors.
    """
    project = _hourly_project(tmp_path, 7)
    every = list(dict.fromkeys(project.simulations["receptor"]))
    done: set[str] = set()
    monkeypatch.setattr(
        Project, "incomplete", lambda self, sel=None: sel[~sel["receptor"].isin(done)]
    )

    first = runner._pending(project, True, task=(0, 3))
    done.update(every[:4])  # some finish before tasks 1 and 2 start
    later = [runner._pending(project, True, task=(i, 3)) for i in (1, 2)]

    shares = [every[i::3] for i in range(3)]
    assert first == shares[0]
    assert later == [[r for r in shares[i] if r not in done] for i in (1, 2)]


def test_a_receptor_list_limits_the_run_and_keeps_its_order(tmp_path, monkeypatch):
    project = _hourly_project(tmp_path, 4)
    every = list(dict.fromkeys(project.simulations["receptor"]))
    monkeypatch.setattr(
        Project, "incomplete", lambda self, sel=None: sel[sel["receptor"] != every[3]]
    )
    listed = [every[3], every[1], every[1], every[0]]
    assert runner._pending(project, True, listed) == [every[1], every[0]]
    assert runner._pending(project, False, listed) == [every[3], every[1], every[0]]
    assert runner._pending(project, False, listed, (1, 2)) == [every[1]]
    with pytest.raises(ValueError, match="not in this project"):
        runner._pending(project, True, ["nope", every[0]])


def test_a_task_runs_here_whatever_the_backend(tmp_path, monkeypatch, commands):
    from stilt.execution import worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    project = _hourly_project(tmp_path, 4)
    every = list(dict.fromkeys(project.simulations["receptor"]))
    ran: list[list[str]] = []
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **kw: ran.append(ids)
    )

    table = runner.run(project, task=(1, 2), execution=ExecutionConfig(backend="slurm"))

    assert commands.ran == []  # no sbatch
    assert ran == [[every[1], every[3]]]
    assert list(table["receptor"]) == [every[1], every[3]]
    assert set(table["state"]) == {"pending"}  # nothing was written
