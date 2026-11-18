"""Tests for stilt.execution.runner: shares of the receptors, the job array script, and submission."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from stilt.execution import runner
from stilt.execution.config import (
    ExecutionConfig,
    FetchConfig,
    SlurmConfig,
    slurm_minutes,
)
from stilt.execution.runner import _project_slug, job_script, task_share
from stilt.project import Project

from ..fixtures.factories import make_project_config

# ---------------------------------------------------------------------------
# The job array script
# ---------------------------------------------------------------------------


def _project_at(directory: Path, output: Path) -> Project:
    """A project in *directory* with an empty ``config.yaml``: the job script reads no setting."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text("")
    return Project(directory, output=output)


def _sbatch(script: str) -> dict[str, str | bool]:
    """The #SBATCH options of a script, a bare flag as True."""
    options: dict[str, str | bool] = {}
    for line in script.splitlines():
        if line.startswith("#SBATCH --"):
            key, _, value = line[len("#SBATCH --") :].partition("=")
            options[key] = value if value else True
    return options


def test_job_script_asks_for_what_the_execution_settings_say(tmp_path):
    project = _project_at(tmp_path / "My_Project", output=tmp_path / "out")
    folder = tmp_path / "My_Project" / "_slurm" / "stamp"
    execution = ExecutionConfig(
        backend="slurm",
        slurm={
            "n_workers": 50,
            "cpus": 4,
            "time": "02:00:00",
            "mem": "8G",
            "partition": "lin-np",
            "account": "lin-np",
            "qos": "lin-np",
            "array_parallelism": 10,
            "setup": ["module load gcc", "export OMP_NUM_THREADS=2"],
            "exclude": "notch345",
            "signal": "B:USR1@300",
            "exclusive": True,
        },
    )

    script = job_script(project, execution, folder, 7)

    assert script.startswith("#!/bin/bash\n")
    assert _sbatch(script) == {
        "job-name": "pystilt-my-project",
        "array": "0-6%10",
        "cpus-per-task": "4",
        "time": "02:00:00",  # as given
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
    header = lines.index("module load gcc") - 2
    assert lines[header].startswith('echo "$(date -u +%FT%TZ) $SLURMD_NODENAME job ')
    # The script's own OMP_NUM_THREADS comes before setup, so a setup line has the last word.
    assert lines.index("export OMP_NUM_THREADS=1") < lines.index("module load gcc")
    assert lines.index("module load gcc") < lines.index("export OMP_NUM_THREADS=2")
    last = lines[-1]
    assert last.startswith("exec ")
    assert f"-m stilt run {project.directory} " in last
    assert f"--receptors {folder / 'receptors.parquet'}" in last
    assert '--task "$SLURM_ARRAY_TASK_ID/7"' in last
    assert f"--execution {folder / 'execution.yaml'}" in last
    # A task runs its share here, so the script tells it the array's cpus.
    assert "--cpus 4" in last
    # Each task writes where the submitting process does, not to config.yaml's output.
    assert f"--output {(tmp_path / 'out').resolve()}" in last
    assert "--no-skip" not in script
    assert "--compute-root" not in script  # each node uses its own scratch


def test_job_script_leaves_out_what_was_not_set(tmp_path):
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
        tmp_path,
        1,
    )
    options = _sbatch(script)
    assert options["array"] == "0-0"
    assert options["signal"] == f"B:USR1@{runner.NOTICE_SECONDS}"
    for unset in ("mem", "partition", "account", "qos"):
        assert unset not in options


def test_job_script_runs_again_only_on_the_first_start_and_names_a_compute_root(
    tmp_path,
):
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
        tmp_path,
        2,
        skip_existing=False,
        compute_root=tmp_path / "scratch",
    )
    assert '[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && skip=""' in script
    assert script.splitlines()[-1].endswith(
        f"--compute-root {(tmp_path / 'scratch').resolve()} $skip"
    )


def _omp_threads_after_setup(script: str) -> str:
    """What OMP_NUM_THREADS is when the script reaches its last line, run in bash."""
    before_exec = script.rstrip("\n").rpartition("\n")[0]
    done = subprocess.run(
        ["bash", "-c", before_exec + '\necho "${OMP_NUM_THREADS-unset}"'],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ["PATH"]},
    )
    return done.stdout.splitlines()[-1]


def test_job_script_gives_each_worker_one_thread_of_the_tasks_cpus(tmp_path):
    # Four workers on four CPUs: without this a worker's BLAS sizes its pool by the node.
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(backend="slurm", slurm={"time": "01:00:00", "cpus": 4}),
        tmp_path,
        1,
    )
    assert "export OMP_NUM_THREADS=1" in script.splitlines()
    assert _omp_threads_after_setup(script) == "1"


def test_job_script_shares_cpus_a_slurm_option_adds_among_the_workers(tmp_path):
    # 2 workers on 8 CPUs (cpus-per-task replaced in `slurm`): 4 threads each.
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(
            backend="slurm", slurm={"time": "01:00:00", "cpus": 2, "cpus_per_task": 8}
        ),
        tmp_path,
        1,
    )
    assert _omp_threads_after_setup(script) == "4"


def test_job_script_gives_one_thread_when_cpus_per_task_is_not_a_number(tmp_path):
    # sbatch refuses the value; writing the script must not fail first.
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(
            backend="slurm", slurm={"time": "01:00:00", "cpus_per_task": "many"}
        ),
        tmp_path,
        1,
    )
    assert _omp_threads_after_setup(script) == "1"


def test_job_script_leaves_the_thread_count_to_a_setup_line(tmp_path):
    script = job_script(
        _project_at(tmp_path, output=tmp_path / "out"),
        ExecutionConfig(
            backend="slurm",
            slurm={
                "time": "01:00:00",
                "cpus": 4,
                "setup": ["export OMP_NUM_THREADS=3"],
            },
        ),
        tmp_path,
        1,
    )
    assert _omp_threads_after_setup(script) == "3"


def test_job_script_is_valid_bash(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(
        job_script(
            _project_at(tmp_path / "with space", output=tmp_path / "out"),
            ExecutionConfig(
                backend="slurm", slurm={"time": "01:00:00", "setup": ["echo ready"]}
            ),
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

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    # Run inside a Slurm job, the tests would see that job's cluster.
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    return SimpleNamespace(ran=ran, answers=answers)


def _out(stdout: str = "", returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_submit_writes_a_submission_and_hands_it_to_sbatch(pending, commands, tmp_path):
    import pandas as pd

    project = _hourly_project(tmp_path, 4)
    ids = list(project.receptors["receptor"])
    pending.extend([ids[2], ids[0], ids[3]])  # the order a run gives
    execution = ExecutionConfig(
        backend="slurm",
        timeout=600,
        slurm={"time": "01:00:00", "n_workers": 2, "cpus": 4},
    )
    commands.answers["sbatch"] = [_out("777;cluster\n")]

    job_id = runner.submit(project, execution=execution, skip_existing=False)

    assert job_id == "777"
    [(sbatch, parsable, script)] = commands.ran
    assert (sbatch, parsable) == ("sbatch", "--parsable")
    folder = Path(script).parent
    assert folder.parent == project.directory / "_slurm"
    assert (folder / "receptors.txt").read_text().split() == pending
    # The tasks read the checked rows, in the order they split them (#191).
    rows = pd.read_parquet(folder / "receptors.parquet")
    assert rows["receptor"].tolist() == pending
    assert {"kind", "location", "altitude_ref"} <= set(rows.columns)
    stored = yaml.safe_load((folder / "execution.yaml").read_text())
    assert ExecutionConfig.model_validate(stored) == execution
    text = Path(script).read_text()
    assert _sbatch(text)["array"] == "0-1"
    assert "--no-skip" in text


def test_submit_uses_no_more_tasks_than_receptors(pending, commands, tmp_path):
    project = _hourly_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    commands.answers["sbatch"] = [_out("5\n")]
    runner.submit(
        project,
        execution=ExecutionConfig(
            backend="slurm", slurm={"time": "01:00:00", "n_workers": 8}
        ),
    )
    assert _sbatch(Path(commands.ran[0][2]).read_text())["array"] == "0-0"


def test_submit_with_nothing_to_do_submits_nothing(pending, commands, tmp_path):
    assert (
        runner.submit(
            _project_at(tmp_path, output=tmp_path / "out"),
            execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
        )
        is None
    )
    assert commands.ran == []


def test_submit_says_why_sbatch_refused(pending, commands, tmp_path):
    project = _hourly_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    commands.answers["sbatch"] = [_out(returncode=1, stderr="Invalid account")]
    with pytest.raises(RuntimeError, match="Invalid account"):
        runner.submit(
            project,
            execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
        )


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
        states = runner.wait("9", poll=0)

    assert [c[0] for c in commands.ran] == ["sacct"] * 4
    warned = caplog.text
    assert "9_2 ended OUT_OF_MEMORY" in warned
    assert "9_3 ended CANCELLED" in warned
    assert "9_1" not in warned  # failed simulations are the status table's to report
    assert states["9_2"] == ("OUT_OF_MEMORY", "0:125")
    assert "9_0" not in warned


def test_a_job_on_another_cluster_is_followed_there(
    pending, commands, tmp_path, monkeypatch
):
    """``sbatch --parsable`` names the cluster when it is not the default (#187)."""
    import pandas as pd

    project = _hourly_project(tmp_path, 2)
    pending.extend(project.receptors["receptor"])
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": ["complete"]})
    )
    commands.answers["sbatch"] = [_out("777;kingspeak\n")]
    commands.answers["sacct"] = [
        _out("777_0|RUNNING|0:0\n"),
        _out("777_0|COMPLETED|0:0\n"),
    ]
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    execution = ExecutionConfig(
        backend="slurm", slurm={"time": "01:00:00", "clusters": "kingspeak"}
    )

    runner.run(project, execution=execution)

    sacct = [c for c in commands.ran if c[0] == "sacct"]
    assert len(sacct) == 2
    assert all(c[1:5] == ["-M", "kingspeak", "-j", "777"] for c in sacct)


def test_a_run_lists_once_more_when_the_files_look_late(
    pending, commands, tmp_path, monkeypatch
):
    """The submitting node can see new files a minute late on a network filesystem (#191)."""
    import pandas as pd

    project = _hourly_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    listings = iter([["pending"], ["complete"]])
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": next(listings)})
    )
    monkeypatch.setattr(runner, "wait", lambda job_id, cluster=None: {})
    slept: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    commands.answers["sbatch"] = [_out("5\n")]

    table = runner.run(
        project, execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"})
    )

    assert slept == [runner.RELIST_SECONDS]
    assert table["state"].tolist() == ["complete"]


def test_a_job_on_the_default_cluster_names_none(commands):
    commands.answers["sacct"] = [_out("9_0|COMPLETED|0:0\n")]
    runner.wait("9", poll=0)
    assert commands.ran[0][:3] == ["sacct", "-j", "9"]


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


def test_a_task_on_another_cluster_asks_and_requeues_there(
    tmp_path, monkeypatch, commands
):
    from stilt.execution import worker

    project = _hourly_project(tmp_path, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "kingspeak")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.setattr(worker, "run_receptors", lambda *a, **k: None)
    commands.answers["scontrol"] = [
        _out("JobId=4321 PreemptTime=2026-10-06T12:00:00 Requeue=1")
    ]

    runner.run(project, task=(0, 1))

    assert commands.ran == [
        ["scontrol", "-M", "kingspeak", "show", "job", "4321"],
        ["scontrol", "-M", "kingspeak", "requeue", "4321"],
    ]


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
        pytest.param(90, 90, id="90-as-int"),
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
    assert slurm_minutes(time) == minutes


@pytest.mark.parametrize("time", ["soon", "1:2:3:4", "-5", "0", "00:00:00", "1-"])
def test_a_time_limit_sbatch_would_not_take_is_an_error(time):
    with pytest.raises(ValueError, match="time limit"):
        ExecutionConfig(slurm={"time": time})


def test_a_slurm_run_needs_a_time_limit():
    with pytest.raises(ValueError, match="execution.slurm.time is required"):
        ExecutionConfig(backend="slurm")
    ExecutionConfig()  # a local run has none
    ExecutionConfig(backend="slurm", slurm={"time": "1-00"})


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

    from stilt.receptors import PointReceptor

    config = make_project_config(tmp_path)
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
    A share is taken before the complete receptors are dropped.

    Taken after, a task that starts late would shift onto another task's
    receptors.
    """
    project = _hourly_project(tmp_path, 7)
    every = list(dict.fromkeys(project.simulations["receptor"]))
    done: set[str] = set()
    monkeypatch.setattr(
        Project, "incomplete", lambda self, sel=None: sel[~sel["receptor"].isin(done)]
    )

    first = runner.pending(project, task=(0, 3))
    done.update(every[:4])  # some finish before tasks 1 and 2 start
    later = [runner.pending(project, task=(i, 3)) for i in (1, 2)]

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
    assert runner.pending(project, receptors=listed) == [every[1], every[0]]
    assert runner.pending(project, receptors=listed, skip_existing=False) == [
        every[3],
        every[1],
        every[0],
    ]
    assert runner.pending(
        project, receptors=listed, task=(1, 2), skip_existing=False
    ) == [every[1]]
    with pytest.raises(ValueError, match="not in this project"):
        runner.pending(project, receptors=["nope", every[0]])


def test_a_task_runs_here_whatever_the_backend(tmp_path, monkeypatch, commands):
    from stilt.execution import worker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    project = _hourly_project(tmp_path, 4)
    every = list(dict.fromkeys(project.simulations["receptor"]))
    ran: list[list[str]] = []
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **kw: ran.append(ids)
    )

    table = runner.run(
        project,
        task=(1, 2),
        execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
    )

    assert commands.ran == []  # no sbatch
    assert ran == [[every[1], every[3]]]
    assert list(table["receptor"]) == [every[1], every[3]]
    assert set(table["state"]) == {"pending"}  # nothing was written


# ---------------------------------------------------------------------------
# The fetch before the receptors run (#436)
# ---------------------------------------------------------------------------


def _downloading_project(tmp_path, n: int) -> Project:
    """A project like :func:`_hourly_project` whose met downloads HRRR."""
    import datetime as dt

    from stilt.receptors import PointReceptor
    from stilt.transport.hysplit import MetConfig

    config = make_project_config(
        tmp_path,
        mets={"hrrr": MetConfig(directory=tmp_path / "met", download="hrrr")},
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


@pytest.mark.parametrize("value", [True, None, {}])
def test_fetch_is_on_with_the_arrays_resources_unless_given(value):
    assert SlurmConfig.model_validate({"fetch": value}).fetch == FetchConfig()
    assert SlurmConfig().fetch == FetchConfig()


def test_fetch_false_turns_it_off_and_a_bad_fetch_setting_is_an_error():
    assert SlurmConfig.model_validate({"fetch": False}).fetch is False
    with pytest.raises(ValueError, match="partiton"):
        SlurmConfig.model_validate({"fetch": {"partiton": "dtn"}})
    with pytest.raises(ValueError, match="Not a Slurm time limit"):
        SlurmConfig.model_validate({"fetch": {"time": "soon"}})


def test_fetch_is_the_setting_of_the_backend_that_runs():
    """A local run's fetch is on or off; a Slurm run's also holds the fetch job's resources."""
    settings = {"local": {"fetch": False}, "slurm": {"time": "01:00:00"}}
    assert ExecutionConfig.model_validate(settings).fetch is False
    slurm = ExecutionConfig.model_validate({"backend": "slurm", **settings})
    assert slurm.fetch == FetchConfig()
    assert ExecutionConfig().fetch is True


def test_submit_sends_a_fetch_job_first_and_the_array_waits_for_it(
    pending, commands, tmp_path
):
    project = _downloading_project(tmp_path, 3)
    pending.extend(project.receptors["receptor"])
    execution = ExecutionConfig.model_validate(
        {
            "backend": "slurm",
            "slurm": {
                "time": "01:00:00",
                "account": "lin-np",
                "partition": "lin-np",
                "mem": "8G",
                "setup": ["module load gcc"],
                "exclude": "notch345",
                "fetch": {"partition": "notchpeak-dtn", "time": "12:00:00"},
            },
        }
    )
    commands.answers["sbatch"] = [_out("41\n"), _out("42\n")]

    job_id = runner.submit(project, execution=execution)

    assert job_id == "42"
    [fetch, array] = commands.ran
    assert fetch[:2] == ["sbatch", "--parsable"]
    assert Path(fetch[-1]).name == "fetch.sh"
    assert array == [
        "sbatch",
        "--parsable",
        "--dependency=afterany:41",
        str(Path(fetch[-1]).parent / "job.sh"),
    ]
    folder = Path(fetch[-1]).parent
    text = (folder / "fetch.sh").read_text()
    options = _sbatch(text)
    assert options["partition"] == "notchpeak-dtn"
    assert options["time"] == "12:00:00"
    assert options["account"] == "lin-np"  # the array's, unset in fetch
    assert options["mem"] == "8G"
    assert options["output"] == str(folder / "fetch.log")
    assert (folder / "fetch.id").read_text() == "41\n"
    assert "exclude" not in options  # the array's own sbatch options stay its own
    assert "array" not in options
    assert "module load gcc" in text
    exec_, _python, *command = text.splitlines()[-1].split()
    assert exec_ == "exec"
    assert command[:4] == ["-m", "stilt", "fetch", str(project.directory)]
    assert command[4:6] == ["--receptors", str(folder / "receptors.parquet")]
    assert command[6:8] == ["--output", str(project.output.directory)]
    assert "--no-skip" not in command
    subprocess.run(["bash", "-n", str(folder / "fetch.sh")], check=True)


def test_the_array_goes_to_the_cluster_its_fetch_job_went_to(
    pending, commands, tmp_path
):
    project = _downloading_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    execution = ExecutionConfig(
        backend="slurm", slurm={"time": "01:00:00", "clusters": "kingspeak,notchpeak"}
    )
    commands.answers["sbatch"] = [_out("41;kingspeak\n"), _out("42;kingspeak\n")]

    runner.submit(project, execution=execution)

    [fetch, array] = commands.ran
    assert _sbatch(Path(fetch[-1]).read_text())["clusters"] == "kingspeak,notchpeak"
    assert array[2:4] == ["--dependency=afterany:41", "--clusters=kingspeak"]


def test_a_met_without_download_gets_no_fetch_job(pending, commands, tmp_path):
    project = _hourly_project(tmp_path, 2)
    pending.extend(project.receptors["receptor"])
    commands.answers["sbatch"] = [_out("42\n")]

    runner.submit(
        project, execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"})
    )

    [array] = commands.ran
    assert Path(array[-1]).name == "job.sh"
    assert not (Path(array[-1]).parent / "fetch.sh").exists()


def test_fetch_false_sends_no_fetch_job(pending, commands, tmp_path):
    project = _downloading_project(tmp_path, 2)
    pending.extend(project.receptors["receptor"])
    commands.answers["sbatch"] = [_out("42\n")]

    runner.submit(
        project,
        execution=ExecutionConfig(
            backend="slurm", slurm={"time": "01:00:00", "fetch": False}
        ),
    )

    [array] = commands.ran
    assert Path(array[-1]).name == "job.sh"


def test_no_fetch_job_when_the_particles_are_there(
    pending, commands, tmp_path, monkeypatch
):
    """Footprints left to make need no met."""
    import pandas as pd

    project = _downloading_project(tmp_path, 2)
    pending.extend(project.receptors["receptor"])
    monkeypatch.setattr(
        Project,
        "_unrun",
        lambda self, sel=None, **_: pd.DataFrame(columns=["receptor"]),
    )
    commands.answers["sbatch"] = [_out("42\n")]

    runner.submit(
        project, execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"})
    )

    assert len(commands.ran) == 1


def test_a_local_run_on_several_cpus_fetches_before_the_workers_start(
    tmp_path, monkeypatch
):
    import pandas as pd

    from stilt.execution import worker

    project = _downloading_project(tmp_path, 3)
    steps: list[str] = []

    def fetch(project, unrun):
        steps.append(f"fetch {len(unrun)}")
        return pd.DataFrame({"met": [], "file": [], "message": []}, dtype="string")

    monkeypatch.setattr(runner, "fetch_project", fetch)
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **_: steps.append("run")
    )
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": ["pending"]})
    )

    runner.run(project, execution=ExecutionConfig(local={"cpus": 2}))
    assert steps == ["fetch 3", "run"]

    steps.clear()
    runner.run(project, execution=ExecutionConfig(local={"cpus": 1}))
    runner.run(project, execution=ExecutionConfig(local={"cpus": 2, "fetch": False}))
    assert steps == ["run", "run"]


def test_a_local_fetch_that_cannot_reach_the_archive_lets_the_run_go(
    tmp_path, monkeypatch, caplog
):
    import pandas as pd

    from stilt.execution import worker

    project = _downloading_project(tmp_path, 1)
    ran: list[str] = []

    def fetch(project, unrun):
        raise ConnectionError("archive busy")

    monkeypatch.setattr(runner, "fetch_project", fetch)
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **_: ran.append("run")
    )
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": ["pending"]})
    )

    with caplog.at_level("WARNING", logger="stilt.execution.runner"):
        runner.run(project, execution=ExecutionConfig(local={"cpus": 2}))

    assert ran == ["run"]
    assert "archive busy" in caplog.text


def test_the_fetch_job_takes_the_arrays_resources_and_none_of_its_other_options(
    pending, commands, tmp_path
):
    """With no ``fetch`` settings, the fetch job runs with the array's resources."""
    project = _downloading_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    execution = ExecutionConfig(
        backend="slurm",
        slurm={
            "account": "lin-np",
            "partition": "lin-np",
            "time": "24:00:00",
            "constraint": "skl",
        },
    )
    commands.answers["sbatch"] = [_out("41\n"), _out("42\n")]

    runner.submit(project, execution=execution)

    [fetch, array] = commands.ran
    options = _sbatch(Path(fetch[-1]).read_text())
    assert options["account"] == "lin-np"
    assert options["partition"] == "lin-np"
    assert options["time"] == "24:00:00"  # the array's, as job.sh has it
    assert "constraint" not in options  # another sbatch option is the array's alone
    assert _sbatch(Path(array[-1]).read_text())["time"] == "24:00:00"


def test_a_no_skip_submission_fetches_for_every_run_it_makes_again(
    pending, commands, tmp_path, monkeypatch
):
    import pandas as pd

    project = _downloading_project(tmp_path, 2)
    pending.extend(project.receptors["receptor"])
    # Every particles file exists: only a rerun needs the met.
    real_unrun = Project._unrun

    def unrun(self, sel=None, *, skip_existing=True):
        if skip_existing:
            return pd.DataFrame(columns=["receptor"])
        return real_unrun(self, sel, skip_existing=False)

    monkeypatch.setattr(Project, "_unrun", unrun)
    commands.answers["sbatch"] = [_out("41\n"), _out("42\n")]

    runner.submit(
        project,
        execution=ExecutionConfig(backend="slurm", slurm={"time": "01:00:00"}),
        skip_existing=False,
    )

    [fetch, array] = commands.ran
    assert Path(fetch[-1]).name == "fetch.sh"
    assert Path(fetch[-1]).read_text().splitlines()[-1].endswith(" --no-skip")


def test_a_refused_array_cancels_its_fetch_job(pending, commands, tmp_path):
    project = _downloading_project(tmp_path, 1)
    pending.extend(project.receptors["receptor"])
    execution = ExecutionConfig(
        backend="slurm", slurm={"time": "01:00:00", "clusters": "kingspeak,notchpeak"}
    )
    commands.answers["sbatch"] = [
        _out("41;notchpeak\n"),
        _out(returncode=1, stderr="invalid partition specified"),
    ]

    with pytest.raises(RuntimeError, match="invalid partition"):
        runner.submit(project, execution=execution)

    assert commands.ran[-1] == ["scancel", "-M", "notchpeak", "41"]


def test_a_local_fetch_raises_an_error_a_retry_would_repeat(tmp_path, monkeypatch):
    from stilt.execution import worker

    project = _downloading_project(tmp_path, 1)
    ran: list[str] = []

    def fetch(project, unrun):
        raise PermissionError(13, "Permission denied", str(tmp_path / "met"))

    monkeypatch.setattr(runner, "fetch_project", fetch)
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **_: ran.append("run")
    )

    with pytest.raises(PermissionError):
        runner.run(project, execution=ExecutionConfig(local={"cpus": 2}))
    assert ran == []


def test_a_local_fetch_lets_the_run_go_after_a_dropped_ftp_connection(
    tmp_path, monkeypatch, caplog
):
    """A dropped FTP connection is an EOFError from ftplib, not an OSError."""
    import ftplib

    import pandas as pd

    from stilt.execution import worker

    project = _downloading_project(tmp_path, 1)
    ran: list[str] = []

    def fetch(project, unrun):
        ftp = ftplib.FTP()
        ftp.file = SimpleNamespace(readline=lambda size: "")  # a closed connection
        ftp.getline()  # raises EOFError from inside ftplib

    monkeypatch.setattr(runner, "fetch_project", fetch)
    monkeypatch.setattr(
        worker, "run_receptors", lambda project, ids, **_: ran.append("run")
    )
    monkeypatch.setattr(
        runner, "_status", lambda project, ids: pd.DataFrame({"state": ["pending"]})
    )

    with caplog.at_level("WARNING", logger="stilt.execution.runner"):
        runner.run(project, execution=ExecutionConfig(local={"cpus": 2}))

    assert ran == ["run"]
    assert "The fetch stopped" in caplog.text
