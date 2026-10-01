"""Tests for stilt.execution.runner: batching, the Slurm settings, and submission."""

from __future__ import annotations

from pathlib import Path

import pytest

from stilt.config import ExecutionConfig
from stilt.execution import Batch, runner
from stilt.execution.runner import slurm_parameters, split
from stilt.project import Project

# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_is_round_robin_and_never_makes_an_empty_batch():
    assert split(list("abcdefg"), 3) == [["a", "d", "g"], ["b", "e"], ["c", "f"]]
    assert split(["a", "b"], 5) == [["a"], ["b"]]
    assert split(["a", "b"], 1) == [["a", "b"]]


# ---------------------------------------------------------------------------
# slurm_parameters
# ---------------------------------------------------------------------------


def test_slurm_parameters_pass_only_what_was_set():
    params = slurm_parameters(
        ExecutionConfig(backend="slurm", n_workers=4), job_name="pystilt-x"
    )
    assert params == {
        "slurm_job_name": "pystilt-x",
        "slurm_cpus_per_task": 1,
        "slurm_additional_parameters": {"requeue": True},
    }


def test_slurm_parameters_map_every_setting():
    execution = ExecutionConfig(
        backend="slurm",
        n_workers=100,
        cpus=4,
        time="02:00:00",
        mem="8G",
        partition="compute",
        account="lab",
        qos="normal",
        array_parallelism=25,
        setup=["module load hysplit"],
        slurm={"exclude": "node1", "no_kill": True, "requeue": False},
    )
    assert slurm_parameters(execution, job_name="pystilt-x") == {
        "slurm_job_name": "pystilt-x",
        "slurm_cpus_per_task": 4,
        "slurm_time": 120,
        "slurm_mem": "8G",
        "slurm_partition": "compute",
        "slurm_account": "lab",
        "slurm_qos": "normal",
        "slurm_array_parallelism": 25,
        "slurm_setup": ["module load hysplit"],
        # Underscores become the hyphens sbatch spells, and an explicit
        # requeue setting wins over the default.
        "slurm_additional_parameters": {
            "exclude": "node1",
            "no-kill": True,
            "requeue": False,
        },
    }


def test_slurm_parameters_render_as_a_submission_script(tmp_path, monkeypatch):
    """submitit accepts the parameters and writes the sbatch lines they stand for."""
    import submitit

    # submitit refuses to build a Slurm executor where it cannot find srun.
    monkeypatch.setattr(submitit.SlurmExecutor, "affinity", classmethod(lambda cls: 1))

    execution = ExecutionConfig(
        backend="slurm",
        cpus=2,
        time="00:10:00",
        mem="4G",
        partition="compute",
        array_parallelism=2,
        slurm={"exclude": "node1"},
    )
    executor = submitit.AutoExecutor(folder=tmp_path, cluster="slurm")
    executor.update_parameters(**slurm_parameters(execution, job_name="pystilt-x"))
    script = executor._executor._make_submission_file_text("CMD", "uid")  # type: ignore[attr-defined]

    for line in (
        "#SBATCH --job-name=pystilt-x",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --time=10",
        "#SBATCH --mem=4G",
        "#SBATCH --partition=compute",
        "#SBATCH --exclude=node1",
        "#SBATCH --requeue",
    ):
        assert line in script


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def test_batch_opens_the_project_and_runs_its_receptors(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_run_receptors(project, receptor_ids, **kwargs):
        calls.append({"project": project, "ids": receptor_ids, **kwargs})
        return ["results"]

    monkeypatch.setattr("stilt.project.Project", lambda path: f"Project({path})")
    monkeypatch.setattr("stilt.execution.worker.run_receptors", fake_run_receptors)

    batch = Batch(
        str(tmp_path),
        ["a", "b"],
        compute_root="/scratch/x",
        cpus=2,
        skip_existing=False,
    )
    assert batch() == ["results"]
    assert calls == [
        {
            "project": f"Project({tmp_path})",
            "ids": ["a", "b"],
            "compute_root": "/scratch/x",
            "n_cores": 2,
            "skip_existing": False,
        }
    ]


def test_batch_checkpoint_resubmits_itself_keeping_what_finished(tmp_path):
    """A preempted or timed-out task runs again and skips its finished receptors."""
    batch = Batch(
        str(tmp_path), ["a", "b"], compute_root="/s", cpus=2, skip_existing=False
    )

    again = batch.checkpoint().function

    assert isinstance(again, Batch)
    assert (again.project, again.receptor_ids) == (str(tmp_path), ["a", "b"])
    assert (again.compute_root, again.cpus) == ("/s", 2)
    assert again.skip_existing is True


def test_batch_survives_pickling(tmp_path):
    import pickle

    batch = Batch(str(tmp_path), ["a"], cpus=3)
    back = pickle.loads(pickle.dumps(batch))
    assert (back.project, back.receptor_ids, back.cpus) == (str(tmp_path), ["a"], 3)


# ---------------------------------------------------------------------------
# Submitting to Slurm
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, job_id: str, error: Exception | None = None) -> None:
        self.job_id = job_id
        self.error = error
        self.waited = False
        self.paths = type("Paths", (), {"folder": Path("/logs")})()

    def wait(self) -> None:
        self.waited = True

    def exception(self) -> Exception | None:
        return self.error

    def result(self) -> list[str]:
        return [f"result of {self.job_id}"]


class _FakeExecutor:
    """Stands in for submitit.AutoExecutor: records what would be submitted."""

    instances: list[_FakeExecutor] = []

    def __init__(self, folder, cluster=None):
        self.folder, self.cluster = Path(folder), cluster
        self.parameters: dict = {}
        self.submitted: list[Batch] = []
        self.in_batch = False
        _FakeExecutor.instances.append(self)

    def update_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def batch(self):
        executor = self

        class _Context:
            def __enter__(self):
                executor.in_batch = True

            def __exit__(self, *exc):
                executor.in_batch = False

        return _Context()

    def submit(self, fn):
        assert self.in_batch, "tasks must be submitted as one array"
        self.submitted.append(fn)
        return _FakeJob(f"777_{len(self.submitted) - 1}")


@pytest.fixture
def fake_submitit(monkeypatch):
    import submitit

    _FakeExecutor.instances = []
    monkeypatch.setattr(submitit, "AutoExecutor", _FakeExecutor)
    return _FakeExecutor


@pytest.fixture
def pending(monkeypatch):
    """Set which receptors the runner finds incomplete."""
    ids: list[str] = []
    monkeypatch.setattr(runner, "_pending", lambda project, skip_existing: list(ids))
    return ids


def test_submit_sends_one_array_of_batches(fake_submitit, pending, tmp_path):
    pending.extend(["a", "b", "c"])
    project = Project(tmp_path / "my_project")
    execution = ExecutionConfig(
        backend="slurm", n_workers=2, cpus=4, partition="compute"
    )

    jobs = runner.submit(project, execution=execution, skip_existing=False)

    [executor] = fake_submitit.instances
    assert executor.cluster == "slurm"
    assert executor.folder.parent == project.directory / "slurm"
    assert executor.parameters["slurm_job_name"] == "pystilt-my-project"
    assert executor.parameters["slurm_partition"] == "compute"
    assert [b.receptor_ids for b in executor.submitted] == [["a", "c"], ["b"]]
    for batch in executor.submitted:
        assert batch.project == str(project.directory)
        assert batch.cpus == 4
        assert batch.skip_existing is False
        assert batch.compute_root is None  # each node resolves its own scratch
    assert [job.job_id for job in jobs] == ["777_0", "777_1"]


def test_submit_passes_an_explicit_compute_root(fake_submitit, pending, tmp_path):
    pending.append("a")
    runner.submit(
        Project(tmp_path),
        execution=ExecutionConfig(backend="slurm"),
        compute_root=tmp_path / "scratch",
    )
    [batch] = fake_submitit.instances[0].submitted
    assert batch.compute_root == str((tmp_path / "scratch").resolve())


def test_submit_with_nothing_to_do_submits_nothing(fake_submitit, pending, tmp_path):
    jobs = runner.submit(Project(tmp_path), execution=ExecutionConfig(backend="slurm"))
    assert jobs == []
    assert fake_submitit.instances == []


def test_waiting_raises_when_a_task_did_not_complete():
    good, bad = _FakeJob("9_0"), _FakeJob("9_1", error=RuntimeError("timed out"))

    assert runner._wait([good]) == ["result of 9_0"]  # type: ignore[list-item]
    assert good.waited

    with pytest.raises(
        RuntimeError, match=r"1 of 2 Slurm tasks did not complete. Task 9_1: timed out"
    ):
        runner._wait([good, bad])  # type: ignore[list-item]
    assert bad.waited


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
    assert "slurm_time" not in slurm_parameters(ExecutionConfig(), job_name="x")
