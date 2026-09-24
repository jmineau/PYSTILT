"""Tests for execution backend launchers."""

import subprocess
import sys
import threading
import types

import pytest

from stilt.execution import (
    KubernetesExecutor,
    KubernetesHandle,
    LocalExecutor,
    LocalHandle,
    SlurmExecutor,
    SlurmHandle,
    get_executor,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# LocalExecutor
# ---------------------------------------------------------------------------


class _FakeModel:
    """Records the constructor arguments LocalExecutor passes to Model."""

    def __init__(self, project, compute_root=None):
        self.project = project
        self.compute_root = compute_root


@pytest.fixture
def local_calls(monkeypatch):
    """
    Capture the run_simulations call made on the executor's worker thread.

    ``local.py`` imports ``Model`` from ``stilt.model`` and ``run_simulations``
    from ``stilt.execution.worker`` lazily inside the thread, so patching those
    module attributes is enough.
    """
    calls: list[dict] = []

    def fake_run_simulations(model, sim_ids, *, n_cores=1, skip_existing=None):
        calls.append(
            {
                "model": model,
                "sim_ids": sim_ids,
                "n_cores": n_cores,
                "skip_existing": skip_existing,
            }
        )

    monkeypatch.setattr("stilt.model.Model", _FakeModel)
    monkeypatch.setattr("stilt.execution.worker.run_simulations", fake_run_simulations)
    return calls


def test_local_executor_start_runs_simulations_on_worker_thread(tmp_path, local_calls):
    """start() builds a Model from the project root and runs the pending ids."""
    ex = LocalExecutor(n_workers=1)
    handle = ex.start(
        ["sim-a", "sim-b"],
        project=str(tmp_path),
        compute_root="/scratch/pystilt",
        skip_existing=False,
    )
    handle.wait()

    assert isinstance(handle, LocalHandle)
    assert handle.done
    [call] = local_calls
    assert isinstance(call["model"], _FakeModel)
    assert call["model"].project == str(tmp_path)
    assert call["model"].compute_root == "/scratch/pystilt"
    assert call["sim_ids"] == ["sim-a", "sim-b"]
    assert call["n_cores"] == 1
    assert call["skip_existing"] is False


def test_local_executor_start_returns_before_work_finishes(tmp_path, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def blocking_run_simulations(model, sim_ids, *, n_cores=1, skip_existing=None):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr("stilt.model.Model", _FakeModel)
    monkeypatch.setattr(
        "stilt.execution.worker.run_simulations", blocking_run_simulations
    )

    handle = LocalExecutor(n_workers=1).start(["sim-a"], project=str(tmp_path))

    assert started.wait(timeout=5)
    assert not handle.done
    release.set()
    handle.wait()
    assert handle.done


def test_local_executor_forwards_none_skip_existing(tmp_path, local_calls):
    LocalExecutor(n_workers=1).start(["sim-a"], project=str(tmp_path)).wait()

    assert local_calls[0]["skip_existing"] is None
    assert local_calls[0]["model"].compute_root is None


def test_local_executor_n_workers_override(tmp_path, local_calls):
    """Explicit n_workers kwarg overrides the instance default."""
    LocalExecutor(n_workers=5).start(
        ["sim-a", "sim-b", "sim-c"], project=str(tmp_path), n_workers=2
    ).wait()

    assert local_calls[0]["n_cores"] == 2


def test_local_executor_uses_instance_n_workers_when_omitted(tmp_path, local_calls):
    LocalExecutor(n_workers=3).start(["sim-a", "sim-b"], project=str(tmp_path)).wait()

    assert local_calls[0]["n_cores"] == 3


def test_local_executor_start_noops_when_pending_is_empty(tmp_path, local_calls):
    handle = LocalExecutor(n_workers=5).start([], project=str(tmp_path), n_workers=2)

    assert isinstance(handle, LocalHandle)
    assert handle.done
    handle.wait()
    assert local_calls == []


def test_local_executor_wait_reraises_worker_exception(tmp_path, monkeypatch):
    def failing_run_simulations(model, sim_ids, *, n_cores=1, skip_existing=None):
        raise RuntimeError("worker boom")

    monkeypatch.setattr("stilt.model.Model", _FakeModel)
    monkeypatch.setattr(
        "stilt.execution.worker.run_simulations", failing_run_simulations
    )

    handle = LocalExecutor(n_workers=1).start(["sim-a"], project=str(tmp_path))

    with pytest.raises(RuntimeError, match="worker boom"):
        handle.wait()
    # The error is surfaced once; a second wait() is a no-op.
    handle.wait()
    assert handle.done


def test_local_executor_dispatch_is_push():
    assert LocalExecutor().dispatch == "push"
    assert LocalExecutor(n_workers=4).n_workers == 4


def test_local_handle_job_id_and_detached():
    handle = LocalHandle()
    assert handle.job_id == "local"
    assert handle.detached is False


def test_local_handle_without_thread_is_done_and_wait_is_noop():
    handle = LocalHandle()
    assert handle.done
    assert handle.wait() is None
    assert handle.wait() is None


def test_local_handle_wait_joins_thread():
    finished = threading.Event()
    thread = threading.Thread(target=finished.set)
    handle = LocalHandle(thread)
    thread.start()
    handle.wait()
    assert finished.is_set()
    assert handle.done


# ---------------------------------------------------------------------------
# SlurmExecutor
# ---------------------------------------------------------------------------


def test_slurm_executor_from_config_extracts_pystilt_keys():
    ex = SlurmExecutor.from_config(
        {
            "backend": "slurm",
            "account": "lin-np",
            "partition": "lin-np",
            "n_workers": 200,
            "cpus_per_task": 24,
            "array_parallelism": 50,
        }
    )
    assert ex._cpus_per_task == 24
    assert ex._array_parallelism == 50
    assert ex.n_workers == 200
    assert ex._kwargs == {"account": "lin-np", "partition": "lin-np"}


def test_slurm_executor_from_config_requires_explicit_n_workers():
    with pytest.raises(ValueError, match="explicit 'n_workers'"):
        SlurmExecutor.from_config({"backend": "slurm"})


def test_slurm_executor_render_sbatch_directives_simple():
    ex = SlurmExecutor(
        n_workers=4,
        cpus_per_task=1,
        account="lin-np",
        partition="notchpeak",
    )
    directives = ex._render_sbatch_directives(n_workers=4, project="/tmp/my_project")
    assert "--array=0-3" in directives
    assert "--account=lin-np" in directives
    assert "--partition=notchpeak" in directives
    assert "--job-name=pystilt-my-project" in directives
    assert "--cpus-per-task" not in directives  # only added when > 1


def test_slurm_executor_render_sbatch_directives_with_parallelism():
    ex = SlurmExecutor(
        n_workers=10,
        cpus_per_task=8,
        array_parallelism=5,
    )
    directives = ex._render_sbatch_directives(n_workers=10, project="/tmp/my_project")
    assert "--array=0-9%5" in directives
    assert "--cpus-per-task=8" in directives


def test_slurm_executor_render_sbatch_bool_true():
    """Boolean True renders as bare flag."""
    ex = SlurmExecutor(exclusive=True, n_workers=1)
    directives = ex._render_sbatch_directives(n_workers=1, project="/tmp/my_project")
    assert "--exclusive" in directives
    assert "--exclusive=" not in directives


def test_slurm_executor_render_sbatch_bool_false():
    """Boolean False is skipped."""
    ex = SlurmExecutor(exclusive=False, n_workers=1)
    directives = ex._render_sbatch_directives(n_workers=1, project="/tmp/my_project")
    assert "exclusive" not in directives


def test_slurm_executor_explicit_job_name_overrides_default():
    ex = SlurmExecutor(job_name="custom-name", n_workers=1)
    directives = ex._render_sbatch_directives(n_workers=1, project="/tmp/my_project")
    assert "--job-name=custom-name" in directives
    assert "--job-name=pystilt-my-project" not in directives


def test_slurm_executor_start_renders_chunk_worker_script(tmp_path, monkeypatch):
    """start() renders a chunk-based push-worker script."""
    import subprocess

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 777\n", stderr=""
        )

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config(
        {"backend": "slurm", "partition": "notchpeak", "n_workers": 4}
    )
    handle = ex.start(
        [f"sim-{i}" for i in range(4)],
        project=str(tmp_path),
        n_workers=4,
    )

    assert isinstance(handle, SlurmHandle)
    assert handle.job_id == "777"

    # Chunk files live under <project>/chunks/<batch>/task_N.txt, one per task.
    [batch_dir] = list((tmp_path / "chunks").iterdir())
    chunk_files = sorted(batch_dir.glob("task_*.txt"))
    assert [p.name for p in chunk_files] == [f"task_{i}.txt" for i in range(4)]
    assert chunk_files[0].read_text() == "sim-0\n"
    assert handle._chunk_dir == batch_dir

    # Script should call 'stilt push-worker <project>' with one resolved chunk path.
    slurm_dir = tmp_path / "slurm"
    scripts = list(slurm_dir.glob("submit_*.sh"))
    assert len(scripts) == 1
    assert scripts[0].name == f"submit_{batch_dir.name}.sh"
    script_text = scripts[0].read_text()
    assert f"stilt push-worker {tmp_path}" in script_text
    assert f"CHUNK_PATH={batch_dir}/task_${{SLURM_ARRAY_TASK_ID}}.txt" in script_text
    assert '--chunk "$CHUNK_PATH"' in script_text
    assert "#SBATCH --job-name=pystilt-" in script_text
    logs_dir = slurm_dir / "logs" / batch_dir.name
    assert f"#SBATCH --output={logs_dir}/%a.out" in script_text
    assert logs_dir.is_dir()
    assert "--output-dir" not in script_text
    assert "--compute-root" not in script_text
    assert "skip-existing" not in script_text
    assert calls == [["sbatch", str(scripts[0])]]


def test_slurm_executor_start_with_compute_root_and_cpus(tmp_path, monkeypatch):
    """Rendered worker scripts include the compute-root and cpus flags."""
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 10\n", stderr=""
        )

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config(
        {"backend": "slurm", "n_workers": 1, "cpus_per_task": 4}
    )
    ex.start(
        ["sim-a"],
        project=str(tmp_path),
        n_workers=1,
        compute_root="/scratch/pystilt",
    )

    script_text = list((tmp_path / "slurm").glob("submit_*.sh"))[0].read_text()
    assert "--compute-root /scratch/pystilt" in script_text
    assert "--cpus 4" in script_text
    assert "#SBATCH --cpus-per-task=4" in script_text
    assert "--output-dir" not in script_text


def test_slurm_executor_start_renders_skip_existing_override(tmp_path, monkeypatch):
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 11\n", stderr=""
        )

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    ex.start(
        ["sim-a"],
        project=str(tmp_path),
        n_workers=1,
        skip_existing=False,
    )

    script_text = list((tmp_path / "slurm").glob("submit_*.sh"))[0].read_text()
    assert "--no-skip-existing" in script_text


def test_slurm_executor_rejects_uri_project(monkeypatch):
    """Cloud (URI) projects are rejected for Slurm push dispatch before sbatch."""
    monkeypatch.setattr(
        "stilt.execution.backends.slurm.subprocess.run",
        lambda *a, **k: pytest.fail("sbatch must not run"),
    )
    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    with pytest.raises(ValueError, match="requires a local project root"):
        ex.start(
            ["sim-a"],
            project="s3://bucket/my_proj",
            n_workers=1,
        )


def test_slurm_executor_start_zero_workers_returns_none_job_id(monkeypatch):
    ex = SlurmExecutor(n_workers=1)
    handle = ex.start([], project=".", n_workers=0)
    assert handle.job_id == "none"


def test_slurm_handle_job_id():
    assert SlurmHandle("12345").job_id == "12345"


def test_slurm_handle_wait_polls_until_complete_and_cleans_chunks(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "task_0.txt").write_text("sim-a\n")

    def fake_run(args, capture_output, text, timeout):
        command = args[0]
        calls.append(command)
        if command == "squeue":
            stdout = "12345 running\n" if calls.count("squeue") == 1 else ""
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if command == "sacct":
            return types.SimpleNamespace(
                returncode=0,
                stdout="COMPLETED|\nCOMPLETED+|\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)
    monkeypatch.setattr("stilt.execution.backends.slurm.time.sleep", lambda _: None)

    handle = SlurmHandle("12345", chunk_dir=chunk_dir)
    handle.wait()

    assert calls == ["squeue", "squeue", "sacct"]
    assert not chunk_dir.exists()


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT"])
def test_slurm_handle_wait_raises_for_unsuccessful_terminal_states(
    tmp_path, monkeypatch, state
):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    def fake_run(args, capture_output, text, timeout):
        if args[0] == "squeue":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[0] == "sacct":
            return types.SimpleNamespace(returncode=0, stdout=f"{state}|\n", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)

    handle = SlurmHandle("12345", chunk_dir=chunk_dir)
    with pytest.raises(RuntimeError, match="finished unsuccessfully"):
        handle.wait()

    assert not chunk_dir.exists()


def test_slurm_handle_wait_keeps_chunks_when_poll_times_out(tmp_path, monkeypatch):
    """
    A scheduler-poll timeout must NOT delete the chunk dir.

    Regression test: a busy controller can make squeue/sacct time out while the
    array job is still running. The old code deleted the chunk dir in a blanket
    ``finally``, starving the still-queued tasks (FileNotFoundError -> failed).
    The chunk dir must survive so the running job can finish.
    """
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "task_0.txt").write_text("sim-a\n")

    def always_timeout(args, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", always_timeout)
    monkeypatch.setattr("stilt.execution.backends.slurm.time.sleep", lambda _: None)

    handle = SlurmHandle("12345", chunk_dir=chunk_dir)
    with pytest.raises(subprocess.TimeoutExpired):
        handle.wait()

    # The job never left the queue, so the chunk files must remain.
    assert chunk_dir.exists()
    assert (chunk_dir / "task_0.txt").exists()


def test_slurm_handle_wait_retries_transient_poll_timeout(tmp_path, monkeypatch):
    """A single transient poll timeout should be retried, not fatal."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "task_0.txt").write_text("sim-a\n")

    calls: list[str] = []

    def flaky_run(args, capture_output, text, timeout):
        command = args[0]
        calls.append(command)
        # First squeue call times out once, then succeeds (job gone).
        if command == "squeue" and calls.count("squeue") == 1:
            raise subprocess.TimeoutExpired(args, timeout)
        if command == "squeue":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if command == "sacct":
            return types.SimpleNamespace(returncode=0, stdout="COMPLETED|\n", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", flaky_run)
    monkeypatch.setattr("stilt.execution.backends.slurm.time.sleep", lambda _: None)

    handle = SlurmHandle("12345", chunk_dir=chunk_dir)
    handle.wait()  # should not raise despite the first timeout

    # Job completed cleanly -> chunk dir cleaned up.
    assert not chunk_dir.exists()


# ---------------------------------------------------------------------------
# get_executor
# ---------------------------------------------------------------------------


def test_get_executor_defaults_to_local_single_worker():
    ex = get_executor()
    assert isinstance(ex, LocalExecutor)
    assert ex.n_workers == 1


def test_resolve_backend_defaults_to_local():
    assert resolve_backend() == "local"
    assert resolve_backend({"backend": "local"}) == "local"


def test_get_executor_local_n_workers_1_uses_local_executor():
    ex = get_executor({"backend": "local", "n_workers": 1})
    assert isinstance(ex, LocalExecutor)
    assert ex.n_workers == 1


def test_get_executor_local_n_workers_gt1_uses_local_executor():
    ex = get_executor({"backend": "local", "n_workers": 4})
    assert isinstance(ex, LocalExecutor)
    assert ex.n_workers == 4


def test_get_executor_slurm_is_slurm_executor():
    ex = get_executor({"backend": "slurm", "partition": "notchpeak", "n_workers": 4})
    assert isinstance(ex, SlurmExecutor)


def test_get_executor_slurm_requires_explicit_n_workers():
    with pytest.raises(ValueError, match="explicit 'n_workers'"):
        get_executor({"backend": "slurm", "partition": "notchpeak"})


def test_get_executor_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown execution backend"):
        get_executor({"backend": "bogus", "n_workers": 2})


# ---------------------------------------------------------------------------
# KubernetesHandle
# ---------------------------------------------------------------------------


def test_kubernetes_handle_job_id_job():
    h = KubernetesHandle("stilt-myproj", "default")
    assert h.job_id == "job/stilt-myproj"


def test_kubernetes_handle_wait_job_polls_until_complete(monkeypatch):
    """Job handles poll the Kubernetes API until completions are satisfied."""
    calls: dict[str, int] = {"reads": 0, "kube_config": 0}

    class FakeConfigException(Exception):
        pass

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.ConfigException = FakeConfigException

    def load_incluster_config():
        raise FakeConfigException()

    def load_kube_config():
        calls["kube_config"] += 1

    config_mod.load_incluster_config = load_incluster_config
    config_mod.load_kube_config = load_kube_config

    class FakeBatchV1Api:
        def read_namespaced_job(self, name, namespace):
            calls["reads"] += 1
            if calls["reads"] == 1:
                status = types.SimpleNamespace(succeeded=0, failed=0)
            else:
                status = types.SimpleNamespace(succeeded=1, failed=0)
            spec = types.SimpleNamespace(completions=1)
            return types.SimpleNamespace(status=status, spec=spec)

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.BatchV1Api = FakeBatchV1Api

    root_mod = types.ModuleType("kubernetes")
    root_mod.client = client_mod
    root_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "kubernetes", root_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)
    monkeypatch.setattr("time.sleep", lambda _: None)

    h = KubernetesHandle("stilt-myproj", "default")
    h.wait()

    assert calls["reads"] == 2
    assert calls["kube_config"] == 1


# ---------------------------------------------------------------------------
# KubernetesExecutor
# ---------------------------------------------------------------------------


def test_kubernetes_executor_from_config():
    cfg = {
        "backend": "kubernetes",
        "n_workers": 4,
        "image": "my-reg/stilt:latest",
        "namespace": "stilt",
        "db_secret": "mydb",
    }
    ex = KubernetesExecutor.from_config(cfg)
    assert ex._image == "my-reg/stilt:latest"
    assert ex._namespace == "stilt"
    assert ex.n_workers == 4
    assert ex._db_secret == "mydb"


def test_kubernetes_executor_from_config_defaults():
    ex = KubernetesExecutor.from_config({"backend": "kubernetes", "image": "img:tag"})
    assert ex._namespace == "default"
    assert ex.n_workers == 1
    assert ex._db_secret == "pystilt-db"


def test_kubernetes_executor_start_batch_applies_job(monkeypatch):
    """start() applies a Job manifest."""
    ex = KubernetesExecutor(image="img")
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    handle = ex.start(
        ["sim-a", "sim-b"],
        project="/data/myproj",
        n_workers=2,
        skip_existing=False,
    )
    assert len(applied) == 1
    assert applied[0]["kind"] == "Job"
    assert applied[0]["metadata"]["name"] == "stilt-myproj"
    assert isinstance(handle, KubernetesHandle)
    assert handle.job_id.startswith("job/")


def test_kubernetes_executor_start_accepts_skip_existing_override(monkeypatch):
    ex = KubernetesExecutor(image="img")
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))

    ex.start(
        ["sim-a"],
        project="/data/myproj",
        skip_existing=True,
    )

    assert len(applied) == 1


def test_kubernetes_executor_start_uses_instance_n_workers_when_omitted(
    monkeypatch,
):
    ex = KubernetesExecutor(image="img", n_workers=4)
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    ex.start(
        ["sim-a"],
        project="/data/myproj",
    )
    assert applied[0]["spec"]["completions"] == 4
    container = applied[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["stilt", "pull-worker", "/data/myproj"]


def test_kubernetes_executor_start_includes_compute_root(monkeypatch):
    ex = KubernetesExecutor(image="img")
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    ex.start(
        ["sim-a"],
        project="gs://bucket/myproj",
        n_workers=1,
        compute_root="/tmp/pystilt",
    )
    container = applied[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "pull-worker",
        "gs://bucket/myproj",
        "--compute-root",
        "/tmp/pystilt",
    ]
    assert applied[0]["metadata"]["name"] == "stilt-myproj"


def test_kubernetes_executor_apply_creates_job(monkeypatch):
    applied: list[tuple[str, str]] = []

    class FakeConfigException(Exception):
        pass

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.ConfigException = FakeConfigException
    config_mod.load_incluster_config = lambda: None
    config_mod.load_kube_config = lambda: None

    class FakeApiException(Exception):
        def __init__(self, status):
            self.status = status

    class FakeBatchV1Api:
        def create_namespaced_job(self, namespace, manifest):
            applied.append((namespace, manifest["metadata"]["name"]))

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.BatchV1Api = FakeBatchV1Api

    rest_mod = types.ModuleType("kubernetes.client.rest")
    rest_mod.ApiException = FakeApiException

    root_mod = types.ModuleType("kubernetes")
    root_mod.client = client_mod
    root_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "kubernetes", root_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client.rest", rest_mod)

    ex = KubernetesExecutor(image="img", namespace="stilt")
    ex._apply({"kind": "Job", "metadata": {"name": "job-a", "namespace": "stilt"}})
    assert applied == [("stilt", "job-a")]


def test_kubernetes_executor_apply_ignores_conflict(monkeypatch):
    """_apply suppresses Kubernetes 409 already-exists errors."""

    class FakeConfigException(Exception):
        pass

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.ConfigException = FakeConfigException
    config_mod.load_incluster_config = lambda: None
    config_mod.load_kube_config = lambda: None

    class FakeApiException(Exception):
        def __init__(self, status):
            self.status = status

    class FakeBatchV1Api:
        def create_namespaced_job(self, namespace, manifest):
            raise FakeApiException(409)

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.BatchV1Api = FakeBatchV1Api
    client_mod.AppsV1Api = lambda: None
    client_mod.CustomObjectsApi = lambda: None

    rest_mod = types.ModuleType("kubernetes.client.rest")
    rest_mod.ApiException = FakeApiException

    root_mod = types.ModuleType("kubernetes")
    root_mod.client = client_mod
    root_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "kubernetes", root_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client.rest", rest_mod)

    ex = KubernetesExecutor(image="img")
    ex._apply({"kind": "Job", "metadata": {"name": "job-a", "namespace": "default"}})


def test_kubernetes_executor_apply_reraises_non_conflict(monkeypatch):
    """_apply re-raises Kubernetes API errors other than 409."""

    class FakeConfigException(Exception):
        pass

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.ConfigException = FakeConfigException
    config_mod.load_incluster_config = lambda: None
    config_mod.load_kube_config = lambda: None

    class FakeApiException(Exception):
        def __init__(self, status):
            self.status = status

    class FakeBatchV1Api:
        def create_namespaced_job(self, namespace, manifest):
            raise FakeApiException(500)

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.BatchV1Api = FakeBatchV1Api
    client_mod.AppsV1Api = lambda: None
    client_mod.CustomObjectsApi = lambda: None

    rest_mod = types.ModuleType("kubernetes.client.rest")
    rest_mod.ApiException = FakeApiException

    root_mod = types.ModuleType("kubernetes")
    root_mod.client = client_mod
    root_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "kubernetes", root_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client.rest", rest_mod)

    ex = KubernetesExecutor(image="img")
    with pytest.raises(FakeApiException):
        ex._apply(
            {"kind": "Job", "metadata": {"name": "job-a", "namespace": "default"}}
        )


def test_get_executor_kubernetes_is_kubernetes_executor():
    ex = get_executor({"backend": "kubernetes", "image": "img:tag"})
    assert isinstance(ex, KubernetesExecutor)
