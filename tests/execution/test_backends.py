"""Tests for execution backend launchers."""

import sys
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


def test_local_executor_start_calls_execute_simulations_inline(tmp_path, monkeypatch):
    """LocalExecutor.start() runs assigned simulations inline when n_workers <= 1."""
    calls = []

    def fake_push_simulations(model, sim_ids, n_cores=1, skip_existing=None):
        calls.append(
            {
                "sim_ids": sim_ids,
                "n_cores": n_cores,
                "skip_existing": skip_existing,
            }
        )

    # LocalExecutor.start() imports these names locally in inline mode.
    monkeypatch.setattr(
        "stilt.model.Model",
        lambda project, output_dir=None, compute_root=None: object(),
    )
    monkeypatch.setattr(
        "stilt.execution.entrypoints.push_simulations",
        fake_push_simulations,
    )

    ex = LocalExecutor(n_workers=1)
    handle = ex.start(
        ["sim-a", "sim-b"],
        project=str(tmp_path),
        n_workers=1,
    )

    assert isinstance(handle, LocalHandle)
    assert calls == [
        {
            "sim_ids": ["sim-a", "sim-b"],
            "n_cores": 1,
            "skip_existing": None,
        }
    ]


def test_local_handle_job_id():
    assert LocalHandle().job_id == "local"


def test_local_handle_wait_is_noop_for_inline():
    assert LocalHandle().wait() is None


def test_local_executor_start_spawns_workers(tmp_path, monkeypatch):
    """LocalExecutor.start() partitions assigned sim IDs across worker processes."""
    submitted = []

    class FakePool:
        def __init__(self, max_workers):
            self._max = max_workers

        def submit(self, func, *args):
            submitted.append(args)

            class FakeFuture:
                def result(self):
                    return None

            return FakeFuture()

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr("stilt.execution.backends.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=3)
    handle = ex.start(
        ["sim-a", "sim-b", "sim-c", "sim-d", "sim-e"],
        project=str(tmp_path),
        n_workers=3,
    )

    assert isinstance(handle, LocalHandle)
    assert len(submitted) == 3
    # Each call: (project_str, sim_id_partition, output_dir, compute_root)
    for args in submitted:
        assert args[0] == str(tmp_path)
        assert args[2] is None
        assert args[3] is None
    assert [args[1] for args in submitted] == [
        ["sim-a", "sim-d"],
        ["sim-b", "sim-e"],
        ["sim-c"],
    ]


def test_local_handle_wait_process_mode(tmp_path, monkeypatch):
    class FakeFuture:
        def result(self):
            return None

    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor.__new__(ProcessPoolExecutor)
    handle = LocalHandle([FakeFuture()], pool)
    assert handle.job_id == "local"


def test_local_handle_wait_is_idempotent():
    class FakePool:
        def __init__(self):
            self.calls = 0

        def shutdown(self, wait=False):
            self.calls += 1
            return None

    from concurrent.futures import Future

    pool = FakePool()
    future = Future()
    future.set_result(None)
    handle = LocalHandle([future], pool)
    handle.wait()
    handle.wait()

    assert pool.calls == 1


def test_local_executor_n_workers_override(tmp_path, monkeypatch):
    """Explicit n_workers kwarg overrides the instance default."""
    submitted = []

    class FakePool:
        def __init__(self, max_workers):
            self._max = max_workers

        def submit(self, func, *args):
            submitted.append(args)

            class FakeFuture:
                def result(self):
                    return None

            return FakeFuture()

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr("stilt.execution.backends.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=5)
    ex.start(
        ["sim-a", "sim-b", "sim-c"],
        project=str(tmp_path),
        n_workers=2,
    )
    assert len(submitted) == 2  # override used, not 5


def test_local_executor_uses_instance_n_workers_when_omitted(tmp_path, monkeypatch):
    submitted = []

    class FakePool:
        def __init__(self, max_workers):
            self._max = max_workers

        def submit(self, func, *args):
            submitted.append(args)

            class FakeFuture:
                def result(self):
                    return None

            return FakeFuture()

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr("stilt.execution.backends.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=3)
    ex.start(
        ["sim-a", "sim-b", "sim-c"],
        project=str(tmp_path),
    )
    assert len(submitted) == 3


def test_local_executor_start_noops_when_pending_is_empty(tmp_path):
    ex = LocalExecutor(n_workers=5)
    handle = ex.start(
        [],
        project=str(tmp_path),
        n_workers=2,
    )
    assert isinstance(handle, LocalHandle)


def test_local_executor_push_dispatch_submits_chunk_workers(tmp_path, monkeypatch):
    submitted = []

    class FakePool:
        def __init__(self, max_workers):
            self._max = max_workers

        def submit(self, func, *args):
            submitted.append(args)

            class FakeFuture:
                def result(self):
                    return None

            return FakeFuture()

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr("stilt.execution.backends.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=5)
    ex.start(
        ["sim-a", "sim-b", "sim-c"],
        project=str(tmp_path),
        n_workers=2,
    )
    assert len(submitted) == 2
    assert submitted[0][1] == ["sim-a", "sim-c"]
    assert submitted[1][1] == ["sim-b"]


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

    # Script should call 'stilt push-worker <project>' with one resolved chunk path.
    slurm_dir = tmp_path / "slurm"
    scripts = list(slurm_dir.glob("submit_*.sh"))
    assert len(scripts) == 1
    script_text = scripts[0].read_text()
    assert f"stilt push-worker {tmp_path}" in script_text
    assert "CHUNK_PATH=" in script_text
    assert '--chunk "$CHUNK_PATH"' in script_text
    assert "#SBATCH --job-name=pystilt-" in script_text


def test_slurm_executor_start_with_output_dir_and_compute_root(tmp_path, monkeypatch):
    """Rendered worker scripts include the output-dir and compute-root flags."""
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 10\n", stderr=""
        )

    monkeypatch.setattr("stilt.execution.backends.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    ex.start(
        ["sim-a"],
        project=str(tmp_path),
        n_workers=1,
        output_dir=str(tmp_path / "output"),
        compute_root="/scratch/pystilt",
    )

    script_text = list((tmp_path / "slurm").glob("submit_*.sh"))[0].read_text()
    assert f"--output-dir {tmp_path / 'output'}" in script_text
    assert "--compute-root /scratch/pystilt" in script_text


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


def test_slurm_executor_start_cloud_project_uses_local_submission_root(
    tmp_path, monkeypatch
):
    """Cloud projects are rejected for Slurm push dispatch."""
    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    with pytest.raises(ValueError, match="requires local project and output roots"):
        ex.start(
            ["sim-a"],
            project="s3://bucket/my_proj",
            n_workers=1,
        )


def test_slurm_executor_rejects_cloud_output_dir(tmp_path):
    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    with pytest.raises(ValueError, match="requires local project and output roots"):
        ex.start(
            ["sim-a"],
            project=str(tmp_path),
            n_workers=1,
            output_dir="s3://bucket/my_proj",
        )


def test_slurm_executor_start_zero_workers_returns_none_job_id(monkeypatch):
    ex = SlurmExecutor(n_workers=1)
    handle = ex.start([], project=".", n_workers=0)
    assert handle.job_id == "none"


def test_slurm_handle_job_id():
    assert SlurmHandle("12345").job_id == "12345"


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


def test_kubernetes_executor_start_includes_output_dir_and_compute_root(monkeypatch):
    ex = KubernetesExecutor(image="img")
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    ex.start(
        ["sim-a"],
        project="/data/myproj",
        n_workers=1,
        output_dir="gs://bucket/project",
        compute_root="/tmp/pystilt",
    )
    container = applied[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "pull-worker",
        "/data/myproj",
        "--output-dir",
        "gs://bucket/project",
        "--compute-root",
        "/tmp/pystilt",
    ]


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
