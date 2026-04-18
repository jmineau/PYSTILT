"""Tests for stilt.executors."""

import sys
import types

import pytest

from stilt.executors import (
    KubernetesExecutor,
    KubernetesHandle,
    LaunchSpec,
    LocalExecutor,
    LocalHandle,
    SlurmExecutor,
    SlurmHandle,
    get_executor,
)
from stilt.executors.factory import resolve_dispatch

# ---------------------------------------------------------------------------
# LocalExecutor
# ---------------------------------------------------------------------------


def test_local_executor_start_calls_pull_worker_loop_inline(tmp_path, monkeypatch):
    """LocalExecutor.start() runs inline for pull dispatch when n_workers <= 1."""
    calls = []

    def fake_worker_loop(model, n_cores=1, follow=False):
        calls.append({"n_cores": n_cores, "follow": follow})

    # LocalExecutor.start() imports these names locally in inline mode.
    monkeypatch.setattr(
        "stilt.model.Model",
        lambda project, output_dir=None, compute_root=None: object(),
    )
    monkeypatch.setattr("stilt.workers.pull_worker_loop", fake_worker_loop)

    ex = LocalExecutor(n_workers=1)
    handle = ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=1,
            dispatch="pull",
        )
    )

    assert isinstance(handle, LocalHandle)
    assert len(calls) == 1
    assert calls[0]["n_cores"] == 1
    assert calls[0]["follow"] is False


def test_local_handle_job_id():
    assert LocalHandle().job_id == "local"


def test_local_handle_wait_is_noop_for_inline():
    assert LocalHandle().wait() is None


def test_local_executor_start_spawns_workers(tmp_path, monkeypatch):
    """LocalExecutor.start() submits pull worker entrypoints N times when n_workers > 1."""
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

    monkeypatch.setattr("stilt.executors.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=3)
    handle = ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=3,
            dispatch="pull",
        )
    )

    assert isinstance(handle, LocalHandle)
    assert len(submitted) == 3
    # Each call: (project_str, 1, follow, output_dir, compute_root)
    for args in submitted:
        assert args[0] == str(tmp_path)
        assert args[1] == 1  # n_cores per worker
        assert args[2] is False
        assert args[3] is None
        assert args[4] is None


def test_local_handle_wait_process_mode(tmp_path, monkeypatch):
    class FakeFuture:
        def result(self):
            return None

    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor.__new__(ProcessPoolExecutor)
    handle = LocalHandle([FakeFuture()], pool)
    assert handle.job_id == "local"


def test_local_executor_pull_n_workers_from_launch_spec(tmp_path, monkeypatch):
    """LaunchSpec.n_workers overrides the instance default for pull dispatch."""
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

    monkeypatch.setattr("stilt.executors.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=5)
    ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=2,
            dispatch="pull",
        )
    )
    assert len(submitted) == 2  # override used, not 5


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

    monkeypatch.setattr("stilt.executors.local.ProcessPoolExecutor", FakePool)

    ex = LocalExecutor(n_workers=5)
    ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=2,
            dispatch="push",
            chunks=("/tmp/task_0.txt", "/tmp/task_1.txt"),
        )
    )
    assert len(submitted) == 2
    assert submitted[0][1] == "/tmp/task_0.txt"
    assert submitted[1][1] == "/tmp/task_1.txt"


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
    assert ex._n_workers == 200
    assert ex._cpus_per_task == 24
    assert ex._array_parallelism == 50
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

    monkeypatch.setattr("stilt.executors.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config(
        {"backend": "slurm", "partition": "notchpeak", "n_workers": 4}
    )
    handle = ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=4,
            dispatch="push",
            chunks=tuple(
                str(tmp_path / "simulations" / "chunks" / "batch-1" / f"task_{i}.txt")
                for i in range(4)
            ),
        )
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
    """Rendered worker scripts include the durable-output and compute-root flags."""
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 10\n", stderr=""
        )

    monkeypatch.setattr("stilt.executors.slurm.subprocess.run", fake_run)

    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    ex.start(
        LaunchSpec(
            project=str(tmp_path),
            n_workers=1,
            dispatch="push",
            output_dir="s3://bucket/project",
            compute_root="/scratch/pystilt",
            chunks=(
                str(tmp_path / "simulations" / "chunks" / "batch-1" / "task_0.txt"),
            ),
        )
    )

    script_text = list((tmp_path / "slurm").glob("submit_*.sh"))[0].read_text()
    assert "--output-dir s3://bucket/project" in script_text
    assert "--compute-root /scratch/pystilt" in script_text


def test_slurm_executor_start_cloud_project_uses_local_submission_root(
    tmp_path, monkeypatch
):
    """Cloud projects are rejected for Slurm push dispatch."""
    ex = SlurmExecutor.from_config({"backend": "slurm", "n_workers": 1})
    with pytest.raises(ValueError, match="requires a local project/output root"):
        ex.start(
            LaunchSpec(
                project="s3://bucket/my_proj",
                n_workers=1,
                dispatch="push",
                chunks=("/tmp/task_0.txt",),
            )
        )


def test_slurm_executor_start_zero_workers_returns_none_job_id(monkeypatch):
    ex = SlurmExecutor(n_workers=1)
    handle = ex.start(LaunchSpec(project=".", n_workers=0, dispatch="push"))
    assert handle.job_id == "none"


def test_slurm_handle_job_id():
    assert SlurmHandle("12345").job_id == "12345"


# ---------------------------------------------------------------------------
# get_executor
# ---------------------------------------------------------------------------


def test_get_executor_defaults_to_local_single_worker():
    ex = get_executor()
    assert isinstance(ex, LocalExecutor)
    assert ex._n_workers == 1


def test_resolve_dispatch_defaults_local_to_pull():
    assert resolve_dispatch() == "pull"
    assert resolve_dispatch({"backend": "local"}) == "pull"


def test_get_executor_local_n_workers_1_uses_local_executor():
    ex = get_executor({"backend": "local", "n_workers": 1})
    assert isinstance(ex, LocalExecutor)
    assert ex._n_workers == 1


def test_get_executor_local_n_workers_gt1_uses_local_executor():
    ex = get_executor({"backend": "local", "n_workers": 4})
    assert isinstance(ex, LocalExecutor)
    assert ex._n_workers == 4


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
    h = KubernetesHandle("stilt-myproj", "default", "Job")
    assert h.job_id == "job/stilt-myproj"


def test_kubernetes_handle_job_id_deployment():
    h = KubernetesHandle("stilt-myproj", "default", "Deployment")
    assert h.job_id == "deployment/stilt-myproj"


def test_kubernetes_handle_wait_deployment_is_noop():
    """Deployment handle returns immediately without hitting k8s."""
    h = KubernetesHandle("stilt-myproj", "default", "Deployment")
    h.wait()  # should not raise or block


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

    h = KubernetesHandle("stilt-myproj", "default", "Job")
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
        "autoscale": True,
        "db_secret": "mydb",
        "keda_target_value": 5,
    }
    ex = KubernetesExecutor.from_config(cfg)
    assert ex._image == "my-reg/stilt:latest"
    assert ex._namespace == "stilt"
    assert ex._autoscale is True
    assert ex._db_secret == "mydb"
    assert ex._keda_target_value == 5


def test_kubernetes_executor_from_config_defaults():
    ex = KubernetesExecutor.from_config({"backend": "kubernetes", "image": "img:tag"})
    assert ex._namespace == "default"
    assert ex._autoscale is False
    assert ex._db_secret == "pystilt-db"
    assert ex._keda_target_value == 10


def test_kubernetes_executor_k8s_name_is_dns_safe():
    ex = KubernetesExecutor(image="img")
    assert ex._k8s_name("/path/to/My Project") == "stilt-my-project"
    assert ex._k8s_name("/path/to/my_proj") == "stilt-my-proj"
    assert ex._k8s_name("s3://bucket/my_proj") == "stilt-my-proj"


def test_kubernetes_executor_k8s_name_truncates():
    ex = KubernetesExecutor(image="img")
    long_proj = "a" * 100
    assert len(ex._k8s_name(long_proj)) <= 63


def test_kubernetes_executor_env_block_with_secret():
    ex = KubernetesExecutor(image="img", db_secret="mypg")
    env = ex._env_block()
    assert len(env) == 1
    assert env[0]["name"] == "PYSTILT_DB_URL"
    assert env[0]["valueFrom"]["secretKeyRef"]["name"] == "mypg"


def test_kubernetes_executor_env_block_without_secret():
    ex = KubernetesExecutor(image="img", db_secret="")
    assert ex._env_block() == []


def test_kubernetes_executor_job_manifest_structure():
    ex = KubernetesExecutor(image="my-reg/stilt:v1", namespace="stilt")
    manifest = ex._job_manifest(
        "stilt-myproj",
        "/data/myproj",
        n_workers=3,
        output_dir=None,
        compute_root=None,
    )
    assert manifest["kind"] == "Job"
    assert manifest["apiVersion"] == "batch/v1"
    spec = manifest["spec"]
    assert spec["completions"] == 3
    assert spec["parallelism"] == 3
    container = spec["template"]["spec"]["containers"][0]
    assert container["image"] == "my-reg/stilt:v1"
    assert "/data/myproj" in container["command"]


def test_kubernetes_executor_deployment_manifest_has_follow_flag():
    ex = KubernetesExecutor(image="img")
    manifest = ex._deployment_manifest(
        "stilt-myproj",
        "/data/myproj",
        n_workers=2,
        output_dir=None,
        compute_root=None,
    )
    assert manifest["kind"] == "Deployment"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "--follow" in container["command"]


def test_kubernetes_executor_job_manifest_includes_output_dir_and_compute_root():
    ex = KubernetesExecutor(image="img")
    manifest = ex._job_manifest(
        "stilt-myproj",
        "/data/myproj",
        n_workers=1,
        output_dir="gs://bucket/project",
        compute_root="/tmp/pystilt",
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "pull-worker",
        "/data/myproj",
        "--output-dir",
        "gs://bucket/project",
        "--compute-root",
        "/tmp/pystilt",
    ]


def test_kubernetes_executor_keda_manifest_structure():
    ex = KubernetesExecutor(image="img", keda_target_value=5)
    manifest = ex._keda_manifest("stilt-myproj")
    assert manifest["kind"] == "ScaledObject"
    assert manifest["apiVersion"] == "keda.sh/v1alpha1"
    spec = manifest["spec"]
    assert spec["minReplicaCount"] == 0
    assert spec["maxReplicaCount"] == 50
    trigger = spec["triggers"][0]
    assert trigger["type"] == "postgresql"
    assert trigger["metadata"]["targetQueryValue"] == "5"
    query = trigger["metadata"]["query"]
    assert "trajectory_status IS NULL" not in query
    assert "LEFT JOIN artifact_index AS ai" in query
    assert "LEFT JOIN claims AS c" in query
    assert "FROM attempts AS a" in query


def test_kubernetes_executor_start_batch_applies_job(monkeypatch):
    """start() applies a Job manifest."""
    ex = KubernetesExecutor(image="img")
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    handle = ex.start(
        LaunchSpec(
            project="/data/myproj",
            n_workers=2,
            dispatch="pull",
        )
    )
    assert len(applied) == 1
    assert applied[0]["kind"] == "Job"
    assert isinstance(handle, KubernetesHandle)
    assert handle.job_id.startswith("job/")


def test_kubernetes_executor_start_ignores_autoscale_for_batch(monkeypatch):
    """Executor launches only Jobs; autoscale config does not change start()."""
    ex = KubernetesExecutor(image="img", autoscale=True)
    applied: list[dict] = []
    monkeypatch.setattr(ex, "_apply", lambda m: applied.append(m))
    ex.start(
        LaunchSpec(
            project="/data/myproj",
            n_workers=4,
            dispatch="pull",
        )
    )
    kinds = [m["kind"] for m in applied]
    assert "ScaledObject" not in kinds
    assert kinds == ["Job"]


def test_kubernetes_executor_apply_dispatches_by_manifest_kind(monkeypatch):
    """_apply routes Job, Deployment, and ScaledObject to the right client API."""
    applied: list[tuple[str, str, str]] = []

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
            applied.append(("Job", namespace, manifest["metadata"]["name"]))

    class FakeAppsV1Api:
        def create_namespaced_deployment(self, namespace, manifest):
            applied.append(("Deployment", namespace, manifest["metadata"]["name"]))

    class FakeCustomObjectsApi:
        def create_namespaced_custom_object(
            self, *, group, version, namespace, plural, body
        ):
            applied.append((body["kind"], namespace, body["metadata"]["name"]))

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.BatchV1Api = FakeBatchV1Api
    client_mod.AppsV1Api = FakeAppsV1Api
    client_mod.CustomObjectsApi = FakeCustomObjectsApi

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
    ex._apply(
        {
            "kind": "Deployment",
            "metadata": {"name": "deploy-a", "namespace": "stilt"},
        }
    )
    ex._apply(
        {
            "kind": "ScaledObject",
            "metadata": {"name": "scale-a", "namespace": "stilt"},
        }
    )

    assert applied == [
        ("Job", "stilt", "job-a"),
        ("Deployment", "stilt", "deploy-a"),
        ("ScaledObject", "stilt", "scale-a"),
    ]


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
