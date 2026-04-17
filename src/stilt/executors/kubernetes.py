"""Kubernetes execution backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stilt.artifacts import project_slug
from stilt.repositories.postgres import POSTGRES_PENDING_SIMULATIONS_SQL

from .protocol import LaunchSpec

DB_URL_ENV = "PYSTILT_DB_URL"
DB_URL_SECRET_KEY = "PYSTILT_DB_URL"
DEFAULT_DB_SECRET = "pystilt-db"
DEFAULT_CONTAINER_NAME = "stilt-worker"


def service_name(project: str) -> str:
    """Return a Kubernetes-safe default name for a STILT project/service."""
    slug = project_slug(project)
    return f"stilt-{slug}"[:63]


def db_secret_env(
    db_secret: str | None = DEFAULT_DB_SECRET,
    *,
    env_name: str = DB_URL_ENV,
    secret_key: str = DB_URL_SECRET_KEY,
) -> list[dict[str, Any]]:
    """Return a K8s env block that exposes the repository DB URL from a Secret."""
    if not db_secret:
        return []
    return [
        {
            "name": env_name,
            "valueFrom": {
                "secretKeyRef": {
                    "name": db_secret,
                    "key": secret_key,
                }
            },
        }
    ]


def worker_command(
    project: str,
    *,
    follow: bool = False,
    output_dir: str | None = None,
    compute_root: str | None = None,
) -> list[str]:
    """Return a CLI command for batch or follow-mode queue workers."""
    command = ["stilt", "pull-worker", project]
    if follow:
        command.append("--follow")
    if output_dir is not None:
        command.extend(["--output-dir", output_dir])
    if compute_root is not None:
        command.extend(["--compute-root", compute_root])
    return command


def job_manifest(
    name: str,
    *,
    image: str,
    command: list[str],
    namespace: str = "default",
    completions: int = 1,
    parallelism: int | None = None,
    env: list[dict[str, Any]] | None = None,
    pod_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a generic worker Job manifest."""
    container: dict[str, Any] = {
        "name": DEFAULT_CONTAINER_NAME,
        "image": image,
        "command": command,
    }
    if env:
        container["env"] = list(env)
    container.update(dict(pod_spec or {}))
    worker_parallelism = completions if parallelism is None else parallelism
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "completions": completions,
            "parallelism": worker_parallelism,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [container],
                }
            },
        },
    }


def deployment_manifest(
    name: str,
    *,
    image: str,
    command: list[str],
    namespace: str = "default",
    replicas: int = 1,
    env: list[dict[str, Any]] | None = None,
    pod_spec: Mapping[str, Any] | None = None,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a generic worker Deployment manifest."""
    app_labels = {"app": name}
    if labels:
        app_labels.update(dict(labels))
    container: dict[str, Any] = {
        "name": DEFAULT_CONTAINER_NAME,
        "image": image,
        "command": command,
    }
    if env:
        container["env"] = list(env)
    container.update(dict(pod_spec or {}))
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": app_labels},
            "template": {
                "metadata": {"labels": app_labels},
                "spec": {
                    "containers": [container],
                },
            },
        },
    }


def worker_job_manifest(
    project: str,
    *,
    image: str,
    n_workers: int = 1,
    namespace: str = "default",
    output_dir: str | None = None,
    compute_root: str | None = None,
    db_secret: str | None = DEFAULT_DB_SECRET,
    pod_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a Job manifest that drains a queue in batch mode."""
    return job_manifest(
        service_name(project),
        image=image,
        command=worker_command(
            project,
            follow=False,
            output_dir=output_dir,
            compute_root=compute_root,
        ),
        namespace=namespace,
        completions=n_workers,
        parallelism=n_workers,
        env=db_secret_env(db_secret),
        pod_spec=pod_spec,
    )


def worker_deployment_manifest(
    project: str,
    *,
    image: str,
    replicas: int = 1,
    namespace: str = "default",
    output_dir: str | None = None,
    compute_root: str | None = None,
    db_secret: str | None = DEFAULT_DB_SECRET,
    pod_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a Deployment manifest that runs follow-mode worker pods."""
    return deployment_manifest(
        service_name(project),
        image=image,
        command=worker_command(
            project,
            follow=True,
            output_dir=output_dir,
            compute_root=compute_root,
        ),
        namespace=namespace,
        replicas=replicas,
        env=db_secret_env(db_secret),
        pod_spec=pod_spec,
    )


def scaled_object_manifest(
    name: str,
    *,
    namespace: str = "default",
    target_query_value: int = 10,
    connection_env: str = DB_URL_ENV,
    query: str | None = None,
    min_replica_count: int = 0,
    max_replica_count: int = 50,
) -> dict[str, Any]:
    """Return a KEDA ScaledObject manifest for queue-depth-based autoscaling."""
    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": f"{name}-scaler", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {"name": name},
            "minReplicaCount": min_replica_count,
            "maxReplicaCount": max_replica_count,
            "triggers": [
                {
                    "type": "postgresql",
                    "metadata": {
                        "query": " ".join(
                            (query or POSTGRES_PENDING_SIMULATIONS_SQL).split()
                        ),
                        "targetQueryValue": str(target_query_value),
                        "connectionFromEnv": connection_env,
                    },
                }
            ],
        },
    }


class KubernetesHandle:
    """Handle for a K8s Job or Deployment created by :class:`KubernetesExecutor`."""

    def __init__(self, name: str, namespace: str, kind: str) -> None:
        self._name = name
        self._namespace = namespace
        self._kind = kind

    @property
    def job_id(self) -> str:
        """Return the Kubernetes resource identifier for the launched workers."""
        return f"{self._kind.lower()}/{self._name}"

    def wait(self) -> None:
        """Block on batch jobs; deployments return immediately."""
        if self._kind == "Deployment":
            return

        import time

        try:
            from kubernetes import client as k8s_client
            from kubernetes import config as k8s_cfg
        except ImportError as exc:
            raise ImportError(
                "KubernetesExecutor requires the kubernetes package. "
                "Install it with: pip install 'pystilt[cloud]'"
            ) from exc

        try:
            k8s_cfg.load_incluster_config()
        except k8s_cfg.ConfigException:
            k8s_cfg.load_kube_config()

        batch_v1 = k8s_client.BatchV1Api()
        while True:
            job = batch_v1.read_namespaced_job(self._name, self._namespace)
            status = job.status  # type: ignore[union-attr]
            if (status.succeeded or 0) + (status.failed or 0) >= (
                job.spec.completions or 1  # type: ignore[union-attr]
            ):
                break
            time.sleep(15)


class KubernetesExecutor:
    """Deploy batch-mode STILT pull workers as Kubernetes Jobs."""

    def __init__(
        self,
        image: str,
        namespace: str = "default",
        autoscale: bool = False,
        db_secret: str = "pystilt-db",
        keda_target_value: int = 10,
        **pod_spec: Any,
    ) -> None:
        self._image = image
        self._namespace = namespace
        self._autoscale = autoscale
        self._db_secret = db_secret
        self._keda_target_value = keda_target_value
        self._pod_spec = pod_spec

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KubernetesExecutor:
        """Build a Kubernetes executor from ``ModelConfig.execution`` values."""
        cfg = dict(config)
        cfg.pop("backend", None)
        cfg.pop("n_workers", None)
        return cls(
            image=cfg.pop("image"),
            namespace=cfg.pop("namespace", "default"),
            autoscale=cfg.pop("autoscale", False),
            db_secret=cfg.pop("db_secret", "pystilt-db"),
            keda_target_value=cfg.pop("keda_target_value", 10),
            **cfg,
        )

    def _k8s_name(self, project: str) -> str:
        """Return the base Kubernetes object name for one project."""
        return service_name(project)

    def _env_block(self) -> list[dict]:
        """Return environment variables injected into worker pods."""
        return db_secret_env(self._db_secret)

    def _worker_command(
        self,
        project: str,
        *,
        follow: bool,
        output_dir: str | None,
        compute_root: str | None,
    ) -> list[str]:
        """Return the worker CLI command for this executor instance."""
        return worker_command(
            project,
            follow=follow,
            output_dir=output_dir,
            compute_root=compute_root,
        )

    def _job_manifest(
        self,
        name: str,
        project: str,
        n_workers: int,
        *,
        output_dir: str | None,
        compute_root: str | None,
    ) -> dict:
        """Return a Job manifest that drains the queue in batch mode."""
        return worker_job_manifest(
            project,
            image=self._image,
            n_workers=n_workers,
            namespace=self._namespace,
            output_dir=output_dir,
            compute_root=compute_root,
            db_secret=self._db_secret,
            pod_spec=self._pod_spec,
        ) | {"metadata": {"name": name, "namespace": self._namespace}}

    def _deployment_manifest(
        self,
        name: str,
        project: str,
        n_workers: int,
        *,
        output_dir: str | None,
        compute_root: str | None,
    ) -> dict:
        """Return a follow-mode Deployment manifest for worker pods."""
        return worker_deployment_manifest(
            project,
            image=self._image,
            replicas=n_workers,
            namespace=self._namespace,
            output_dir=output_dir,
            compute_root=compute_root,
            db_secret=self._db_secret,
            pod_spec=self._pod_spec,
        ) | {"metadata": {"name": name, "namespace": self._namespace}}

    def _keda_manifest(self, name: str) -> dict:
        """Return the autoscaling manifest for a follow-mode deployment."""
        return scaled_object_manifest(
            name,
            namespace=self._namespace,
            target_query_value=self._keda_target_value,
        )

    def _apply(self, manifest: dict) -> None:
        """Apply one manifest, tolerating already-exists conflicts."""
        try:
            from kubernetes import client as k8s_client
            from kubernetes import config as k8s_cfg
            from kubernetes.client.rest import ApiException
        except ImportError as exc:
            raise ImportError(
                "KubernetesExecutor requires the kubernetes package. "
                "Install it with: pip install 'pystilt[cloud]'"
            ) from exc

        try:
            k8s_cfg.load_incluster_config()
        except k8s_cfg.ConfigException:
            k8s_cfg.load_kube_config()

        kind = manifest["kind"]
        ns = manifest["metadata"]["namespace"]

        try:
            if kind == "Job":
                k8s_client.BatchV1Api().create_namespaced_job(ns, manifest)
            elif kind == "Deployment":
                k8s_client.AppsV1Api().create_namespaced_deployment(ns, manifest)
            elif kind == "ScaledObject":
                k8s_client.CustomObjectsApi().create_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=ns,
                    plural="scaledobjects",
                    body=manifest,
                )
        except ApiException as exc:
            if exc.status != 409:
                raise

    def start(self, spec: LaunchSpec) -> KubernetesHandle:
        """Create the worker Job and return its handle."""
        if spec.dispatch != "pull":
            raise ValueError("KubernetesExecutor supports only pull dispatch.")

        name = self._k8s_name(spec.project)
        manifest = self._job_manifest(
            name,
            spec.project,
            spec.n_workers,
            output_dir=spec.output_dir,
            compute_root=spec.compute_root,
        )

        self._apply(manifest)

        return KubernetesHandle(name, self._namespace, "Job")
