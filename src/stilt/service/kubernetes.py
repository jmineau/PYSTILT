"""Kubernetes manifest helpers for queue-backed STILT services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stilt.index.postgres import POSTGRES_PENDING_SIMULATIONS_SQL
from stilt.storage import project_slug

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
    """Return a K8s env block that exposes the runtime DB URL from a Secret."""
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


def serve_command(
    project: str,
    *,
    output_dir: str | None = None,
    compute_root: str | None = None,
) -> list[str]:
    """Return a CLI command for long-lived queue-service workers."""
    command = ["stilt", "serve", project]
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
    template_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
    }
    template_spec.update(dict(pod_spec or {}))
    worker_parallelism = completions if parallelism is None else parallelism
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "completions": completions,
            "parallelism": worker_parallelism,
            "template": {
                "spec": template_spec,
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
    template_spec: dict[str, Any] = {"containers": [container]}
    template_spec.update(dict(pod_spec or {}))
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": app_labels},
            "template": {
                "metadata": {"labels": app_labels},
                "spec": template_spec,
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


def service_deployment_manifest(
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
    """Return a Deployment manifest that runs the service-facing `stilt serve` CLI."""
    return deployment_manifest(
        service_name(project),
        image=image,
        command=serve_command(
            project,
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


def secret_manifest(
    *,
    name: str = DEFAULT_DB_SECRET,
    namespace: str = "default",
    data: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a minimal Secret manifest for the runtime DB URL."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": "Opaque",
        "stringData": dict(
            data
            or {
                DB_URL_SECRET_KEY: (
                    "postgresql://user:password@postgres-host:5432/pystilt"
                )
            }
        ),
    }


__all__ = [
    "DB_URL_ENV",
    "DB_URL_SECRET_KEY",
    "DEFAULT_DB_SECRET",
    "db_secret_env",
    "deployment_manifest",
    "job_manifest",
    "scaled_object_manifest",
    "secret_manifest",
    "serve_command",
    "service_deployment_manifest",
    "service_name",
    "worker_command",
    "worker_deployment_manifest",
    "worker_job_manifest",
]
