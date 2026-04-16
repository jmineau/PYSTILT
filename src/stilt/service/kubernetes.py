"""Kubernetes manifest helpers for queue-backed STILT services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stilt.executors.kubernetes import (
    DB_URL_ENV,
    DB_URL_SECRET_KEY,
    DEFAULT_DB_SECRET,
    db_secret_env,
    deployment_manifest,
    job_manifest,
    scaled_object_manifest,
    service_name,
    worker_command,
    worker_deployment_manifest,
    worker_job_manifest,
)


def serve_command(
    project: str,
    *,
    cpus: int | None = None,
    output_dir: str | None = None,
    compute_root: str | None = None,
) -> list[str]:
    """Return a CLI command for long-lived queue-service workers."""
    command = ["stilt", "serve", project]
    if cpus is not None:
        command.extend(["--cpus", str(cpus)])
    if output_dir is not None:
        command.extend(["--output-dir", output_dir])
    if compute_root is not None:
        command.extend(["--compute-root", compute_root])
    return command


def service_deployment_manifest(
    project: str,
    *,
    image: str,
    replicas: int = 1,
    namespace: str = "default",
    cpus: int | None = None,
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
            cpus=cpus,
            output_dir=output_dir,
            compute_root=compute_root,
        ),
        namespace=namespace,
        replicas=replicas,
        env=db_secret_env(db_secret),
        pod_spec=pod_spec,
    )


def secret_manifest(
    *,
    name: str = DEFAULT_DB_SECRET,
    namespace: str = "default",
    data: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a minimal Secret manifest for the repository DB URL."""
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
