"""Tests for stilt.service.kubernetes manifest helpers."""

from stilt.repositories.postgres import POSTGRES_PENDING_SIMULATIONS_SQL
from stilt.service.kubernetes import (
    db_secret_env,
    scaled_object_manifest,
    secret_manifest,
    serve_command,
    service_deployment_manifest,
    service_name,
    worker_command,
    worker_deployment_manifest,
    worker_job_manifest,
)


def test_service_name_uses_project_slug():
    assert service_name("gs://bucket/my-project").startswith("stilt-")


def test_db_secret_env_defaults_to_pystilt_db():
    env = db_secret_env()
    assert env == [
        {
            "name": "PYSTILT_DB_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "pystilt-db",
                    "key": "PYSTILT_DB_URL",
                }
            },
        }
    ]


def test_db_secret_env_empty_when_secret_disabled():
    assert db_secret_env(None) == []


def test_worker_command_follow_mode_includes_follow_flag():
    assert worker_command("/data/project", follow=True) == [
        "stilt",
        "pull-worker",
        "/data/project",
        "--follow",
    ]


def test_serve_command_uses_service_cli_and_cpus():
    assert serve_command("/data/project", cpus=2) == [
        "stilt",
        "serve",
        "/data/project",
        "--cpus",
        "2",
    ]


def test_worker_job_manifest_structure():
    manifest = worker_job_manifest(
        "/data/project",
        image="img",
        n_workers=2,
        namespace="stilt",
        output_dir="gs://bucket/project",
        compute_root="/tmp/pystilt",
    )
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "stilt-project"
    assert manifest["metadata"]["namespace"] == "stilt"
    assert manifest["spec"]["parallelism"] == 2
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "pull-worker",
        "/data/project",
        "--output-dir",
        "gs://bucket/project",
        "--compute-root",
        "/tmp/pystilt",
    ]


def test_worker_deployment_manifest_matches_follow_worker_shape():
    manifest = worker_deployment_manifest("/data/project", image="img", replicas=3)
    assert manifest["kind"] == "Deployment"
    assert manifest["spec"]["replicas"] == 3
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "pull-worker",
        "/data/project",
        "--follow",
    ]


def test_service_deployment_manifest_uses_serve_cli():
    manifest = service_deployment_manifest(
        "/data/project",
        image="img",
        replicas=1,
        cpus=4,
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "stilt",
        "serve",
        "/data/project",
        "--cpus",
        "4",
    ]


def test_scaled_object_manifest_uses_pending_query():
    manifest = scaled_object_manifest("stilt-project", namespace="stilt")
    trigger = manifest["spec"]["triggers"][0]
    assert trigger["metadata"]["connectionFromEnv"] == "PYSTILT_DB_URL"
    assert trigger["metadata"]["query"] == " ".join(
        POSTGRES_PENDING_SIMULATIONS_SQL.split()
    )


def test_secret_manifest_defaults():
    manifest = secret_manifest(namespace="stilt")
    assert manifest["kind"] == "Secret"
    assert manifest["metadata"]["name"] == "pystilt-db"
    assert "PYSTILT_DB_URL" in manifest["stringData"]
