"""Backend that runs workers as a Kubernetes Job."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import DispatchMode

from stilt.service.kubernetes import (
    service_name,
    worker_job_manifest,
)


class KubernetesHandle:
    """Handle to a Kubernetes Job started by :class:`KubernetesExecutor`."""

    def __init__(self, name: str, namespace: str) -> None:
        self._name = name
        self._namespace = namespace
        self._completed = False

    @property
    def job_id(self) -> str:
        """Resource name of the Job, as ``job/<name>``."""
        return f"job/{self._name}"

    @property
    def detached(self) -> bool:
        """Always True, since the Job runs on after this process exits."""
        return True

    def wait(self) -> None:
        """Block until the Job's pods have all succeeded or failed, polling every 15 s."""
        if self._completed:
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
        self._completed = True


class KubernetesExecutor:
    """
    Run pull workers as a Kubernetes Job.

    The workers take receptors from the PostgreSQL work queue until it is
    empty, so the receptors must be registered first.

    Parameters
    ----------
    image : str
        Container image with PYSTILT installed.
    namespace : str, default "default"
        Kubernetes namespace of the Job.
    n_workers : int, default 1
        Number of worker pods.
    db_secret : str, default "pystilt-db"
        Kubernetes secret holding the queue's ``PYSTILT_DB_URL``.
    **pod_spec
        Extra fields for the pod spec.
    """

    dispatch: DispatchMode = "pull"

    def __init__(
        self,
        image: str,
        namespace: str = "default",
        n_workers: int = 1,
        db_secret: str = "pystilt-db",
        **pod_spec: Any,
    ) -> None:
        self._image = image
        self._namespace = namespace
        self._n_workers = n_workers
        self._db_secret = db_secret
        self._pod_spec = pod_spec

    @property
    def n_workers(self) -> int:
        """Number of worker pods when :meth:`start` is not given one."""
        return self._n_workers

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KubernetesExecutor:
        """Return an executor for a config's ``execution`` settings, which must set ``image``."""
        cfg = dict(config)
        cfg.pop("backend", None)
        return cls(
            image=cfg.pop("image"),
            namespace=cfg.pop("namespace", "default"),
            n_workers=cfg.pop("n_workers", 1),
            db_secret=cfg.pop("db_secret", "pystilt-db"),
            **cfg,
        )

    def _apply(self, manifest: dict) -> None:
        """Create the Job, doing nothing if a Job with its name already exists."""
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

        ns = manifest["metadata"]["namespace"]

        try:
            k8s_client.BatchV1Api().create_namespaced_job(ns, manifest)
        except ApiException as exc:
            if exc.status != 409:
                raise

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        n_workers: int | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> KubernetesHandle:
        """Create the worker Job and return its handle. ``pending`` is not used."""
        name = service_name(project)
        n = n_workers if n_workers is not None else self._n_workers
        manifest = worker_job_manifest(
            project,
            image=self._image,
            n_workers=n,
            namespace=self._namespace,
            compute_root=compute_root,
            db_secret=self._db_secret,
            pod_spec=self._pod_spec,
        )
        manifest["metadata"] = {"name": name, "namespace": self._namespace}

        self._apply(manifest)

        return KubernetesHandle(name, self._namespace)
