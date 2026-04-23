"""Executor factory."""

from __future__ import annotations

from typing import Any

from .kubernetes import KubernetesExecutor
from .local import LocalExecutor
from .protocol import Executor
from .slurm import SlurmExecutor


def resolve_backend(execution: dict[str, Any] | None = None) -> str:
    """Return the configured execution backend."""
    resolved: dict[str, Any] = dict(execution or {})
    return resolved.get("backend", "local")


def get_executor(execution: dict[str, Any] | None = None) -> Executor:
    """Create an executor from an execution config dict."""
    resolved: dict[str, Any] = dict(execution or {})
    backend = resolve_backend(resolved)

    if backend == "local":
        return LocalExecutor(n_workers=resolved.get("n_workers", 1))

    if backend == "slurm":
        return SlurmExecutor.from_config(resolved)

    if backend == "kubernetes":
        return KubernetesExecutor.from_config(resolved)

    raise ValueError(
        f"Unknown execution backend: {backend!r}. "
        "Supported: 'local', 'slurm', 'kubernetes'."
    )
