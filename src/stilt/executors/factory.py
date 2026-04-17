"""Executor factory."""

from __future__ import annotations

from typing import Any

from .kubernetes import KubernetesExecutor
from .local import LocalExecutor
from .protocol import DispatchMode, Executor
from .slurm import SlurmExecutor

_DISPATCH_DEFAULTS: dict[str, DispatchMode] = {
    "local": "pull",
    "slurm": "push",
    "kubernetes": "pull",
}

_SUPPORTED_BACKENDS = frozenset(_DISPATCH_DEFAULTS)

_SUPPORTED_DISPATCHES: dict[str, frozenset[DispatchMode]] = {
    "local": frozenset({"push", "pull"}),
    "slurm": frozenset({"push"}),
    "kubernetes": frozenset({"pull"}),
}


def resolve_backend(execution: dict[str, Any] | None = None) -> str:
    """Return the configured execution backend."""
    resolved: dict[str, Any] = dict(execution or {})
    return resolved.get("backend", "local")


def resolve_dispatch(execution: dict[str, Any] | None = None) -> DispatchMode:
    """Return the explicit or backend-default dispatch mode."""
    backend = resolve_backend(execution)
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown execution backend: {backend!r}. "
            "Supported: 'local', 'slurm', 'kubernetes'."
        )
    dispatch = dict(execution or {}).get("dispatch", _DISPATCH_DEFAULTS.get(backend))
    if dispatch not in {"push", "pull"}:
        raise ValueError(
            f"Unknown dispatch mode: {dispatch!r}. Supported: 'push', 'pull'."
        )
    if dispatch not in _SUPPORTED_DISPATCHES.get(backend, frozenset()):
        supported = ", ".join(sorted(_SUPPORTED_DISPATCHES.get(backend, frozenset())))
        raise ValueError(
            f"Execution backend {backend!r} does not support dispatch {dispatch!r}. "
            f"Supported dispatch modes: {supported or 'none'}."
        )
    return dispatch


def get_executor(execution: dict[str, Any] | None = None) -> Executor:
    """Create an executor from an execution config dict."""
    resolved: dict[str, Any] = dict(execution or {})
    backend = resolve_backend(resolved)
    resolve_dispatch(resolved)

    if backend == "local":
        return LocalExecutor(n_workers=resolved.get("n_workers", 1))

    if backend == "slurm":
        return SlurmExecutor.from_config(resolved)

    if backend == "kubernetes":
        return KubernetesExecutor.from_config(resolved)

    raise AssertionError(f"Unhandled execution backend after validation: {backend!r}")
