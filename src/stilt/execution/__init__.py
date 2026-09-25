"""Execution: worker functions plus local, Slurm, and Kubernetes backends."""

from .backends import (
    DispatchMode,
    Executor,
    JobHandle,
    KubernetesExecutor,
    KubernetesHandle,
    LocalExecutor,
    LocalHandle,
    SlurmExecutor,
    SlurmHandle,
    get_executor,
)
from .backends.factory import resolve_backend
from .backends.protocol import sigterm_as_interrupt
from .worker import (
    ReceptorResult,
    SimulationResult,
    pull_receptors,
    run_receptor,
    run_receptors,
    run_simulation,
)

__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "KubernetesExecutor",
    "KubernetesHandle",
    "LocalExecutor",
    "LocalHandle",
    "ReceptorResult",
    "SimulationResult",
    "SlurmExecutor",
    "SlurmHandle",
    "get_executor",
    "pull_receptors",
    "resolve_backend",
    "run_receptor",
    "run_receptors",
    "run_simulation",
    "sigterm_as_interrupt",
]
