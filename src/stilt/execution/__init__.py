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
    SimulationResult,
    pull_simulations,
    run_simulation,
    run_simulations,
)

__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "KubernetesExecutor",
    "KubernetesHandle",
    "LocalExecutor",
    "LocalHandle",
    "SimulationResult",
    "SlurmExecutor",
    "SlurmHandle",
    "get_executor",
    "pull_simulations",
    "resolve_backend",
    "run_simulation",
    "run_simulations",
    "sigterm_as_interrupt",
]
