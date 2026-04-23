"""Execution package public surface."""

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
from .entrypoints import pull_simulations, push_simulations
from .execute import execute_batch, execute_task
from .tasks import SimulationResult, SimulationTask

__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "KubernetesExecutor",
    "KubernetesHandle",
    "LocalExecutor",
    "LocalHandle",
    "SimulationResult",
    "SimulationTask",
    "SlurmExecutor",
    "SlurmHandle",
    "execute_batch",
    "execute_task",
    "get_executor",
    "pull_simulations",
    "push_simulations",
    "resolve_backend",
    "sigterm_as_interrupt",
]
