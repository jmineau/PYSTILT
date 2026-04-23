"""Execution backends for dispatching STILT workers."""

from .factory import get_executor
from .kubernetes import KubernetesExecutor, KubernetesHandle
from .local import LocalExecutor, LocalHandle
from .protocol import DispatchMode, Executor, JobHandle
from .slurm import SlurmExecutor, SlurmHandle

__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "KubernetesExecutor",
    "KubernetesHandle",
    "LocalExecutor",
    "LocalHandle",
    "SlurmExecutor",
    "SlurmHandle",
    "get_executor",
]
