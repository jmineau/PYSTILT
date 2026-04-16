"""Execution backends for dispatching STILT workers."""

import subprocess
from concurrent.futures import ProcessPoolExecutor

from .factory import get_executor
from .kubernetes import KubernetesExecutor, KubernetesHandle
from .local import LocalExecutor, LocalHandle
from .protocol import Executor, JobHandle, _sigterm_as_interrupt
from .slurm import SlurmExecutor, SlurmHandle

__all__ = [
    "Executor",
    "JobHandle",
    "KubernetesExecutor",
    "KubernetesHandle",
    "LocalExecutor",
    "LocalHandle",
    "SlurmExecutor",
    "SlurmHandle",
    "_sigterm_as_interrupt",
    "get_executor",
    "ProcessPoolExecutor",
    "subprocess",
]
