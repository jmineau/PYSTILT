"""Running simulations: worker functions and the local, Slurm, and Kubernetes backends."""

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
from .runner import register, resolve_compute_root, run
from .worker import (
    ReceptorResult,
    SimulationResult,
    pull_receptors,
    run_receptor,
    run_receptors,
    run_simulation,
    run_trajectories,
    write_footprint,
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
    "register",
    "resolve_backend",
    "resolve_compute_root",
    "run",
    "run_receptor",
    "run_receptors",
    "run_simulation",
    "run_trajectories",
    "sigterm_as_interrupt",
    "write_footprint",
]
