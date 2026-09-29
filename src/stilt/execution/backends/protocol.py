"""Interfaces shared by the execution backends."""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import Literal, Protocol

DispatchMode = Literal["push", "pull"]


@contextlib.contextmanager
def sigterm_as_interrupt():
    """
    Context manager that makes SIGTERM raise ``KeyboardInterrupt``.

    Signal handlers can only be set from the main thread, so in any other
    thread this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = signal.getsignal(signal.SIGTERM)

    def _handle(signum: int, frame: object) -> None:
        """Turn SIGTERM into KeyboardInterrupt so cleanup runs."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class JobHandle(Protocol):
    """Handle to the workers started by :meth:`Executor.start`."""

    @property
    def job_id(self) -> str:
        """Job id from the backend, such as a Slurm job id."""
        ...

    @property
    def detached(self) -> bool:
        """
        Whether the workers keep running after this process exits.

        True for Slurm and Kubernetes. False for the local backend, whose
        workers must be waited for.
        """
        ...

    def wait(self) -> None:
        """Block until the workers have stopped."""
        ...


class Executor(Protocol):
    """
    Interface for a backend that starts workers.

    ``dispatch`` is ``"push"`` when the executor is given the receptor ids
    to run, and ``"pull"`` when its workers take them from the work queue.
    """

    dispatch: DispatchMode

    @property
    def n_workers(self) -> int:
        """Number of workers :meth:`start` starts."""
        ...

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> JobHandle:
        """
        Start workers for a project and return a handle to them.

        Parameters
        ----------
        pending : list of str
            Receptor ids to run. Pull executors ignore it.
        project : str
            Project root, a local path or a URI.
        compute_root : str, optional
            Directory where workers run HYSPLIT.
        skip_existing : bool, optional
            Keep outputs that already exist. Defaults to True.

        Returns
        -------
        JobHandle
        """
        ...


__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "sigterm_as_interrupt",
]
