"""Shared executor protocols and utilities."""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import Literal, Protocol

DispatchMode = Literal["push", "pull"]


@contextlib.contextmanager
def sigterm_as_interrupt():
    """
    Temporarily convert SIGTERM into ``KeyboardInterrupt``.

    Signal handlers can only be installed from the main thread; elsewhere
    this is a no-op so worker code can run in a background thread.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = signal.getsignal(signal.SIGTERM)

    def _handle(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class JobHandle(Protocol):
    """
    Handle returned by :meth:`Executor.start`.

    ``wait()`` blocks until the launched work is no longer running.
    """

    @property
    def job_id(self) -> str:
        """Backend-specific job identifier."""
        ...

    @property
    def detached(self) -> bool:
        """
        Whether the launched work runs independently of this process.

        ``True`` for backends whose workers survive the submitting process
        (Slurm, Kubernetes). ``False`` for the local backend, whose workers
        must be awaited before this process exits.
        """
        ...

    def wait(self) -> None:
        """Block until the launched work is no longer running."""
        ...


class Executor(Protocol):
    """
    Worker-launch protocol: start workers, get a :class:`JobHandle` back.

    ``dispatch`` says whether the executor is handed the pending ids
    (``"push"``) or whether its workers claim from the queue (``"pull"``).
    """

    dispatch: DispatchMode

    @property
    def n_workers(self) -> int:
        """Default worker count."""
        ...

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        n_workers: int | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> JobHandle:
        """Launch workers for one project root and return a handle."""
        ...


__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "sigterm_as_interrupt",
]
