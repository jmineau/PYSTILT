"""Shared executor protocols and utilities."""

from __future__ import annotations

import contextlib
import signal
from typing import Literal, Protocol

DispatchMode = Literal["push", "pull"]


@contextlib.contextmanager
def sigterm_as_interrupt():
    """Temporarily convert SIGTERM into KeyboardInterrupt."""
    previous = signal.getsignal(signal.SIGTERM)

    def _handle(signum: int, frame: object) -> None:
        """Translate SIGTERM into ``KeyboardInterrupt`` for worker loops."""
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
        """Return the backend-specific job identifier."""
        ...

    def wait(self) -> None:
        """Block until the launched work is no longer running."""
        ...


class Executor(Protocol):
    """
    Worker-launch protocol: start workers, get a :class:`JobHandle` back immediately.

    Implementors declare a ``dispatch`` class attribute (``"push"`` or ``"pull"``).
    The coordinator reads this to handle dispatch-specific queue/index setup
    and post-run output-index rebuild policy before/after calling :meth:`start`.
    """

    dispatch: DispatchMode

    @property
    def n_workers(self) -> int:
        """Return the executor's default worker count."""
        ...

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        n_workers: int | None = None,
        output_dir: str | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> JobHandle:
        """Launch workers for one project and return a handle immediately."""
        ...


__all__ = [
    "DispatchMode",
    "Executor",
    "JobHandle",
    "sigterm_as_interrupt",
]
