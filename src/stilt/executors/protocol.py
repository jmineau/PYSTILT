"""Shared executor protocols and utilities."""

from __future__ import annotations

import contextlib
import signal
from dataclasses import dataclass, field
from typing import Literal, Protocol

DispatchMode = Literal["push", "pull"]


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Concrete launch request passed from the coordinator to executors."""

    project: str
    n_workers: int = 1
    dispatch: DispatchMode = "push"
    output_dir: str | None = None
    compute_root: str | None = None
    chunks: tuple[str, ...] = field(default_factory=tuple)


@contextlib.contextmanager
def _sigterm_as_interrupt():
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
    """Handle returned by :meth:`Executor.start`; optionally blocks via :meth:`wait`."""

    @property
    def job_id(self) -> str:
        """Return the backend-specific job identifier."""
        ...

    def wait(self) -> None:
        """Block until the launched work is no longer running."""
        ...


class Executor(Protocol):
    """Worker-launch protocol: start workers, get a :class:`JobHandle` back immediately."""

    def start(self, spec: LaunchSpec) -> JobHandle:
        """Launch workers for one project and return a handle immediately."""
        ...
