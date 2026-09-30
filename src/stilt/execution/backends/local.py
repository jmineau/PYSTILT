"""Backend that runs workers on this machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import DispatchMode

__all__ = ["LocalExecutor", "LocalHandle"]


class LocalHandle:
    """Handle to a local run, which has already finished when it is returned."""

    @property
    def job_id(self) -> str:
        """Always ``"local"``, since local runs have no scheduler id."""
        return "local"

    @property
    def detached(self) -> bool:
        """Always False, since local workers stop when this process exits."""
        return False

    def wait(self) -> None:
        """Return at once: the run finished inside :meth:`LocalExecutor.start`."""


class LocalExecutor:
    """
    Run receptors on this machine, in one process or a process pool.

    :meth:`start` runs the receptors and returns when they are done, so
    Ctrl-C and SIGTERM reach the workers and stop them cleanly.

    Parameters
    ----------
    n_workers : int, default 1
        Number of worker processes. 1 runs in this process.
    """

    dispatch: DispatchMode = "push"

    def __init__(self, n_workers: int = 1) -> None:
        self._n_workers = n_workers

    @property
    def n_workers(self) -> int:
        """Number of worker processes."""
        return self._n_workers

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> LocalHandle:
        """Run ``pending`` receptors and return a handle once they are done."""
        if pending:
            from stilt.model import Model

            from ..worker import run_receptors

            run_receptors(
                Model(project=project),
                pending,
                compute_root=compute_root,
                n_cores=self._n_workers,
                skip_existing=True if skip_existing is None else skip_existing,
            )
        return LocalHandle()
