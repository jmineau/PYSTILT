"""Local execution backend."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import DispatchMode

__all__ = ["LocalExecutor", "LocalHandle"]


class LocalHandle:
    """Handle for local execution: joins the worker thread on ``wait()``."""

    def __init__(self, thread: threading.Thread | None = None) -> None:
        self._thread = thread
        self._error: BaseException | None = None

    @property
    def job_id(self) -> str:
        return "local"

    @property
    def detached(self) -> bool:
        """Local workers belong to this process and must be awaited."""
        return False

    @property
    def done(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def wait(self) -> None:
        """Block until the local workers finish; re-raise any worker error."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error


class LocalExecutor:
    """
    Run simulations in this process (``n_workers=1``) or a local process pool.

    Work runs on a background thread so callers (the CLI) can report progress
    while waiting; ``LocalHandle.wait()`` joins it.
    """

    dispatch: DispatchMode = "push"

    def __init__(self, n_workers: int = 1) -> None:
        self._n_workers = n_workers

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        n_workers: int | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> LocalHandle:
        if not pending:
            return LocalHandle()

        n = n_workers if n_workers is not None else self._n_workers
        handle = LocalHandle()

        def _work() -> None:
            from stilt.model import Model

            from ..worker import run_simulations

            try:
                run_simulations(
                    Model(project=project, compute_root=compute_root),
                    pending,
                    n_cores=n,
                    skip_existing=skip_existing,
                )
            except BaseException as exc:  # surfaced by wait()
                handle._error = exc

        thread = threading.Thread(target=_work, name="pystilt-local", daemon=True)
        handle._thread = thread
        thread.start()
        return handle
