"""Backend that runs workers on this machine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import DispatchMode

__all__ = ["LocalExecutor", "LocalHandle"]


class LocalHandle:
    """Handle to a local run, which runs on a background thread."""

    def __init__(self, thread: threading.Thread | None = None) -> None:
        self._thread = thread
        self._error: BaseException | None = None

    @property
    def job_id(self) -> str:
        """Always ``"local"``, since local runs have no scheduler id."""
        return "local"

    @property
    def detached(self) -> bool:
        """Always False, since local workers stop when this process exits."""
        return False

    @property
    def done(self) -> bool:
        """Whether the run has finished."""
        return self._thread is None or not self._thread.is_alive()

    def wait(self) -> None:
        """Block until the run finishes, raising any error it raised."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error


class LocalExecutor:
    """
    Run receptors on this machine, in one process or a process pool.

    The run happens on a background thread, so :meth:`start` returns at
    once. Call ``wait()`` on the handle to block until it finishes.

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
        n_workers: int | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> LocalHandle:
        """Start running ``pending`` receptors and return a handle."""
        if not pending:
            return LocalHandle()

        n = n_workers if n_workers is not None else self._n_workers
        handle = LocalHandle()

        def _work() -> None:
            """Run the receptors, keeping any error for ``wait()``."""
            from stilt.model import Model

            from ..worker import run_receptors

            try:
                run_receptors(
                    Model(project=project, compute_root=compute_root),
                    pending,
                    n_cores=n,
                    skip_existing=True if skip_existing is None else skip_existing,
                )
            except BaseException as exc:  # surfaced by wait()
                handle._error = exc

        thread = threading.Thread(target=_work, name="pystilt-local", daemon=True)
        handle._thread = thread
        thread.start()
        return handle
