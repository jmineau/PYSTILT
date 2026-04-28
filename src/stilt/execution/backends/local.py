"""Local execution backend."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import DispatchMode

__all__ = ["LocalExecutor", "LocalHandle"]


class LocalHandle:
    """Handle for local execution."""

    def __init__(
        self,
        futures: list | None = None,
        pool: ProcessPoolExecutor | None = None,
        *,
        completed: bool | None = None,
    ) -> None:
        self._futures = futures
        self._pool = pool
        self._completed = (
            completed if completed is not None else futures is None and pool is None
        )

    @property
    def job_id(self) -> str:
        """Return the synthetic identifier used for local execution."""
        return "local"

    def wait(self) -> None:
        """Wait for all local worker futures and then shut down the pool."""
        if self._completed:
            return
        try:
            if self._futures is not None and self._pool is not None:
                try:
                    for future in as_completed(self._futures):
                        future.result()
                finally:
                    self._pool.shutdown(wait=False)
        finally:
            self._completed = True


def _local_worker_entrypoint(
    project: str,
    sim_ids: list[str],
    output_dir: str | None = None,
    compute_root: str | None = None,
    skip_existing: bool | None = None,
) -> None:
    """Entry point for spawned local worker processes."""
    from stilt.model import Model

    from ..entrypoints import push_simulations

    push_simulations(
        Model(project=project, output_dir=output_dir, compute_root=compute_root),
        sim_ids,
        skip_existing=skip_existing,
    )


class LocalExecutor:
    """
    Local executor: runs simulations in-process (n=1) or across a process pool (n>1).

    Uses push dispatch — pending sim IDs are distributed directly without chunk
    files or queue claims. The coordinator synchronizes the output index after
    execution.
    """

    dispatch: DispatchMode = "push"

    def __init__(self, n_workers: int) -> None:
        self._n_workers = n_workers

    @property
    def n_workers(self) -> int:
        """Return the default worker count used by this executor."""
        return self._n_workers

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        n_workers: int | None = None,
        output_dir: str | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> LocalHandle:
        """Run pending simulations in-process or distribute across a process pool."""
        if not pending:
            return LocalHandle(completed=True)

        n = n_workers if n_workers is not None else self._n_workers

        if n <= 1:
            from stilt.model import Model

            from ..entrypoints import push_simulations

            push_simulations(
                Model(
                    project=project,
                    output_dir=output_dir,
                    compute_root=compute_root,
                ),
                pending,
                skip_existing=skip_existing,
            )
            return LocalHandle(completed=True)

        # Partition pending sims across worker processes.
        partitions: list[list[str]] = [[] for _ in range(n)]
        for i, sim_id in enumerate(pending):
            partitions[i % n].append(sim_id)
        partitions = [p for p in partitions if p]

        pool = ProcessPoolExecutor(max_workers=len(partitions))
        futures = [
            pool.submit(
                _local_worker_entrypoint,
                project,
                partition,
                output_dir,
                compute_root,
                skip_existing,
            )
            for partition in partitions
        ]
        return LocalHandle(futures, pool)
