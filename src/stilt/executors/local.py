"""Local execution backend."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

__all__ = ["LocalExecutor", "LocalHandle"]


class LocalHandle:
    """Handle for local execution."""

    def __init__(
        self,
        futures: list | None = None,
        pool: ProcessPoolExecutor | None = None,
    ) -> None:
        self._futures = futures
        self._pool = pool

    @property
    def job_id(self) -> str:
        """Return the synthetic identifier used for local execution."""
        return "local"

    def wait(self) -> None:
        """Wait for all local worker futures and then shut down the pool."""
        if self._futures is None or self._pool is None:
            return
        try:
            for future in as_completed(self._futures):
                future.result()
        finally:
            self._pool.shutdown(wait=False)


def _worker_entrypoint(
    project: str,
    n_cores: int,
    follow: bool,
    output_dir: str | None = None,
    compute_root: str | None = None,
) -> None:
    """Worker entry point for :class:`LocalExecutor`."""
    from stilt.model import Model
    from stilt.workers import worker_loop

    worker_loop(
        Model(project=project, output_dir=output_dir, compute_root=compute_root),
        n_cores=n_cores,
        follow=follow,
    )


class LocalExecutor:
    """Local executor: n=1 runs in-process; n>1 spawns a process pool."""

    def __init__(self, n_workers: int) -> None:
        self._n_workers = n_workers

    def start(
        self,
        project: str,
        n_workers: int | None = None,
        follow: bool = False,
        output_dir: str | None = None,
        compute_root: str | None = None,
    ) -> LocalHandle:
        """Start local workers in-process or via a process pool."""
        n = n_workers if n_workers is not None else self._n_workers
        if n <= 1:
            from stilt.model import Model
            from stilt.workers import worker_loop

            worker_loop(
                Model(
                    project=project, output_dir=output_dir, compute_root=compute_root
                ),
                n_cores=max(n, 1),
                follow=follow,
            )
            return LocalHandle()

        pool = ProcessPoolExecutor(max_workers=n)
        futures = [
            pool.submit(
                _worker_entrypoint,
                project,
                1,
                follow,
                output_dir,
                compute_root,
            )
            for _ in range(n)
        ]
        return LocalHandle(futures, pool)
