"""Local execution backend."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from .protocol import LaunchSpec

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
    from stilt.workers import pull_worker_loop

    pull_worker_loop(
        Model(project=project, output_dir=output_dir, compute_root=compute_root),
        n_cores=n_cores,
        follow=follow,
    )


def _push_worker_entrypoint(
    project: str,
    chunk_path: str,
    n_cores: int,
    output_dir: str | None = None,
    compute_root: str | None = None,
) -> None:
    """Chunk worker entry point for :class:`LocalExecutor`."""
    from stilt.model import Model
    from stilt.workers import push_worker_loop

    push_worker_loop(
        Model(project=project, output_dir=output_dir, compute_root=compute_root),
        chunk_path=chunk_path,
        n_cores=n_cores,
    )


class LocalExecutor:
    """Local executor: n=1 runs in-process; n>1 spawns a process pool."""

    def __init__(self, n_workers: int) -> None:
        self._n_workers = n_workers

    def start(self, spec: LaunchSpec) -> LocalHandle:
        """Start local workers in-process or via a process pool."""
        n = spec.n_workers if spec.n_workers is not None else self._n_workers
        if spec.dispatch == "push":
            chunks = list(spec.chunks)
            if not chunks:
                return LocalHandle()
            if len(chunks) == 1:
                _push_worker_entrypoint(
                    spec.project,
                    chunks[0],
                    1,
                    spec.output_dir,
                    spec.compute_root,
                )
                return LocalHandle()

            pool = ProcessPoolExecutor(max_workers=min(n, len(chunks)))
            futures = [
                pool.submit(
                    _push_worker_entrypoint,
                    spec.project,
                    chunk,
                    1,
                    spec.output_dir,
                    spec.compute_root,
                )
                for chunk in chunks
            ]
            return LocalHandle(futures, pool)

        if n <= 1:
            from stilt.model import Model
            from stilt.workers import pull_worker_loop

            pull_worker_loop(
                Model(
                    project=spec.project,
                    output_dir=spec.output_dir,
                    compute_root=spec.compute_root,
                ),
                n_cores=max(n, 1),
                follow=False,
            )
            return LocalHandle()

        pool = ProcessPoolExecutor(max_workers=n)
        futures = [
            pool.submit(
                _worker_entrypoint,
                spec.project,
                1,
                False,
                spec.output_dir,
                spec.compute_root,
            )
            for _ in range(n)
        ]
        return LocalHandle(futures, pool)
