"""Backend that submits workers as a Slurm job array."""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import DispatchMode

from stilt.project import project_slug
from stilt.store import is_uri

logger = logging.getLogger(__name__)

# Number of tries for a scheduler query that times out. A busy controller can
# be slow to answer squeue or sacct, and one slow answer should not end the wait.
_POLL_RETRIES = 5


def _run_scheduler_query(
    cmd: list[str], *, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run a Slurm query command, retrying when it times out."""
    last_exc: subprocess.TimeoutExpired | None = None
    for _ in range(_POLL_RETRIES):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            time.sleep(5)
    assert last_exc is not None
    raise last_exc


def _write_chunks(
    chunk_dir: Path,
    sim_ids: list[str],
    *,
    n_workers: int,
) -> int:
    """
    Split receptor ids round-robin into one chunk file per array task.

    Returns the number of chunk files written.
    """
    if not sim_ids:
        return 0
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, min(n_workers, len(sim_ids)))
    buckets: list[list[str]] = [[] for _ in range(n_chunks)]
    for idx, sim_id in enumerate(sim_ids):
        buckets[idx % n_chunks].append(sim_id)
    count = 0
    for idx, chunk in enumerate(buckets):
        if not chunk:
            continue
        (chunk_dir / f"task_{idx}.txt").write_text(
            "\n".join(chunk) + "\n", encoding="utf-8"
        )
        count += 1
    return count


class SlurmHandle:
    """Handle to a Slurm job array submitted with ``sbatch``."""

    def __init__(
        self,
        job_id: str,
        *,
        chunk_dir: Path | None = None,
    ) -> None:
        self._job_id = job_id
        self._chunk_dir = chunk_dir
        self._completed = False

    @property
    def job_id(self) -> str:
        """Job id reported by ``sbatch``."""
        return self._job_id

    @property
    def detached(self) -> bool:
        """Always True, since Slurm jobs run on after this process exits."""
        return True

    def wait(self) -> None:
        """
        Block until the job leaves the Slurm queue.

        Polls ``squeue`` every 30 s, then checks the final state with
        ``sacct`` and raises ``RuntimeError`` if any task failed, was
        cancelled, or timed out. The chunk files are deleted once the job has
        left the queue.
        """
        if self._completed:
            return
        # Chunk files are deleted only after the job has left the queue,
        # whether or not it succeeded. If wait() exits early (a query that
        # keeps timing out, an interrupt), tasks still to run need them.
        job_left_queue = False
        try:
            while True:
                result = _run_scheduler_query(
                    ["squeue", "--job", self._job_id, "--noheader"]
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "squeue failed")
                if not result.stdout.strip():
                    job_left_queue = True
                    break
                time.sleep(30)
            status = _run_scheduler_query(
                [
                    "sacct",
                    "--jobs",
                    self._job_id,
                    "--noheader",
                    "--parsable2",
                    "--format=State",
                ]
            )
            if status.returncode != 0:
                raise RuntimeError(status.stderr.strip() or "sacct failed")
            states = {
                line.strip().split("|")[0]
                for line in status.stdout.splitlines()
                if line.strip()
            }
            if any(
                state.startswith(prefix)
                for state in states
                for prefix in ("FAILED", "CANCELLED", "TIMEOUT")
            ):
                raise RuntimeError(
                    f"Slurm job {self._job_id} finished unsuccessfully: {sorted(states)}"
                )
            self._completed = True
        finally:
            # Once the job has left the queue no task will read the chunks
            # again. Otherwise they are left in place.
            if job_left_queue and self._chunk_dir is not None:
                shutil.rmtree(self._chunk_dir, ignore_errors=True)


class SlurmExecutor:
    """
    Run receptors as a Slurm job array submitted with ``sbatch``.

    :meth:`start` splits the receptor ids into one chunk file per array task
    under ``<project>/chunks/<batch>/``, writes a submission script under
    ``<project>/slurm/``, and submits it. Each task runs
    ``stilt push-worker`` on its chunk. The project must be a local
    directory.

    Parameters
    ----------
    n_workers : int
        Number of array tasks.
    cpus_per_task : int, default 1
        CPUs per array task. With more than one, each task runs its
        receptors in a process pool of that size.
    array_parallelism : int, optional
        Maximum number of array tasks running at once (the ``%N`` suffix
        of ``--array``).
    setup : list of str, optional
        Shell commands to run before the worker, such as loading modules.
    **kwargs
        Other ``sbatch`` options, written as ``#SBATCH --key=value``.
        Underscores in keys become hyphens, and ``True`` writes a bare flag.
    """

    dispatch: DispatchMode = "push"

    def __init__(
        self,
        n_workers: int,
        cpus_per_task: int = 1,
        array_parallelism: int | None = None,
        setup: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._n_workers = n_workers
        self._cpus_per_task = cpus_per_task
        self._array_parallelism = array_parallelism
        self._setup: list[str] = setup or []
        self._kwargs = kwargs

    @property
    def n_workers(self) -> int:
        """Number of array tasks."""
        return self._n_workers

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SlurmExecutor:
        """Return an executor for a config's ``execution`` settings, which must set ``n_workers``."""
        cfg = dict(config)
        cfg.pop("backend", None)
        n_workers = cfg.pop("n_workers", None)
        if n_workers is None:
            raise ValueError(
                "SlurmExecutor requires explicit 'n_workers' in execution config."
            )
        cpus_per_task = cfg.pop("cpus_per_task", cfg.pop("cpus-per-task", 1))
        array_parallelism = cfg.pop("array_parallelism", None)
        setup = cfg.pop("setup", None)
        if isinstance(setup, str):
            setup = [setup]
        return cls(
            n_workers=n_workers,
            cpus_per_task=cpus_per_task,
            array_parallelism=array_parallelism,
            setup=setup,
            **cfg,
        )

    def _resolved_slurm_kwargs(self, project: str) -> dict[str, Any]:
        """Return the ``sbatch`` options, with a default job name."""
        kwargs = dict(self._kwargs)
        kwargs.setdefault("job_name", f"pystilt-{project_slug(project)}")
        return kwargs

    def _render_sbatch_directives(self, n_workers: int, *, project: str) -> str:
        """Return the ``#SBATCH`` lines of a submission script."""
        lines: list[str] = []
        array_spec = f"0-{n_workers - 1}"
        if self._array_parallelism is not None:
            array_spec += f"%{self._array_parallelism}"
        lines.append(f"#SBATCH --array={array_spec}")
        if self._cpus_per_task > 1:
            lines.append(f"#SBATCH --cpus-per-task={self._cpus_per_task}")
        for key, value in self._resolved_slurm_kwargs(project).items():
            flag = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    lines.append(f"#SBATCH --{flag}")
            else:
                lines.append(f"#SBATCH --{flag}={value}")
        return "\n".join(lines)

    def start(
        self,
        pending: list[str],
        *,
        project: str,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> SlurmHandle:
        """Write the chunk files and submission script, submit it, and return a handle."""
        if is_uri(project):
            raise ValueError("Slurm push dispatch requires a local project root.")

        project_dir = Path(project)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_dir = project_dir / "chunks" / batch_id
        n_written = _write_chunks(chunk_dir, pending, n_workers=self._n_workers)
        if not n_written:
            return SlurmHandle("none")

        slurm_dir = project_dir / "slurm"
        # One directory per submission, so a later array never overwrites these.
        logs_dir = slurm_dir / "logs" / batch_id
        logs_dir.mkdir(parents=True, exist_ok=True)

        script_path = slurm_dir / f"submit_{batch_id}.sh"
        directives = self._render_sbatch_directives(n_written, project=project)

        cpus_flag = f" --cpus {self._cpus_per_task}" if self._cpus_per_task > 1 else ""
        compute_flag = (
            f" --compute-root {shlex.quote(compute_root)}"
            if compute_root is not None
            else ""
        )
        skip_flag = " --no-skip" if skip_existing is False else ""
        script_lines = [
            "#!/bin/bash",
            directives,
            f"#SBATCH --output={logs_dir}/%a.out",
            f"#SBATCH --error={logs_dir}/%a.err",
            "",
            *self._setup,
            *([""] if self._setup else []),
            f"CHUNK_PATH={shlex.quote(str(chunk_dir))}/task_${{SLURM_ARRAY_TASK_ID}}.txt",
            (
                f"stilt push-worker {shlex.quote(project)}"
                ' --chunk "$CHUNK_PATH"'
                f"{cpus_flag}{compute_flag}{skip_flag}"
            ),
        ]
        script_path.write_text("\n".join(script_lines) + "\n")
        script_path.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {result.returncode}):\n"
                f"  script: {script_path}\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}"
            )
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted job: {job_id}")
        return SlurmHandle(job_id, chunk_dir=chunk_dir)
