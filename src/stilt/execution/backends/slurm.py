"""Slurm execution backend."""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import DispatchMode

from stilt.storage import ProjectFiles, is_cloud_project, project_slug

logger = logging.getLogger(__name__)


def _write_chunks(
    output_dir: Path,
    sim_ids: list[str],
    *,
    n_workers: int,
    batch_id: str,
) -> int:
    """Partition sim IDs into chunk files for array tasks.

    Returns the number of chunk files written.
    """
    if not sim_ids:
        return 0
    chunk_dir = ProjectFiles(output_dir).chunks_dir / batch_id
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


def _slurm_submission_root(project: str) -> Path:
    """Return the local directory used for Slurm submission files."""
    if is_cloud_project(project):
        return Path(tempfile.mkdtemp(prefix=f"pystilt-slurm-{project_slug(project)}-"))
    return Path(project)


class SlurmHandle:
    """Handle for a fire-and-forget Slurm array job submitted via ``sbatch``."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._completed = False

    @property
    def job_id(self) -> str:
        """Return the scheduler job id reported by ``sbatch``."""
        return self._job_id

    def wait(self) -> None:
        """Poll ``squeue`` until the submitted job no longer appears."""
        if self._completed:
            return
        while True:
            result = subprocess.run(
                ["squeue", "--job", self._job_id, "--noheader"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                break
            time.sleep(30)
        self._completed = True


class SlurmExecutor:
    """
    Fire-and-forget executor that submits Slurm array jobs via ``sbatch``.

    Always uses push dispatch — the coordinator writes immutable chunk files
    before calling :meth:`start`, and ``SlurmExecutor`` derives the chunk
    directory from ``spec.output_dir`` and ``spec.batch_id``.

    Parameters
    ----------
    n_workers
        Number of array tasks to use for the submission. Each task processes one chunk.
    cpus_per_task
        Number of CPUs to request per array task. This is passed to the push worker
        via ``--cpus``, enabling parallel execution within each task if greater than 1.
    array_parallelism
        Maximum number of array tasks to run in parallel (``%N`` suffix).
    setup
        Optional list of shell commands to run before the push worker command.
    **kwargs
        Additional keyword arguments passed as ``--key=value`` sbatch directives.
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
        """Return the default array-task count used by this executor."""
        return self._n_workers

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SlurmExecutor:
        """Build a Slurm executor from ``ModelConfig.execution`` values."""
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
        """Return sbatch kwargs with PYSTILT defaults applied."""
        kwargs = dict(self._kwargs)
        kwargs.setdefault("job_name", f"pystilt-{project_slug(project)}")
        return kwargs

    def _render_sbatch_directives(self, n_workers: int, *, project: str) -> str:
        """Render the ``#SBATCH`` directive block for one submission script."""
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
        n_workers: int | None = None,
        output_dir: str | None = None,
        compute_root: str | None = None,
        skip_existing: bool | None = None,
    ) -> SlurmHandle:
        """Write chunk files, generate a submission script, submit via ``sbatch``."""
        if is_cloud_project(project) or (
            output_dir is not None and is_cloud_project(output_dir)
        ):
            raise ValueError(
                "Slurm push dispatch currently requires local project and output roots."
            )

        output_target = output_dir or project
        chunk_root = Path(output_target)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = n_workers if n_workers is not None else self._n_workers
        n_written = _write_chunks(chunk_root, pending, n_workers=n, batch_id=batch_id)
        if not n_written:
            return SlurmHandle("none")

        chunk_dir = ProjectFiles(chunk_root).chunks_dir / batch_id
        project_dir = _slurm_submission_root(project)
        slurm_dir = project_dir / "slurm"
        logs_dir = slurm_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        script_path = slurm_dir / f"submit_{batch_id}.sh"
        directives = self._render_sbatch_directives(n_written, project=project)

        cpus_flag = f" --cpus {self._cpus_per_task}" if self._cpus_per_task > 1 else ""
        output_flag = (
            f" --output-dir {shlex.quote(output_target)}"
            if output_dir is not None
            else ""
        )
        compute_flag = (
            f" --compute-root {shlex.quote(compute_root)}"
            if compute_root is not None
            else ""
        )
        skip_flag = (
            ""
            if skip_existing is None
            else (" --skip-existing" if skip_existing else " --no-skip-existing")
        )
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
                f"{cpus_flag}{output_flag}{compute_flag}{skip_flag}"
            ),
        ]
        script_path.write_text("\n".join(script_lines) + "\n")
        script_path.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
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
        return SlurmHandle(job_id)
