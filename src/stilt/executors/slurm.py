"""Slurm execution backend."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from stilt.artifacts import is_cloud_project, project_slug

from .protocol import LaunchSpec


def _slurm_submission_root(project: str) -> Path:
    """Return the local directory used for Slurm submission artifacts."""
    if is_cloud_project(project):
        return Path(tempfile.mkdtemp(prefix=f"pystilt-slurm-{project_slug(project)}-"))
    return Path(project)


class SlurmHandle:
    """Handle for a fire-and-forget Slurm array job submitted via ``sbatch``."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    @property
    def job_id(self) -> str:
        """Return the scheduler job id reported by ``sbatch``."""
        return self._job_id

    def wait(self) -> None:
        """Poll ``squeue`` until the submitted job no longer appears."""
        import time

        while True:
            result = subprocess.run(
                ["squeue", "--job", self._job_id, "--noheader"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                break
            time.sleep(30)


class SlurmExecutor:
    """
    Fire-and-forget executor that submits Slurm array jobs via ``sbatch``.

    Parameters
    ----------
    n_workers
        Number of array tasks to use for the submission. Each task processes one chunk.
    cpus_per_task
        Number of CPUs to request per array task. This is passed to the push worker, which
        sets the ``--cpus`` flag on the STILT command line, enabling parallel execution within each task if greater than 1.
    array_parallelism
        Maximum number of array tasks to run in parallel. Passed as the ``%N`` suffix in the ``--array`` directive. If not set, all tasks may run in parallel.
    setup
        Optional list of shell commands to run before the push worker command in the submission script.
        This can be used to load modules, activate python environments, or perform other setup steps required for the job.
        Each command should be a complete shell command as it would be typed in the terminal.
    **kwargs
        Additional keyword arguments are passed as ``--key=value`` sbatch directives.
        Underscores in keys are converted to dashes for the directive flags.
        Directives with boolean values are included as ``--key`` if True and omitted if False.
    """

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

    def start(self, spec: LaunchSpec) -> SlurmHandle:
        """Write a submission script, call ``sbatch``, and return the job handle."""
        if spec.dispatch != "push":
            raise ValueError("SlurmExecutor supports only push dispatch.")
        if is_cloud_project(spec.project):
            raise ValueError(
                "Slurm push dispatch requires a local project/output root that compute nodes can read."
            )

        chunk_paths = list(spec.chunks)
        if not chunk_paths:
            return SlurmHandle("none")

        project_dir = _slurm_submission_root(spec.project)
        slurm_dir = project_dir / "slurm"
        logs_dir = slurm_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        n_workers = min(self._n_workers, len(chunk_paths))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = slurm_dir / f"submit_{timestamp}.sh"
        directives = self._render_sbatch_directives(n_workers, project=spec.project)

        chunk_dir = Path(chunk_paths[0]).parent

        # Set flags to pass to the push worker
        cpus_flag = f" --cpus {self._cpus_per_task}" if self._cpus_per_task > 1 else ""
        output_flag = (
            f" --output-dir {shlex.quote(spec.output_dir)}"
            if spec.output_dir is not None
            else ""
        )
        compute_flag = (
            f" --compute-root {shlex.quote(spec.compute_root)}"
            if spec.compute_root is not None
            else ""
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
                f"stilt push-worker {shlex.quote(spec.project)}"
                ' --chunk "$CHUNK_PATH"'
                f"{cpus_flag}{output_flag}{compute_flag}"
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
        return SlurmHandle(job_id)
