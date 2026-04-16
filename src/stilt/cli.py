"""STILT command-line interface.

Thin Typer wrapper over the Model API. Each command loads a project or durable
output root, delegates to the appropriate ``Model`` method, and prints a brief
status summary. All heavy lifting lives in ``model.py`` and ``workers.py``;
this module has no orchestration logic of its own.

Usage examples::

    stilt init                        # scaffold a new project in cwd
    stilt init ./my_project           # scaffold a new project in ./my_project
    stilt run                         # run locally, block until done
    stilt run ./my_project --no-skip  # re-run all simulations
    stilt run --wait                  # submit to Slurm and block until done
    stilt worker ./my_project                       # drain pending simulations
    stilt serve ./my_project --cpus 8               # long-lived streaming mode
    stilt submit ./my_project --receptors new_receptors.csv  # register sims without running
    stilt rebuild                     # rebuild repository DB from disk
    stilt status                      # show status from cwd
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel

from stilt.artifacts import (
    is_cloud_project,
    project_config_path,
    project_receptors_path,
    simulation_state_db_path,
)
from stilt.config import STILTParams
from stilt.executors import SlurmHandle, get_executor
from stilt.model import Model
from stilt.receptor import read_receptors
from stilt.service import Service, summarize_queue
from stilt.workers import worker_loop

app = typer.Typer(
    name="stilt",
    help="STILT model command-line interface.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _to_commented_yaml(model_cls: type[BaseModel], data: dict[str, Any]) -> str:
    """Render *data* as YAML with a ``#`` comment above each key that has a description."""
    fields = model_cls.model_fields
    lines: list[str] = []
    for key, value in data.items():
        field = fields.get(key)
        if field is not None and field.description:
            desc = " ".join(field.description.split())
            lines.append(f"# {desc}")
        lines.append(yaml.dump({key: value}, default_flow_style=False).rstrip())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shared options / arguments
# ---------------------------------------------------------------------------

_PROJECT_DIR_ARG = typer.Argument(
    None,
    help="Path or URI of the STILT project. Defaults to the current directory.",
)
_NEW_PROJECT_DIR_ARG = typer.Argument(
    None,
    help="Path to the new STILT project directory. Defaults to the current directory.",
)
_REQUIRED_PROJECT_DIR_ARG = typer.Argument(
    ..., help="Path or URI of the STILT project."
)
_NO_SKIP = typer.Option(
    False, "--no-skip", help="Re-run simulations that already have output."
)
_OUTPUT_DIR = typer.Option(
    None,
    "--output-dir",
    help="Durable output root. May be a local path or object-storage URI.",
)
_COMPUTE_ROOT = typer.Option(
    None,
    "--compute-root",
    help="Parent directory under which worker simulation dirs are created.",
)


def _resolve_project_dir(
    path: str | Path | None, *, require_inputs: bool = True
) -> str:
    """Resolve a local root path or pass through a cloud project/output URI."""
    raw = str(path or Path.cwd())
    if is_cloud_project(raw):
        return raw

    resolved = Path(raw).resolve()
    has_inputs = (resolved / "config.yaml").exists()
    has_state = (resolved / "simulations").exists() or simulation_state_db_path(
        resolved
    ).exists()
    if require_inputs and not has_inputs:
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not require_inputs and not (has_inputs or has_state):
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project or durable output root.",
            err=True,
        )
        raise typer.Exit(code=1)
    return str(resolved)


def _resolve_model_root(
    project_dir: str | Path | None,
    output_dir: str | None,
    *,
    require_inputs: bool,
) -> tuple[str, str | None]:
    """Return the root passed to Model plus any separate durable output override."""
    if project_dir is not None:
        return _resolve_project_dir(
            project_dir, require_inputs=require_inputs
        ), output_dir
    if output_dir is not None:
        return _resolve_project_dir(output_dir, require_inputs=require_inputs), None
    return _resolve_project_dir(None, require_inputs=require_inputs), None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    project_dir: Path = _NEW_PROJECT_DIR_ARG,
) -> None:
    """Scaffold a new STILT project directory with a default config.yaml.

    Creates a starter config.yaml and receptors.csv. Edit both files
    before running ``stilt run``.

    Examples
    --------
    ::

        stilt init
        stilt init /path/to/project
    """
    project_dir = (project_dir or Path.cwd()).resolve()
    config_path = project_config_path(project_dir)
    receptors_path = project_receptors_path(project_dir)

    if config_path.exists():
        typer.echo(
            f"Error: '{project_dir}' already contains a config.yaml. Aborting.",
            err=True,
        )
        raise typer.Exit(code=1)

    project_dir.mkdir(parents=True, exist_ok=True)

    stilt_yaml = _to_commented_yaml(STILTParams, {"n_hours": -24, "numpar": 1000})
    config_path.write_text(
        "# STILT project configuration\n"
        "# Edit mets.hrrr.directory to point to your ARL meteorology files.\n"
        "\n" + stilt_yaml + "\nmets:\n"
        "  hrrr:\n"
        "    directory: /path/to/arl/meteorology\n"
        '    file_format: "%Y%m%d_%H"\n'
        "    file_tres: 1h\n"
        "\n"
        "# execution:\n"
        "#   backend: local  # increase n_workers for parallel; use 'slurm' for HPC\n"
        "#   n_workers: 1\n"
    )

    receptors_path.write_text(
        "time,longitude,latitude,altitude\n"
        "# Example: 2023-01-01 12:00:00,-111.85,40.77,5\n"
    )

    typer.echo(f"Initialized STILT project at '{project_dir}'")
    typer.echo("  config.yaml   — edit met directory and footprint settings")
    typer.echo("  receptors.csv — add receptor times/locations here")


@app.command()
def run(
    project_dir: str | None = _PROJECT_DIR_ARG,
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        help="Path to config.yaml. Project dir defaults to its parent.",
    ),
    no_skip: bool = _NO_SKIP,
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override execution backend: local | slurm.",
    ),
    n_workers: int | None = typer.Option(
        None,
        "--n-workers",
        help="Override number of workers (overrides config.yaml execution.n_workers).",
    ),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help=(
            "Block until submitted Slurm jobs finish before returning. "
            "By default, ``stilt run`` returns immediately after ``sbatch`` when "
            "using backend: slurm (fire-and-forget). Local backends always "
            "complete inline regardless of this flag."
        ),
    ),
    batch_id: str | None = typer.Option(
        None,
        "--batch-id",
        help="Label this submission for batch progress tracking.",
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run trajectories (and footprints if configured).

    Reads ``config.yaml`` in the project directory.  If footprint configs are
    defined there, footprints are generated after trajectories.

    For ``backend: local`` (default), the command always blocks until all
    simulations complete.  For ``backend: slurm``, the command submits the
    job array and returns immediately — use ``--wait`` to poll until done.
    Pass ``--no-skip`` to re-run simulations that already have output.
    """
    # Resolve project dir: --config parent takes precedence over positional arg.
    if config_path is not None:
        resolved_dir = str(config_path.resolve().parent)
        if (
            not (Path(resolved_dir) / "config.yaml").exists()
            and not config_path.exists()
        ):
            typer.echo(f"Error: config file not found: {config_path}", err=True)
            raise typer.Exit(code=1)
    else:
        resolved_dir, resolved_output = _resolve_model_root(
            project_dir, output_dir, require_inputs=True
        )

    model = Model(
        project=resolved_dir,
        output_dir=resolved_output,
        compute_root=compute_root,
    )

    # Build executor override if --backend or --n-workers are given.
    executor = None
    if backend is not None or n_workers is not None:
        execution = dict(model.config.execution or {})
        if backend is not None:
            execution["backend"] = backend
        if n_workers is not None:
            execution["n_workers"] = n_workers
        executor = get_executor(execution)

    skip_existing = None if not no_skip else False
    handle = model.run(
        executor=executor,
        skip_existing=skip_existing,
        wait=False,
        batch_id=batch_id,
    )
    if isinstance(handle, SlurmHandle):
        typer.echo(f"Submitted job: {handle.job_id}")
        if wait:
            handle.wait()
            _print_status(model)
    else:
        # Local backend — always complete inline so no orphan workers.
        handle.wait()
        _print_status(model)


@app.command()
def submit(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    receptors_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--receptors",
        help=(
            "Path to a receptors CSV to register. "
            "Defaults to the project's receptors.csv."
        ),
    ),
    batch_id: str | None = typer.Option(
        None,
        "--batch-id",
        help="Label this group of simulations for batch progress tracking.",
    ),
    output_dir: str | None = _OUTPUT_DIR,
) -> None:
    """Register receptors as pending simulations without starting workers.

    Use this to pre-populate a project queue before launching a separate
    worker deployment (e.g. a long-lived Slurm or Kubernetes deployment
    that drains the queue independently).

    Examples
    --------
    ::

        stilt submit ./hrrr_24h --receptors new_overpasses.csv --batch-id "overpass_2025-01"
        stilt run ./hrrr_24h --backend slurm --n-workers 50
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)

    if receptors_path is not None:
        receptors = read_receptors(receptors_path)
    else:
        receptors = model.receptors

    sim_ids = model.submit(receptors=receptors, batch_id=batch_id)
    typer.echo(f"Registered {len(sim_ids)} simulation(s).")
    if batch_id:
        typer.echo(f"Batch: {batch_id}")


@app.command()
def worker(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    cpus: int = typer.Option(
        1, "--cpus", help="Number of CPU cores to use for within-task parallelism."
    ),
    follow: bool = typer.Option(
        False,
        "--follow/--no-follow",
        help=(
            "Keep polling when the queue is empty. "
            "Use for long-lived streaming deployments (Slurm follow mode)."
        ),
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Drain pending simulations from the project repository.

    Atomically claims and processes simulations until the queue is empty
    (batch mode) or indefinitely (``--follow``).  This is the command run
    by each ``SlurmExecutor`` array task.

    Examples
    --------
    ::

        stilt worker ./hrrr_24h
        stilt worker ./hrrr_24h --cpus 8 --follow
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(
        project=resolved_dir, output_dir=resolved_output, compute_root=compute_root
    )
    worker_loop(model, n_cores=cpus, follow=follow)


@app.command()
def serve(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    cpus: int = typer.Option(
        1, "--cpus", help="Number of CPU cores to use for within-task parallelism."
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run long-lived queue workers that keep polling for new simulations.

    This is the user-facing streaming consumer command. It is equivalent to
    ``stilt worker --follow`` but uses language that better matches the
    deployment model for always-on queue consumers.

    Examples
    --------
    ::

        stilt serve ./hrrr_24h
        stilt serve ./hrrr_24h --cpus 8
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(
        project=resolved_dir, output_dir=resolved_output, compute_root=compute_root
    )
    worker_loop(model, n_cores=cpus, follow=True)


@app.command()
def rebuild(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
) -> None:
    """Rebuild the repository database by scanning simulation output on disk.

    Useful after manual file operations or interrupted runs that left the
    SQLite database out of sync with the filesystem.
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)
    model.repository.rebuild()
    _print_status(model)


@app.command()
def status(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
) -> None:
    """Show simulation counts for a project."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)
    _print_status(model)


@app.command()
def claims(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
    include_expired: bool = typer.Option(
        False,
        "--include-expired",
        help="Include expired claim rows instead of only currently active claims.",
    ),
) -> None:
    """Show current queue claim ownership."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    service = Service(project=resolved_dir, output_dir=resolved_output)
    claims = service.active_claims(include_expired=include_expired)
    if not claims:
        typer.echo("No claims found." if include_expired else "No active claims found.")
        return

    typer.echo("sim_id\tworker_id\tclaimed_at\texpires_at")
    for claim in claims:
        typer.echo(
            f"{claim.sim_id}\t{claim.worker_id}\t"
            f"{claim.claimed_at.isoformat()}\t{claim.expires_at.isoformat()}"
        )


@app.command()
def attempts(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
    sim_id: str | None = typer.Option(
        None,
        "--sim-id",
        help="Filter attempt history to a single simulation id.",
    ),
) -> None:
    """Show recorded execution attempts."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    service = Service(project=resolved_dir, output_dir=resolved_output)
    attempts = service.attempts(sim_id)
    if not attempts:
        typer.echo("No attempts found.")
        return

    typer.echo("attempt_id\tsim_id\toutcome\tterminal\tstarted_at\tfinished_at\terror")
    for attempt in attempts:
        typer.echo(
            f"{attempt.attempt_id}\t{attempt.sim_id}\t{attempt.outcome}\t"
            f"{str(attempt.terminal).lower()}\t{attempt.started_at.isoformat()}\t"
            f"{attempt.finished_at.isoformat() if attempt.finished_at else '-'}\t"
            f"{attempt.error or '-'}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_status(model: Model) -> None:
    """Print a project status summary."""
    repo = model.repository
    status = summarize_queue(model)
    typer.echo(
        "Project: "
        f"{status.project}  total={status.total}  completed={status.completed}  "
        f"running={status.running}  pending={status.pending}  failed={status.failed}"
    )
    batches = repo.all_batches()
    if batches:
        typer.echo("\nBatches:")
        for batch_id, done, tot in batches:
            pct = f"{100 * done / tot:.1f}%" if tot > 0 else "0%"
            check = " \u2713" if done == tot and tot > 0 else ""
            typer.echo(f"  {batch_id}   {done} / {tot}   ({pct}){check}")
