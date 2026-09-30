"""Starting a model's work: saving its inputs and handing receptors to workers."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.config import RuntimeSettings
from stilt.service import resolve_queue

from .backends import Executor, JobHandle, LocalHandle, get_executor

if TYPE_CHECKING:
    from stilt.model import Model
    from stilt.project import Project
    from stilt.receptors import Receptor

logger = logging.getLogger(__name__)


def resolve_compute_root(
    project: Project, compute_root: str | Path | None = None
) -> Path:
    """
    Return the scratch directory under which HYSPLIT runs for *project*.

    That is *compute_root* when given, else ``PYSTILT_COMPUTE_ROOT``, else
    ``$TMPDIR/pystilt/<project name>``. The path is absolute and resolved, so
    a worker handed it gets the same directory.
    """
    if compute_root is not None:
        raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
        return Path(raw).resolve()
    configured = RuntimeSettings().compute_root
    if configured is not None:
        return configured.expanduser().resolve()
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return (Path(tmp_root) / "pystilt" / project.name).resolve()


def _save_config(model: Model) -> None:
    """Write the model's config to the project when the project lacks it or differs."""
    project = model.project
    if project.has_config:
        try:
            if project.load_config() == model.config:
                return
        except Exception:  # an unreadable config.yaml is replaced
            logger.warning("replacing unreadable %s", project.config_path)
    project.save_config(model.config)


def register(model: Model, receptors: Iterable[Receptor] | None = None) -> list[str]:
    """
    Save a model's settings and receptors to its project.

    Workers rebuild the model from the project alone, so :func:`run` calls
    this first. ``config.yaml`` is written when the project has none or its
    settings differ from the model's, with only the settings that were set.
    Receptors not yet in ``receptors.csv`` are appended to it. When a work
    queue is configured (``PYSTILT_DB_URL``), the receptors are added to it.

    Parameters
    ----------
    model : Model
        Model whose inputs to save.
    receptors : iterable of Receptor, optional
        Receptors to add to the project. Defaults to the model's own.

    Returns
    -------
    list of str
        Ids of the receptors registered, including any the project already
        had.
    """
    _save_config(model)
    batch = list(model.receptors) if receptors is None else list(receptors)
    if receptors is not None or not model.receptors.from_project:
        model.project.add_receptors(batch)
    receptor_ids = [r.id for r in batch]
    queue = resolve_queue(RuntimeSettings())
    if queue is not None:
        queue.register(receptor_ids)
    return receptor_ids


def run(
    model: Model,
    executor: Executor | None = None,
    *,
    skip_existing: bool = True,
    wait: bool = True,
    compute_root: str | Path | None = None,
) -> JobHandle:
    """
    Run every simulation of a model that has not finished.

    Saves the settings and receptors to the project (:func:`register`), then
    starts workers for each receptor with missing results. A worker runs
    HYSPLIT once for each distinct set of transport settings whose particles
    are missing, then calculates the footprint of every variant that has a
    grid.

    Parameters
    ----------
    model : Model
        Model to run.
    executor : Executor, optional
        Where to run the workers. Defaults to the one set by
        ``config.execution`` (local processes unless configured).
    skip_existing : bool, default True
        Skip simulations whose outputs all exist. ``False`` runs every
        simulation again.
    wait : bool, default True
        Block until the workers finish. With ``False`` a Slurm or Kubernetes
        run returns once it is submitted. A local run always finishes before
        this returns.
    compute_root : str or Path, optional
        Scratch directory under which HYSPLIT runs
        (:func:`resolve_compute_root`).

    Returns
    -------
    JobHandle
        Handle to the started workers.
    """
    resolved_executor = executor or get_executor(model.config.execution or {})

    receptor_ids = register(model)
    if not receptor_ids:
        logger.info("run: no receptors configured — nothing to do")
        return LocalHandle()

    pending = (
        model.simulations.incomplete().receptors if skip_existing else receptor_ids
    )
    if not pending:
        logger.info("run: all simulations already complete — nothing to do")
        return LocalHandle()

    logger.info(
        "run(%s): starting %s workers for %d receptors",
        ", ".join(model.variants),
        resolved_executor.dispatch,
        len(pending),
    )
    handle = resolved_executor.start(
        pending,
        project=model.project.root,
        compute_root=str(resolve_compute_root(model.project, compute_root)),
        skip_existing=skip_existing,
    )
    if wait:
        handle.wait()
    return handle


__all__ = ["register", "resolve_compute_root", "run"]
