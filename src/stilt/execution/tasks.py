"""Simulation task data types and worker planning helpers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from stilt.config import FootprintConfig, STILTParams
from stilt.meteorology import MetSource
from stilt.receptor import Receptor
from stilt.simulation import SimID
from stilt.storage import ProjectFiles, SimulationFiles, Storage

if TYPE_CHECKING:
    from stilt.model import Model


class SimulationTask(BaseModel):
    """
    Serialisable bundle of everything a worker needs to run one simulation.

    Passed as the single argument to :func:`~stilt.execution.execute_task` so
    it can be pickled and shipped to a remote executor (local subprocess, Slurm
    task, Kubernetes Job) without needing shared runtime objects. Atomic claim
    handling stays outside the task on claim-capable index backends.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    compute_root: Path
    sim_id: SimID
    meteorology: MetSource
    receptor: Receptor
    params: STILTParams
    skip_existing: bool = False
    foot_configs: dict[str, FootprintConfig] | None = None
    storage: Storage


@dataclass
class SimulationResult:
    """Typed result contract returned by one worker-run simulation."""

    sim_id: SimID
    status: Literal["complete", "complete-empty", "failed", "error", "interrupted"]
    traj_present: bool = False
    traj_path: Path | None = None
    error_traj_path: Path | None = None
    log_path: Path | None = None
    wrote_traj: bool = False
    foot_paths: list[Path] = field(default_factory=list)
    empty_footprints: list[str] = field(default_factory=list)
    footprint_statuses: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    def __repr__(self) -> str:
        """Compact developer-facing simulation result representation."""
        return f"SimulationResult(sim_id={self.sim_id!r}, status={self.status!r})"


def build_simulation_task(
    model: Model,
    sim_id: str,
    *,
    foot_configs: dict[str, FootprintConfig] | None = None,
    skip_existing: bool | None = None,
) -> SimulationTask:
    """Build one worker task from output project inputs."""
    sid = SimID(sim_id)
    return SimulationTask(
        compute_root=model.compute_root,
        sim_id=sid,
        meteorology=model.mets[sid.met],
        receptor=model.receptors[sid.receptor],
        params=model.config.to_stilt_params(),
        skip_existing=(
            model.config.skip_existing if skip_existing is None else skip_existing
        ),
        foot_configs=foot_configs,
        storage=model.storage,
    )


def _planned_foot_configs(
    model: Model,
    sim_id: str,
    *,
    skip_existing: bool | None = None,
) -> dict[str, FootprintConfig] | None:
    """
    Return the footprint configs that still need work for one simulation.

    Checks the output store (filesystem) for per-footprint existence. Workers
    must never touch the index — that belongs to the submit side.
    """
    all_foot_configs = dict(model.config.footprints)
    if not all_foot_configs:
        return None
    resolved_skip = (
        model.config.skip_existing if skip_existing is None else skip_existing
    )
    if not resolved_skip:
        return all_foot_configs
    sim_dir = ProjectFiles(model.storage.output_dir).by_id_dir / sim_id
    files = SimulationFiles(directory=sim_dir, sim_id=sim_id)
    pending = {
        name: cfg
        for name, cfg in all_foot_configs.items()
        if not (
            model.storage.exists(sim_id, files.footprint_path(name))
            or model.storage.exists(sim_id, files.empty_footprint_path(name))
        )
    }
    return pending or None


def plan_simulation_task(
    model: Model,
    sim_id: str,
    *,
    skip_existing: bool | None = None,
) -> SimulationTask:
    """
    Plan one runnable worker task for a sim already known to need work.

    The caller is expected to pass sim IDs from ``index.pending_trajectories()``
    or ``index.claim_one()``, both of which filter on the SQL completion
    predicate — so this function does not re-check skip-existing.
    """
    return build_simulation_task(
        model,
        sim_id,
        foot_configs=_planned_foot_configs(
            model,
            sim_id,
            skip_existing=skip_existing,
        ),
        skip_existing=skip_existing,
    )
