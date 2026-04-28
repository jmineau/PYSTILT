"""Helpers for rebuilding durable indexes from canonical outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from stilt.receptor import Receptor
from stilt.storage import (
    ProjectFiles,
    SimulationFiles,
    error_trajectory_filename,
    make_store,
    trajectory_filename,
)

from .protocol import OutputSummary


@dataclass(frozen=True, slots=True)
class ScannedSimulation:
    """Durable output snapshot for one simulation id."""

    sim_id: str
    summary: OutputSummary
    receptor: Receptor | None


def _footprint_name_from_filename(
    sim_id: str,
    filename: str,
    *,
    suffix: str,
) -> str | None:
    """Return the footprint name encoded in one durable output filename."""
    if not filename.endswith(suffix):
        return None
    without_suffix = filename[: -len(suffix)]
    if without_suffix == sim_id:
        return ""
    prefix = f"{sim_id}_"
    if without_suffix.startswith(prefix):
        return without_suffix[len(prefix) :]
    return None


def _receptor_from_parquet(parquet_file: Path | None) -> Receptor | None:
    """Return the receptor stored in parquet metadata, if present."""
    if parquet_file is None:
        return None
    try:
        meta = pq.ParquetFile(parquet_file).schema_arrow.metadata
        receptor_blob = None if meta is None else meta.get(b"stilt:receptor")
        if receptor_blob is None:
            return None
        return Receptor.from_dict(json.loads(receptor_blob))
    except Exception:
        return None


def scan_durable_simulations(output_root: str | Path) -> list[ScannedSimulation]:
    """Scan one durable output root and summarize every simulation directory."""
    store = make_store(output_root)
    grouped: dict[str, set[str]] = {}
    for key in store.list_prefix(ProjectFiles.simulation_prefix()):
        parts = PurePosixPath(key).parts
        if len(parts) < 4 or parts[:2] != ("simulations", "by-id"):
            continue
        sim_id = parts[2]
        grouped.setdefault(sim_id, set()).add(parts[-1])

    records: list[ScannedSimulation] = []
    for sim_id in sorted(grouped):
        filenames = grouped[sim_id]
        traj_name = trajectory_filename(sim_id)
        error_name = error_trajectory_filename(sim_id)
        footprints: dict[str, str] = {}

        for filename in filenames:
            footprint_name = _footprint_name_from_filename(
                sim_id,
                filename,
                suffix="_foot.nc",
            )
            if footprint_name is not None:
                footprints[footprint_name] = "complete"
        for filename in filenames:
            footprint_name = _footprint_name_from_filename(
                sim_id,
                filename,
                suffix="_foot.empty",
            )
            if footprint_name is None:
                continue
            complete_name = (
                f"{sim_id}{'_' + footprint_name if footprint_name else ''}_foot.nc"
            )
            if complete_name in filenames:
                continue
            footprints.setdefault(footprint_name, "complete-empty")

        summary = OutputSummary(
            traj_present=traj_name in filenames,
            error_traj_present=error_name in filenames,
            log_present="stilt.log" in filenames,
            footprints=footprints,
        )
        if not (
            summary.traj_present
            or summary.error_traj_present
            or summary.log_present
            or summary.footprints
        ):
            continue

        parquet_key = None
        if traj_name in filenames:
            parquet_key = SimulationFiles.key_for(sim_id, traj_name)
        elif error_name in filenames:
            parquet_key = SimulationFiles.key_for(sim_id, error_name)

        records.append(
            ScannedSimulation(
                sim_id=sim_id,
                summary=summary,
                receptor=(
                    _receptor_from_parquet(store.local_path(parquet_key))
                    if parquet_key
                    else None
                ),
            )
        )
    return records
