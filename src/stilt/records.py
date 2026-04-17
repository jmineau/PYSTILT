"""Metadata-oriented artifact records and query accessors for STILT models."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from stilt.artifacts import (
    error_trajectory_path,
    footprint_path,
    simulation_dir_path,
    trajectory_path,
)
from stilt.footprint import Footprint
from stilt.simulation import SimID
from stilt.trajectory import Trajectories

if TYPE_CHECKING:
    from stilt.model import Model


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Metadata-only view of one trajectory or footprint artifact."""

    sim_id: str
    kind: Literal["trajectory", "footprint"]
    met: str
    time: dt.datetime
    location_id: str
    status: str | None
    path: Path | None
    name: str | None = None

    def load(self) -> Trajectories | Footprint:
        """Load the full artifact from its path."""
        if self.path is None:
            raise FileNotFoundError(
                f"No path available for {self.kind} artifact from simulation: {self.sim_id}"
            )
        if self.kind == "trajectory":
            return Trajectories.from_parquet(self.path)
        if self.kind == "footprint":
            return Footprint.from_netcdf(self.path)
        raise ValueError(f"Unknown artifact kind: {self.kind}")


class ModelRecordAccessor:
    """Metadata/query namespace for model trajectory and footprint artifacts."""

    def __init__(self, model: Model):
        self._model = model

    def trajectories(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[ArtifactRecord]:
        """Return metadata records for matching trajectory artifacts."""
        sim_ids = self._model.get_simulation_ids(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )
        statuses = self._model.status.bulk_traj_status(sim_ids)

        records: list[ArtifactRecord] = []
        for sim_id in sim_ids:
            sid = SimID(sim_id)
            status = statuses.get(sim_id)
            resolved: Path | None = None
            if status == "complete":
                sim_dir = simulation_dir_path(self._model.output_dir, sim_id)
                artifact_path = (
                    error_trajectory_path(sim_dir, sim_id)
                    if error
                    else trajectory_path(sim_dir, sim_id)
                )
                resolved = self._model.artifact_locator.resolve(sim_id, artifact_path)
            records.append(
                ArtifactRecord(
                    sim_id=sim_id,
                    kind="trajectory",
                    met=sid.met,
                    time=sid.time,
                    location_id=sid.location_id,
                    status=status,
                    path=resolved,
                )
            )
        return records

    def footprints(
        self,
        name: str,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[ArtifactRecord]:
        """Return metadata records for one named footprint across simulations."""
        sim_ids = self._model._filter_simulation_ids(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )
        statuses = self._model.status.bulk_footprint_status(name, sim_ids)

        records: list[ArtifactRecord] = []
        for sim_id in sim_ids:
            sid = SimID(sim_id)
            status = statuses.get(sim_id)
            resolved: Path | None = None
            if status == "complete":
                sim_dir = simulation_dir_path(self._model.output_dir, sim_id)
                artifact_path = footprint_path(sim_dir, name, sim_id=sim_id)
                resolved = self._model.artifact_locator.resolve(sim_id, artifact_path)
            records.append(
                ArtifactRecord(
                    sim_id=sim_id,
                    kind="footprint",
                    met=sid.met,
                    time=sid.time,
                    location_id=sid.location_id,
                    status=status,
                    path=resolved,
                    name=name,
                )
            )
        return records

    def load(self, record: ArtifactRecord) -> Trajectories | Footprint:
        """Load one full artifact from a metadata record."""
        return record.load()
