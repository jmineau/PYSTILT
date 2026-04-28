"""Canonical project files and simulation output filenames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIG_FILENAME = "config.yaml"
RECEPTORS_FILENAME = "receptors.csv"
SIMULATIONS_DIRNAME = "simulations"
SIMULATION_INDEX_DIRNAME = "by-id"
SIMULATION_INDEX_DB_FILENAME = "index.sqlite"
PARTICLE_INDEX_DIRNAME = "particles"
FOOTPRINT_INDEX_DIRNAME = "footprints"
CHUNKS_DIRNAME = "chunks"
SIMULATION_LOG_FILENAME = "stilt.log"
SIMULATION_MET_DIRNAME = "met"


def trajectory_filename(sim_id: str) -> str:
    """Return the canonical trajectory filename for *sim_id*."""
    return f"{sim_id}_traj.parquet"


def error_trajectory_filename(sim_id: str) -> str:
    """Return the canonical error-trajectory filename for *sim_id*."""
    return trajectory_filename(sim_id).replace("_traj.parquet", "_error.parquet")


def footprint_filename(sim_id: str, footprint_name: str = "") -> str:
    """Return the canonical footprint filename for *sim_id* and *footprint_name*."""
    suffix = f"_{footprint_name}" if footprint_name else ""
    return f"{sim_id}{suffix}_foot.nc"


def empty_footprint_filename(sim_id: str, footprint_name: str = "") -> str:
    """Return the canonical empty-footprint marker filename."""
    return footprint_filename(sim_id, footprint_name).replace("_foot.nc", "_foot.empty")


@dataclass(frozen=True, slots=True)
class SimulationFiles:
    """Local paths and output keys for one simulation's standard outputs."""

    directory: Path
    sim_id: str

    @staticmethod
    def key_prefix_for(sim_id: str) -> str:
        return f"{SIMULATIONS_DIRNAME}/{SIMULATION_INDEX_DIRNAME}/{sim_id}"

    @staticmethod
    def key_for(sim_id: str, filename: str) -> str:
        return f"{SimulationFiles.key_prefix_for(sim_id)}/{filename}"

    @property
    def key_prefix(self) -> str:
        return self.key_prefix_for(self.sim_id)

    @property
    def log_path(self) -> Path:
        return self.directory / SIMULATION_LOG_FILENAME

    @property
    def met_dir(self) -> Path:
        return self.directory / SIMULATION_MET_DIRNAME

    @property
    def trajectory_path(self) -> Path:
        return self.directory / trajectory_filename(self.sim_id)

    @property
    def error_trajectory_path(self) -> Path:
        return self.directory / error_trajectory_filename(self.sim_id)

    def footprint_path(self, footprint_name: str = "") -> Path:
        return self.directory / footprint_filename(self.sim_id, footprint_name)

    def empty_footprint_path(self, footprint_name: str = "") -> Path:
        return self.directory / empty_footprint_filename(self.sim_id, footprint_name)

    def key(self, filename: str | Path) -> str:
        return self.key_for(self.sim_id, Path(filename).name)

    def write_empty_footprint_marker(self, footprint_name: str = "") -> Path:
        marker = self.empty_footprint_path(footprint_name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
        return marker

    def clear_empty_footprint_marker(self, footprint_name: str = "") -> None:
        self.empty_footprint_path(footprint_name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ProjectFiles:
    """Local paths plus output key conventions for one project root."""

    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def receptors_path(self) -> Path:
        return self.root / RECEPTORS_FILENAME

    @property
    def simulations_dir(self) -> Path:
        return self.root / SIMULATIONS_DIRNAME

    @property
    def by_id_dir(self) -> Path:
        return self.simulations_dir / SIMULATION_INDEX_DIRNAME

    @property
    def index_db_path(self) -> Path:
        return self.simulations_dir / SIMULATION_INDEX_DB_FILENAME

    @property
    def particle_index_dir(self) -> Path:
        return self.simulations_dir / PARTICLE_INDEX_DIRNAME

    @property
    def footprint_index_dir(self) -> Path:
        return self.simulations_dir / FOOTPRINT_INDEX_DIRNAME

    @property
    def chunks_dir(self) -> Path:
        return self.root / CHUNKS_DIRNAME

    def simulation(self, sim_id: str) -> SimulationFiles:
        return SimulationFiles(self.by_id_dir / sim_id, sim_id)

    @staticmethod
    def config_key() -> str:
        return CONFIG_FILENAME

    @staticmethod
    def receptors_key() -> str:
        return RECEPTORS_FILENAME

    @staticmethod
    def particle_index_key(filename: str) -> str:
        return f"{SIMULATIONS_DIRNAME}/{PARTICLE_INDEX_DIRNAME}/{filename}"

    @staticmethod
    def footprint_index_key(filename: str) -> str:
        return f"{SIMULATIONS_DIRNAME}/{FOOTPRINT_INDEX_DIRNAME}/{filename}"

    @staticmethod
    def simulation_prefix() -> str:
        return f"{SIMULATIONS_DIRNAME}/{SIMULATION_INDEX_DIRNAME}"
