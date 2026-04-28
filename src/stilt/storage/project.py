"""STILT-aware project storage facade."""

from __future__ import annotations

import tempfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from .files import ProjectFiles, SimulationFiles
from .store import Store

if TYPE_CHECKING:
    from stilt.config import ModelConfig
    from stilt.receptor import Receptor
    from stilt.simulation import Simulation


class Storage:
    """STILT-aware output storage facade over a lower-level store backend."""

    def __init__(
        self,
        project_dir: Path,
        output_dir: Path,
        store: Store,
        *,
        is_cloud_project: bool = False,
    ) -> None:
        self.project_dir = project_dir
        self.output_dir = output_dir
        self.store = store
        self.is_cloud_project = is_cloud_project

    def _config_bytes(self, config: ModelConfig) -> bytes:
        """Serialize one in-memory config for output bootstrap."""
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="pystilt_config_",
                suffix=".yaml",
                delete=False,
            ) as handle:
                tmp_name = handle.name
            config.to_yaml(tmp_name)
            return Path(tmp_name).read_bytes()
        finally:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)

    def publish_config(self, config: ModelConfig) -> None:
        """Persist config needed to reconstruct the project."""
        self.store.write_bytes(ProjectFiles.config_key(), self._config_bytes(config))

    def _receptor_bytes(self, receptors: list[Receptor]) -> bytes:
        """Serialize receptors to a CSV compatible with ``read_receptors()``."""
        import csv

        rows = StringIO()
        writer = csv.DictWriter(
            rows,
            fieldnames=[
                "r_idx",
                "time",
                "longitude",
                "latitude",
                "altitude",
                "altitude_ref",
            ],
        )
        writer.writeheader()
        for idx, receptor in enumerate(receptors):
            for lon, lat, altitude in zip(
                receptor.longitudes,
                receptor.latitudes,
                receptor.altitudes,
                strict=False,
            ):
                writer.writerow(
                    {
                        "r_idx": idx,
                        "time": receptor.time.isoformat(sep=" "),
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "altitude": float(altitude),
                        "altitude_ref": receptor.altitude_ref,
                    }
                )
        return rows.getvalue().encode()

    def publish_receptors(
        self,
        receptors: list[Receptor] | None = None,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        """Persist receptor inputs needed to reconstruct the project."""
        if receptors is not None and source_path is not None:
            raise TypeError(
                "publish_receptors() accepts either receptors or source_path, not both."
            )
        if source_path is not None:
            data = Path(source_path).read_bytes()
        else:
            data = self._receptor_bytes(receptors or [])
        self.store.write_bytes(ProjectFiles.receptors_key(), data)

    def load_config(self) -> ModelConfig:
        """Load project config from the local project root or output storage."""
        from stilt.config import ModelConfig

        config_path = ProjectFiles(self.project_dir).config_path
        if config_path.exists():
            return ModelConfig.from_yaml(config_path)
        if not self.store.exists(ProjectFiles.config_key()):
            raise FileNotFoundError(
                f"No config.yaml found in {self.project_dir} or {self.output_dir}. "
                "Create one with ModelConfig.to_yaml()."
            )
        return ModelConfig.from_yaml(self.store.local_path(ProjectFiles.config_key()))

    def receptor_source_path(self) -> Path | None:
        """Return the best local path to output project receptors, if present."""
        local_path = ProjectFiles(self.project_dir).receptors_path
        if local_path.exists():
            return local_path
        if self.store.exists(ProjectFiles.receptors_key()):
            return self.store.local_path(ProjectFiles.receptors_key())
        return None

    def load_receptors(self) -> list[Receptor] | None:
        """Load project receptors from output storage when available."""
        from stilt.receptor import read_receptors

        source = self.receptor_source_path()
        if source is None:
            return None
        return read_receptors(source)

    def resolve(self, sim_id: str, output_path: Path) -> Path | None:
        """Return a local path to a output when it exists."""
        if output_path.exists():
            return output_path
        key = SimulationFiles.key_for(sim_id, output_path.name)
        if self.store.exists(key):
            return self.store.local_path(key)
        return None

    def exists(self, sim_id: str, output_path: Path) -> bool:
        """Return whether one simulation output already exists durably."""
        return self.resolve(sim_id, output_path) is not None

    def publish_simulation(self, sim: Simulation) -> None:
        """Publish one simulation's standard outputs."""
        self.store.publish_simulation(sim)
