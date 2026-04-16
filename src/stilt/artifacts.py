"""Artifact layout and durable artifact storage helpers.

This module is the single source of truth for both durable artifact keys and
their local filesystem layout under a project or simulation root.
"""

from __future__ import annotations

import posixpath
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import fsspec

if TYPE_CHECKING:
    from stilt.simulation import Simulation


def resolve_directory(
    directory: str | Path | None = None, *, prefix: str = "pystilt_"
) -> Path:
    """Return a resolved directory path, creating a temp root when omitted."""
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=prefix))
    directory = Path(directory)
    if directory.parent == Path("."):
        directory = directory.resolve()
    return directory


def is_cloud_project(project: str) -> bool:
    """Return True when *project* is an object-storage URI."""
    return project.startswith(("s3://", "gs://"))


def project_slug(project: str) -> str:
    """Derive a DNS-safe/local-safe slug from a project path or URI."""
    raw = project.rstrip("/")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    parts = [part for part in raw.split("/") if part]
    candidate = parts[-1] if parts else "project"
    slug = candidate.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


def project_config_relpath() -> Path:
    """Return the relative path for the durable project config."""
    return Path("config.yaml")


def project_receptors_relpath() -> Path:
    """Return the relative path for the durable project receptors CSV."""
    return Path("receptors.csv")


def simulations_relpath() -> Path:
    """Return the relative path for the simulations root."""
    return Path("simulations")


def simulation_index_relpath() -> Path:
    """Return the relative path for the by-id simulation index."""
    return simulations_relpath() / "by-id"


def simulation_state_db_relpath() -> Path:
    """Return the relative path for local simulation state storage."""
    return simulations_relpath() / "state.sqlite"


def particle_index_relpath(filename: str) -> Path:
    """Return the relative particle-index path for one trajectory artifact."""
    return simulations_relpath() / "particles" / filename


def footprint_index_relpath(filename: str) -> Path:
    """Return the relative footprint-index path for one footprint artifact."""
    return simulations_relpath() / "footprints" / filename


def simulation_dir_relpath(sim_id: str) -> Path:
    """Return the relative simulation directory path for *sim_id*."""
    return simulation_index_relpath() / sim_id


def simulation_artifact_relpath(sim_id: str, filename: str) -> Path:
    """Return the relative by-simulation artifact path for one filename."""
    return simulation_dir_relpath(sim_id) / filename


def simulation_log_relpath(sim_id: str) -> Path:
    """Return the relative path for one simulation log."""
    return simulation_artifact_relpath(sim_id, "stilt.log")


def simulation_met_relpath(sim_id: str) -> Path:
    """Return the relative met staging directory for one simulation."""
    return simulation_dir_relpath(sim_id) / "met"


def trajectory_filename(sim_id: str) -> str:
    """Return the canonical trajectory filename for *sim_id*."""
    return f"{sim_id}_traj.parquet"


def error_trajectory_filename(sim_id: str) -> str:
    """Return the canonical error-trajectory filename for *sim_id*."""
    return f"{sim_id}_error.parquet"


def footprint_filename(sim_id: str, footprint_name: str = "") -> str:
    """Return the canonical footprint filename for *sim_id* and *footprint_name*."""
    suffix = f"_{footprint_name}" if footprint_name else ""
    return f"{sim_id}{suffix}_foot.nc"


def trajectory_relpath(sim_id: str) -> Path:
    """Return the relative trajectory artifact path for *sim_id*."""
    return simulation_artifact_relpath(sim_id, trajectory_filename(sim_id))


def error_trajectory_relpath(sim_id: str) -> Path:
    """Return the relative error-trajectory artifact path for *sim_id*."""
    return simulation_artifact_relpath(sim_id, error_trajectory_filename(sim_id))


def footprint_relpath(sim_id: str, footprint_name: str = "") -> Path:
    """Return the relative footprint artifact path for *sim_id*."""
    return simulation_artifact_relpath(
        sim_id, footprint_filename(sim_id, footprint_name)
    )


def project_config_key() -> str:
    """Return the durable key for the project config file."""
    return project_config_relpath().as_posix()


def project_receptors_key() -> str:
    """Return the durable key for the project receptors CSV."""
    return project_receptors_relpath().as_posix()


def simulation_artifact_prefix(sim_id: str) -> str:
    """Return the durable by-simulation prefix for *sim_id*."""
    return simulation_dir_relpath(sim_id).as_posix()


def simulation_artifact_key(sim_id: str, filename: str) -> str:
    """Return the durable by-simulation key for one artifact filename."""
    return simulation_artifact_relpath(sim_id, filename).as_posix()


def particle_index_key(filename: str) -> str:
    """Return the shared particle index key for a trajectory artifact."""
    return particle_index_relpath(filename).as_posix()


def footprint_index_key(filename: str) -> str:
    """Return the shared footprint index key for a footprint artifact."""
    return footprint_index_relpath(filename).as_posix()


def project_config_path(project_dir: str | Path) -> Path:
    """Return the local path for the project config under *project_dir*."""
    return Path(project_dir) / project_config_relpath()


def project_receptors_path(project_dir: str | Path) -> Path:
    """Return the local path for the project receptors CSV under *project_dir*."""
    return Path(project_dir) / project_receptors_relpath()


def simulations_path(project_dir: str | Path) -> Path:
    """Return the local simulations root under *project_dir*."""
    return Path(project_dir) / simulations_relpath()


def simulation_index_path(project_dir: str | Path) -> Path:
    """Return the local by-id simulation root under *project_dir*."""
    return Path(project_dir) / simulation_index_relpath()


def simulation_dir_path(project_dir: str | Path, sim_id: str) -> Path:
    """Return the local simulation directory for *sim_id* under *project_dir*."""
    return Path(project_dir) / simulation_dir_relpath(sim_id)


def simulation_state_db_path(project_dir: str | Path) -> Path:
    """Return the local SQLite state path under *project_dir*."""
    return Path(project_dir) / simulation_state_db_relpath()


def particle_index_path(project_dir: str | Path, filename: str) -> Path:
    """Return the local particle-index alias path under *project_dir*."""
    return Path(project_dir) / particle_index_relpath(filename)


def footprint_index_path(project_dir: str | Path, filename: str) -> Path:
    """Return the local footprint-index alias path under *project_dir*."""
    return Path(project_dir) / footprint_index_relpath(filename)


def simulation_log_path(sim_dir: str | Path, sim_id: str | None = None) -> Path:
    """Return the local log path for one simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_id or sim_dir.name
    return sim_dir / simulation_log_relpath(sim_id).name


def simulation_met_path(sim_dir: str | Path, sim_id: str | None = None) -> Path:
    """Return the local met staging directory for one simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_id or sim_dir.name
    return sim_dir / simulation_met_relpath(sim_id).name


def trajectory_path(sim_dir: str | Path, sim_id: str | None = None) -> Path:
    """Return the local trajectory path for one simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_id or sim_dir.name
    return sim_dir / trajectory_filename(sim_id)


def error_trajectory_path(sim_dir: str | Path, sim_id: str | None = None) -> Path:
    """Return the local error-trajectory path for one simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_id or sim_dir.name
    return sim_dir / error_trajectory_filename(sim_id)


def footprint_path(
    sim_dir: str | Path,
    footprint_name: str = "",
    *,
    sim_id: str | None = None,
) -> Path:
    """Return the local footprint path for one simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_id or sim_dir.name
    return sim_dir / footprint_filename(sim_id, footprint_name)


def _normalize_output_dir(output_dir: str | Path) -> str:
    """Return a stable output-root string for local paths and URIs."""
    raw = str(output_dir)
    if "://" in raw:
        return raw.rstrip("/")
    return str(Path(raw).resolve())


def _is_local_filesystem_protocol(protocol: object) -> bool:
    """Return True when the resolved filesystem writes directly to local disk."""
    if isinstance(protocol, str):
        return protocol in {"file", "local"}
    if isinstance(protocol, tuple):
        return any(item in {"file", "local"} for item in protocol)
    return False


@runtime_checkable
class ArtifactStore(Protocol):
    """Durable artifact access independent of the local compute workspace."""

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def local_path(self, key: str) -> Path: ...
    def publish_simulation(self, sim: Simulation) -> None: ...


class FsspecArtifactStore:
    """Artifact store backed by an ``fsspec`` filesystem."""

    def __init__(
        self,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = _normalize_output_dir(output_dir)
        self.fs, self.root = fsspec.core.url_to_fs(self.output_dir)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _cache_storage(self) -> Path:
        """Return the local cache directory used for remote `local_path()` calls."""
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="pystilt_cache_"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _full_key(self, key: str) -> str:
        """Join a store-relative key onto the filesystem-specific root prefix."""
        clean = key.strip("/")
        root = str(self.root).rstrip("/")
        if not root:
            return clean
        if not clean:
            return root
        return posixpath.join(root, clean)

    def _key_uri(self, key: str) -> str:
        """Return a fully-qualified URI/path for use with `fsspec.open_local()`."""
        clean = key.strip("/")
        if "://" in self.output_dir:
            if not clean:
                return self.output_dir
            return f"{self.output_dir.rstrip('/')}/{clean}"
        if not clean:
            return self.output_dir
        return str((Path(self.output_dir) / clean).resolve())

    def _relative_key(self, full_key: str) -> str:
        """Strip the store root from a filesystem-reported path."""
        root = str(self.root).rstrip("/")
        if root and full_key.startswith(root + "/"):
            return full_key[len(root) + 1 :]
        return full_key

    def read_bytes(self, key: str) -> bytes:
        return self.fs.cat(self._full_key(key))

    def write_bytes(self, key: str, data: bytes) -> None:
        full_key = self._full_key(key)
        parent = posixpath.dirname(full_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(full_key, "wb") as handle:
            handle.write(data)

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._full_key(key))

    def list_prefix(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        if not self.fs.exists(full_prefix):
            return []
        return sorted(self._relative_key(path) for path in self.fs.find(full_prefix))

    def local_path(self, key: str) -> Path:
        """Return a local path for a durable key, caching remote objects as needed."""
        if _is_local_filesystem_protocol(self.fs.protocol):
            return Path(self._full_key(key))
        local = fsspec.open_local(
            f"simplecache::{self._key_uri(key)}",
            simplecache={"cache_storage": str(self._cache_storage())},
        )
        if not isinstance(local, str):
            raise TypeError(f"Expected one local path for {key!r}, got {type(local)!r}")
        return Path(local)

    def publish_file(self, local_path: str | Path, key: str) -> None:
        """Copy one local artifact into the durable store under *key*."""
        src = Path(local_path)
        if not src.exists():
            return
        if _is_local_filesystem_protocol(self.fs.protocol):
            target = Path(self._full_key(key))
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() == target.resolve():
                return
            shutil.copy2(src, target)
            return
        full_key = self._full_key(key)
        parent = posixpath.dirname(full_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.put_file(str(src), full_key)

    def _publish_local_alias(self, src: Path, key: str) -> None:
        """Create a local symlink alias inside the durable store."""
        target = Path(self._full_key(key))
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == target.resolve():
            return
        target.unlink(missing_ok=True)
        target.symlink_to(src.resolve())

    def publish_simulation(self, sim: Simulation) -> None:
        """Publish the standard durable artifacts produced by one simulation."""
        sim_id = str(sim.id)
        is_local = _is_local_filesystem_protocol(self.fs.protocol)
        log_path = sim.log_path
        traj_path = sim.trajectories_path
        error_path = sim.error_trajectories_path

        def _publish(src: Path, canonical_key: str, *index_keys: str) -> None:
            if not src.exists():
                return
            self.publish_file(src, canonical_key)
            local_canonical = self.local_path(canonical_key) if is_local else None
            for key in index_keys:
                if is_local:
                    assert local_canonical is not None
                    self._publish_local_alias(local_canonical, key)
                else:
                    self.publish_file(src, key)

        _publish(
            log_path,
            simulation_artifact_key(sim_id, log_path.name),
        )
        _publish(
            traj_path,
            simulation_artifact_key(sim_id, traj_path.name),
            particle_index_key(traj_path.name),
        )
        _publish(
            error_path,
            simulation_artifact_key(sim_id, error_path.name),
        )
        for footprint in sorted(sim.directory.glob(f"{sim_id}*_foot.nc")):
            _publish(
                footprint,
                simulation_artifact_key(sim_id, footprint.name),
                footprint_index_key(footprint.name),
            )
