"""Durable output store backends."""

from __future__ import annotations

import posixpath
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import fsspec

from .files import ProjectFiles, SimulationFiles
from .layout import uri_join

if TYPE_CHECKING:
    from stilt.simulation import Simulation


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
class Store(Protocol):
    """Durable file access independent of the local compute workspace."""

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def local_path(self, key: str) -> Path: ...
    def publish_simulation(self, sim: Simulation) -> None: ...


class FsspecStore:
    """Durable store backed by an ``fsspec`` filesystem."""

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
        return uri_join(root, clean)

    def _key_uri(self, key: str) -> str:
        """Return a fully-qualified URI/path for use with `fsspec.open_local()`."""
        clean = key.strip("/")
        if not clean:
            return self.output_dir
        joined = uri_join(self.output_dir, clean)
        if "://" in joined:
            return joined
        return str(Path(joined).resolve())

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
        """Copy one local file into the durable store under *key*."""
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
        """Publish the standard durable outputs produced by one simulation."""
        files = SimulationFiles(sim.directory, str(sim.id))
        is_local = _is_local_filesystem_protocol(self.fs.protocol)
        log_path = files.log_path
        traj_path = files.trajectory_path
        error_path = files.error_trajectory_path

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

        _publish(log_path, files.key(log_path))
        _publish(
            traj_path,
            files.key(traj_path),
            ProjectFiles.particle_index_key(traj_path.name),
        )
        _publish(error_path, files.key(error_path))
        for footprint in sorted(sim.directory.glob(f"{files.sim_id}*_foot.nc")):
            _publish(
                footprint,
                files.key(footprint),
                ProjectFiles.footprint_index_key(footprint.name),
            )
        for marker in sorted(sim.directory.glob(f"{files.sim_id}*_foot.empty")):
            _publish(marker, files.key(marker))
