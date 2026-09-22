"""
Output store backends.

A store maps *keys* (POSIX-style relative paths such as
``simulations/by-id/<sim_id>/<sim_id>_traj.parquet``) onto bytes. The key is
the only address an output has; local filesystem paths are derived from it.
"""

from __future__ import annotations

import posixpath
import shutil
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import fsspec


def is_uri(root: str | Path) -> bool:
    """Return True when *root* is an ``scheme://`` URI rather than a local path."""
    return "://" in str(root)


def uri_join(root: str, *parts: str) -> str:
    """Join path fragments onto a local path or object-store URI."""
    clean = [p.strip("/") for p in parts if p and p.strip("/")]
    if is_uri(root):
        base = root.rstrip("/")
        return f"{base}/{'/'.join(clean)}" if clean else base
    path = Path(root)
    for part in clean:
        path /= part
    return str(path)


@runtime_checkable
class Store(Protocol):
    """Byte storage addressed by canonical output keys."""

    def exists(self, key: str) -> bool:
        """Return whether *key* currently exists."""
        ...

    def read_bytes(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        ...

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write *data* under *key*."""
        ...

    def publish_file(self, local_path: str | Path, key: str) -> None:
        """Copy one local file into the store under *key* (no-op if missing)."""
        ...

    def local_path(self, key: str) -> Path:
        """Return a local filesystem path holding the bytes under *key*."""
        ...


class LocalStore:
    """
    Store backed by a local directory.

    ``publish_file`` writes to a sibling ``.tmp`` file and then renames it onto
    the final key, so concurrent readers never observe a partial file.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def __repr__(self) -> str:
        return f"LocalStore({str(self.root)!r})"

    def path(self, key: str) -> Path:
        """Return the absolute local path for *key*."""
        return self.root / key.strip("/")

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def publish_file(self, local_path: str | Path, key: str) -> None:
        src = Path(local_path)
        if not src.exists():
            return
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == target.resolve():
            return
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            shutil.copy2(src, tmp)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    def local_path(self, key: str) -> Path:
        return self.path(key)


class FsspecStore:
    """
    Store backed by an ``fsspec`` filesystem (``s3://``, ``gs://``, ``memory://``, ...).

    ``local_path`` downloads through ``simplecache`` into *cache_dir* (a temp
    directory when omitted).
    """

    def __init__(self, root: str, cache_dir: str | Path | None = None) -> None:
        self.root = root.rstrip("/")
        self.fs, self._fs_root = fsspec.core.url_to_fs(self.root)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __repr__(self) -> str:
        return f"FsspecStore({self.root!r})"

    def _cache(self) -> Path:
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="pystilt_cache_"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _fs_key(self, key: str) -> str:
        """Return the filesystem-native path for *key*."""
        clean = key.strip("/")
        root = str(self._fs_root).rstrip("/")
        return f"{root}/{clean}" if root else clean

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._fs_key(key))

    def read_bytes(self, key: str) -> bytes:
        return self.fs.cat(self._fs_key(key))

    def write_bytes(self, key: str, data: bytes) -> None:
        fs_key = self._fs_key(key)
        parent = posixpath.dirname(fs_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(fs_key, "wb") as handle:
            handle.write(data)

    def publish_file(self, local_path: str | Path, key: str) -> None:
        src = Path(local_path)
        if not src.exists():
            return
        fs_key = self._fs_key(key)
        parent = posixpath.dirname(fs_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.put_file(str(src), fs_key)

    def local_path(self, key: str) -> Path:
        local = fsspec.open_local(
            f"simplecache::{uri_join(self.root, key)}",
            simplecache={"cache_storage": str(self._cache())},
        )
        if not isinstance(local, str):
            raise TypeError(f"Expected one local path for {key!r}, got {type(local)!r}")
        return Path(local)


def make_store(root: str | Path, *, cache_dir: str | Path | None = None) -> Store:
    """Return a ``LocalStore`` for local paths or an ``FsspecStore`` for URIs."""
    if is_uri(root):
        return FsspecStore(str(root), cache_dir=cache_dir)
    return LocalStore(root)


__all__ = [
    "FsspecStore",
    "LocalStore",
    "Store",
    "is_uri",
    "make_store",
    "uri_join",
]
