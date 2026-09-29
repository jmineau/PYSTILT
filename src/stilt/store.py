"""
Storage for project files.

A store reads and writes bytes by key. A key is a file's path relative to
the project root, such as
``simulations/by-id/<receptor_id>/<variant>/<receptor_id>_traj.parquet``.
:class:`LocalStore` keeps the files in a local directory and
:class:`FsspecStore` in any fsspec filesystem (``s3://``, ``gs://``).
"""

from __future__ import annotations

import posixpath
import shutil
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import fsspec


def is_uri(root: str | Path) -> bool:
    """Return whether *root* is a URI such as ``s3://bucket/project``."""
    return "://" in str(root)


@runtime_checkable
class Store(Protocol):
    """Interface for reading and writing project files by key."""

    def exists(self, key: str) -> bool:
        """Return whether *key* exists."""
        ...

    def read_bytes(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        ...

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write *data* under *key*."""
        ...

    def publish_file(self, local_path: str | Path, key: str) -> None:
        """Copy a local file into the store under *key*. A missing file is skipped."""
        ...

    def local_path(self, key: str) -> Path:
        """Return a local path to the file stored under *key*."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key*. A missing key is not an error."""
        ...


class LocalStore:
    """
    Store backed by a local directory.

    ``publish_file`` copies to a temporary file and renames it into place, so
    a reader never sees a partly written file.

    Parameters
    ----------
    root : str or Path
        Directory that keys are relative to.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def __repr__(self) -> str:
        return f"LocalStore({str(self.root)!r})"

    def local_path(self, key: str) -> Path:
        """Return the absolute local path of *key*."""
        return self.root / key.strip("/")

    def exists(self, key: str) -> bool:
        """Return whether *key* exists."""
        return self.local_path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        return self.local_path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write *data* under *key*, creating parent directories."""
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def publish_file(self, local_path: str | Path, key: str) -> None:
        """Copy a local file into the store under *key*. A missing file is skipped."""
        src = Path(local_path)
        if not src.exists():
            return
        target = self.local_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == target.resolve():
            return
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            shutil.copy2(src, tmp)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        """Remove the file for *key* if it exists."""
        self.local_path(key).unlink(missing_ok=True)


class FsspecStore:
    """
    Store backed by an fsspec filesystem such as ``s3://`` or ``gs://``.

    ``local_path`` downloads a file once into a local cache that mirrors the
    key layout. Writing or deleting a key drops its cached copy, so a
    rewritten file is downloaded again.

    Parameters
    ----------
    root : str
        URI that keys are relative to.
    cache_dir : str or Path, optional
        Local directory for downloaded files. A temporary directory is
        created on first use when omitted.
    """

    def __init__(self, root: str, cache_dir: str | Path | None = None) -> None:
        self.root = root.rstrip("/")
        self.fs, self._fs_root = fsspec.core.url_to_fs(self.root)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __repr__(self) -> str:
        return f"FsspecStore({self.root!r})"

    def _cache(self) -> Path:
        """Return the download cache directory, creating a temporary one if needed."""
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="pystilt_cache_"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _fs_key(self, key: str) -> str:
        """Return the path of *key* inside the fsspec filesystem."""
        clean = key.strip("/")
        root = str(self._fs_root).rstrip("/")
        return f"{root}/{clean}" if root else clean

    def exists(self, key: str) -> bool:
        """Return whether *key* exists."""
        return self.fs.exists(self._fs_key(key))

    def read_bytes(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        return self.fs.cat(self._fs_key(key))

    def _forget(self, key: str) -> None:
        """Drop the cached copy of *key*, if any."""
        if self._cache_dir is not None:
            (self._cache_dir / key.strip("/")).unlink(missing_ok=True)

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write *data* under *key*."""
        fs_key = self._fs_key(key)
        parent = posixpath.dirname(fs_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(fs_key, "wb") as handle:
            handle.write(data)
        self._forget(key)

    def publish_file(self, local_path: str | Path, key: str) -> None:
        """Upload a local file to *key*. A missing file is skipped."""
        src = Path(local_path)
        if not src.exists():
            return
        fs_key = self._fs_key(key)
        parent = posixpath.dirname(fs_key)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.put_file(str(src), fs_key)
        self._forget(key)

    def local_path(self, key: str) -> Path:
        """Return a local copy of *key*, downloading it on first use."""
        local = self._cache() / key.strip("/")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(local.suffix + ".tmp")
            self.fs.get_file(self._fs_key(key), str(tmp))
            tmp.replace(local)
        return local

    def delete(self, key: str) -> None:
        """Remove *key* if it exists."""
        fs_key = self._fs_key(key)
        if self.fs.exists(fs_key):
            self.fs.rm(fs_key)
        self._forget(key)


def make_store(root: str | Path, *, cache_dir: str | Path | None = None) -> Store:
    """
    Return the store for a project root.

    Parameters
    ----------
    root : str or Path
        Local directory or URI.
    cache_dir : str or Path, optional
        Download cache, used only for a URI.

    Returns
    -------
    Store
        :class:`FsspecStore` for a URI, else :class:`LocalStore`.
    """
    if is_uri(root):
        return FsspecStore(str(root), cache_dir=cache_dir)
    return LocalStore(root)


__all__ = [
    "FsspecStore",
    "LocalStore",
    "Store",
    "is_uri",
    "make_store",
]
