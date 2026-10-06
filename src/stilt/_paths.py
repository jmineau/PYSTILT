"""
Paths as a user writes them: ``~`` and environment variables expanded, made absolute.

An output directory may also be on an object store, written as a URL such
as ``s3://bucket/output``. It is then a universal path
(:class:`upath.UPath`), which reads and writes through fsspec, and every
other path stays a :class:`pathlib.Path`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from upath import UPath

    #: A place on this filesystem or on an object store.
    Location = Path | UPath


def absolute(path: str | Path, base: str | Path | None = None) -> Path:
    """
    Return *path* with ``~`` and ``$VARIABLES`` expanded, absolute, and resolved.

    A relative path starts from *base*, or from the working directory when
    *base* is ``None``. Resolved paths compare equal however they were
    reached, and a worker started in another directory finds the same place.
    """
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if base is not None and not expanded.is_absolute():
        expanded = Path(base) / expanded
    return expanded.resolve()


def is_url(path: Any) -> bool:
    """Whether *path* is on a store other than this filesystem, such as ``s3://bucket/output``."""
    if isinstance(path, Path):  # a local universal path is a Path too
        return False
    if hasattr(path, "protocol"):  # a universal path of a store
        return True
    text = str(path)
    return "://" in text and not text.startswith("file://")


def location(path: str | Path | UPath, base: str | Path | None = None) -> Path | UPath:
    """
    Return a place to read or write: a URL as a universal path, anything else as :func:`absolute` does.

    Raises
    ------
    ImportError
        For a URL whose fsspec package (such as ``s3fs``) is not installed,
        when the path is first used.
    """
    if isinstance(path, str) and path.startswith("file://"):
        path = path[len("file://") :]
    if not is_url(path):
        return absolute(path, base)  # type: ignore[arg-type]
    from upath import UPath

    return path if isinstance(path, UPath) else UPath(str(path).rstrip("/"))


@contextmanager
def readable(path: str | Path | UPath) -> Iterator[Path | IO[bytes]]:
    """Yield what pyarrow reads *path* from: the path on this filesystem, an open file on an object store."""
    where = location(path)
    if isinstance(where, Path):
        yield where
        return
    with where.open("rb") as file:
        yield file
