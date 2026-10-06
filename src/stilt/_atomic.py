"""
Writing a file so that no reader or other writer sees it half-written.

On a filesystem a file is written under a temporary name and renamed into
place. An object store has no rename, and needs none: a single put there is
already atomic, so the file is written directly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    import pyarrow as pa
    from upath import UPath

#: A path on this filesystem, or a universal path on an object store.
P = TypeVar("P", bound="Path | UPath")


@contextmanager
def atomic_path(path: P) -> Iterator[P]:
    """
    Yield where to write *path* so that it appears whole: beside it on a filesystem, itself on an object store.

    Write the file to the yielded path inside the ``with`` block. Two workers
    can write the same file at once (the settings of a run they both start,
    or a receptor run twice by accident). On a filesystem each writes under
    its own name and renames it into place, so neither can move or remove
    the other's half-written file, and the last rename wins. The leading
    dot keeps dataset readers from listing a file in progress. The
    temporary file is removed if the block raises. On an object store the
    last put wins in the same way.
    """
    if not isinstance(path, Path):
        yield path
        return
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_parquet(table: pa.Table, path: P) -> P:
    """
    Write *table* as a zstd-compressed Parquet file, so no reader sees it half-written.

    The parent directory is created when needed. Returns *path*.
    """
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as target:
        if isinstance(target, Path):
            pq.write_table(table, target, compression="zstd")
        else:
            with target.open("wb") as file:
                pq.write_table(table, file, compression="zstd")
    return path
