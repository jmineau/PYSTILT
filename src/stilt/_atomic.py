"""Writing a file so that no reader or other writer sees it half-written."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """
    Yield a temporary path beside *path*, and rename it onto *path* on success.

    Write the file to the yielded path inside the ``with`` block. Two workers
    can write the same file at once (the settings of a run they both start,
    or a receptor run twice by accident). Each writes under its own name and
    renames it into place, so neither can move or remove the other's
    half-written file, and the last rename wins. The leading dot keeps
    dataset readers from listing a file in progress. The temporary file is
    removed if the block raises.
    """
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
