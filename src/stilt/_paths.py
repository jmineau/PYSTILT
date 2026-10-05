"""Paths as a user writes them: ``~`` and environment variables expanded, made absolute."""

from __future__ import annotations

import os
from pathlib import Path


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
