"""
Optional Postgres-backed work queue plus Kubernetes deployment helpers.

This is not a general service API.  ``stilt.service`` exposes only the lean
queue (``PostgresQueue`` / ``resolve_queue``) used by ``stilt pull-worker``
and ``stilt serve``, and ``stilt.service.kubernetes`` is the public
namespace for manifest generation.  Local and SLURM workflows never need
this package; ``model.queue`` is ``None`` unless ``PYSTILT_DB_URL`` is set.
"""

from __future__ import annotations

from .factory import resolve_queue
from .postgres import PostgresQueue

__all__ = [
    "PostgresQueue",
    "resolve_queue",
]
