"""
The optional PostgreSQL work queue and Kubernetes manifest helpers.

``stilt pull-worker`` and ``stilt serve`` take receptors from the queue
(:class:`PostgresQueue`). ``stilt.service.kubernetes`` builds manifests for
running those workers on Kubernetes. Local and Slurm runs do not use this
package, and ``model.queue`` is ``None`` unless ``PYSTILT_DB_URL`` is set.
"""

from __future__ import annotations

from .factory import resolve_queue
from .postgres import PostgresQueue

__all__ = [
    "PostgresQueue",
    "resolve_queue",
]
