"""Service-facing runtime helpers built on top of the core STILT model API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import BatchStatus, QueueStatus, Service, summarize_queue

__all__ = [
    "BatchStatus",
    "QueueStatus",
    "Service",
    "summarize_queue",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .project import BatchStatus, QueueStatus, Service, summarize_queue

        mapping = {
            "BatchStatus": BatchStatus,
            "QueueStatus": QueueStatus,
            "Service": Service,
            "summarize_queue": summarize_queue,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
