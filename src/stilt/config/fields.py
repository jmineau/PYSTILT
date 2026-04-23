"""Shared field helpers and type aliases for STILT config models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

_MISSING = object()
ConfigVisibility = Literal["public", "advanced", "internal"]


def cfg_field(
    default: Any = _MISSING,
    *,
    description: str,
    visibility: ConfigVisibility = "public",
    target: str | None = None,
    namelist: str | None = None,
    default_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Field with shared config metadata."""
    if default_factory is not None and default is not _MISSING:
        raise TypeError(
            "cfg_field() accepts either default or default_factory, not both."
        )

    meta = dict(kwargs.pop("json_schema_extra", {}) or {})
    meta["visibility"] = visibility
    if target is not None:
        meta["target"] = target
    if namelist is not None:
        meta["namelist"] = namelist

    field_kwargs = {
        "description": description,
        "json_schema_extra": meta,
        **kwargs,
    }
    if default_factory is not None:
        return Field(default_factory=default_factory, **field_kwargs)
    if default is _MISSING:
        default = ...
    return Field(default, **field_kwargs)


def _field_meta(field: Any) -> dict[str, Any]:
    """Return normalized metadata stored on a Pydantic field."""
    return dict(field.json_schema_extra or {})


__all__ = [
    "ConfigVisibility",
    "T",
    "cfg_field",
]
