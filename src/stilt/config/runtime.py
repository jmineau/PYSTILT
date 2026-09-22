"""Runtime-only settings for deployment and worker bootstrap."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """
    Where a deployment keeps its queue, cache, and scratch: read from
    ``PYSTILT_*`` environment variables, or passed explicitly.

    Nothing here changes a simulation's result, only where and how it runs;
    science configuration lives in :class:`~stilt.config.ModelConfig`.
    """

    model_config = SettingsConfigDict(env_prefix="PYSTILT_", extra="ignore")

    db_url: str | None = Field(
        default=None, description="Postgres URL for the shared work queue."
    )
    cache_dir: Path | None = Field(
        default=None, description="Local cache for downloads from a remote store."
    )
    compute_root: Path | None = Field(
        default=None, description="Scratch parent for worker simulation directories."
    )


__all__ = ["RuntimeSettings"]
