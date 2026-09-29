"""Settings for where a deployment runs, read from the environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """
    Queue, cache, and scratch locations for a deployment.

    Each field is read from the matching ``PYSTILT_*`` environment variable
    (``PYSTILT_DB_URL`` and so on) unless passed directly. None of them change
    a simulation's result.
    """

    model_config = SettingsConfigDict(env_prefix="PYSTILT_", extra="ignore")

    db_url: str | None = Field(
        default=None, description="PostgreSQL URL of the shared work queue."
    )
    cache_dir: Path | None = Field(
        default=None,
        description=(
            "Local directory for files downloaded from a cloud project. Unset "
            "uses a new temporary directory."
        ),
    )
    compute_root: Path | None = Field(
        default=None,
        description=(
            "Directory where workers run HYSPLIT before copying outputs into "
            "the project. Unset runs inside a local project, and under "
            "``$TMPDIR/pystilt/`` for a cloud project."
        ),
    )


__all__ = ["RuntimeSettings"]
