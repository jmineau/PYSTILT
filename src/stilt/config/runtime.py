"""Runtime-only settings for deployment and worker bootstrap."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _RuntimeEnvSettings(BaseSettings):
    """Environment-backed loader for runtime-only settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    db_url: str | None = Field(default=None, alias="PYSTILT_DB_URL")
    cache_dir: Path | None = Field(default=None, alias="PYSTILT_CACHE_DIR")
    compute_root: Path | None = Field(default=None, alias="PYSTILT_COMPUTE_ROOT")
    max_rows: int | None = Field(default=None, alias="PYSTILT_MAX_ROWS", ge=1)


class RuntimeSettings(BaseModel):
    """Typed runtime settings shared across CLI, workers, and executors."""

    model_config = ConfigDict(extra="ignore")

    db_url: str | None = Field(
        default=None,
        description="Database URL for a shared simulation index backend.",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Directory for cached downloads and reusable runtime artifacts.",
    )
    compute_root: Path | None = Field(
        default=None,
        description="Scratch or worker-local root used for active simulation workdirs.",
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of index rows to claim or process in one batch.",
    )

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        """Build runtime settings from the process environment."""
        loaded = _RuntimeEnvSettings()
        return cls(
            db_url=loaded.db_url,
            cache_dir=loaded.cache_dir,
            compute_root=loaded.compute_root,
            max_rows=loaded.max_rows,
        )


def resolve_runtime_settings(
    runtime: RuntimeSettings | None = None,
) -> RuntimeSettings:
    """Return explicit runtime settings or load them from the environment."""
    if runtime is not None:
        return runtime
    return RuntimeSettings.from_env()


__all__ = ["RuntimeSettings", "resolve_runtime_settings"]
