"""Runtime-only settings for deployment and worker bootstrap."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["RuntimeSettings", "resolve_runtime_settings"]


class _RuntimeEnvSettings(BaseSettings):
    """Environment-backed loader for runtime-only settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    db_url: str | None = Field(default=None, alias="PYSTILT_DB_URL")
    met_archive: Path | None = Field(default=None, alias="STILT_MET_ARCHIVE")
    cache_dir: Path | None = Field(default=None, alias="PYSTILT_CACHE_DIR")
    compute_root: Path | None = Field(default=None, alias="PYSTILT_COMPUTE_ROOT")


class RuntimeSettings(BaseModel):
    """Typed runtime settings shared across CLI, workers, and executors."""

    model_config = ConfigDict(extra="ignore")

    db_url: str | None = Field(default=None)
    met_archive: Path | None = Field(default=None)
    cache_dir: Path | None = Field(default=None)
    compute_root: Path | None = Field(default=None)

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        """Build runtime settings from the process environment."""
        loaded = _RuntimeEnvSettings()
        return cls(
            db_url=loaded.db_url,
            met_archive=loaded.met_archive,
            cache_dir=loaded.cache_dir,
            compute_root=loaded.compute_root,
        )


def resolve_runtime_settings(runtime: RuntimeSettings | None = None) -> RuntimeSettings:
    """Return explicit runtime settings or load them from the environment."""
    if runtime is not None:
        return runtime
    return RuntimeSettings.from_env()
