"""Settings for where a deployment runs, read from the environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """
    Where work runs on this machine.

    Each field is read from the matching ``PYSTILT_*`` environment variable
    (``PYSTILT_COMPUTE_ROOT``) unless passed directly. None of them change a
    simulation's result.
    """

    model_config = SettingsConfigDict(env_prefix="PYSTILT_", extra="ignore")

    compute_root: Path | None = Field(
        default=None,
        description=(
            "Scratch directory under which workers run HYSPLIT. Unset uses "
            "``$TMPDIR/pystilt/<project name>``."
        ),
    )


__all__ = ["RuntimeSettings"]
