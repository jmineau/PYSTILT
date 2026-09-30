"""Settings for where and how a project's receptors are run."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ExecutionConfig(BaseModel):
    """
    Where a project's receptors run, and with what resources.

    With ``backend: local`` the receptors run on this machine. With
    ``backend: slurm`` they are split among ``n_workers`` tasks of one Slurm
    job array. The Slurm settings are ignored by a local run, so a config can
    hold them and be run either way. None of these settings change a result.

    Examples
    --------
    >>> ExecutionConfig(backend="slurm", n_workers=100, time="02:00:00", mem="8G")
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    backend: Literal["local", "slurm"] = Field(
        "local",
        description="Where to run: on this machine (``local``) or as a Slurm job array (``slurm``).",
    )
    n_workers: int = Field(
        1,
        ge=1,
        description=(
            "Local: number of worker processes. Slurm: number of array tasks, "
            "which the receptors are split evenly among."
        ),
    )
    cpus: int = Field(
        1,
        ge=1,
        validation_alias=AliasChoices("cpus", "cpus_per_task", "cpus-per-task"),
        description=(
            "CPUs per Slurm task. With more than one, a task runs that many "
            "receptors at once."
        ),
    )
    time: str | int | None = Field(
        None,
        description="Slurm time limit per task, as ``HH:MM:SS``, ``D-HH:MM:SS``, or minutes.",
    )
    mem: str | None = Field(None, description="Memory per Slurm task, such as ``8G``.")
    partition: str | None = Field(None, description="Slurm partition.")
    account: str | None = Field(None, description="Slurm account.")
    qos: str | None = Field(None, description="Slurm quality of service.")
    array_parallelism: int | None = Field(
        None,
        ge=1,
        description="Most Slurm tasks running at once. 256 when unset.",
    )
    setup: list[str] = Field(
        default_factory=list,
        description=(
            "Shell commands run in each Slurm task before the worker, such as "
            "``module load`` lines. The worker itself runs with the Python "
            "that submitted it."
        ),
    )
    slurm: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any other ``sbatch`` options by name, such as ``exclude: node1`` "
            "or ``constraint: skl``. ``true`` writes the bare flag."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _known_settings(cls, data: Any) -> Any:
        """Reject a setting this model does not have, saying where it goes."""
        if not isinstance(data, dict):
            return data
        known = set(cls.model_fields) | {"cpus_per_task", "cpus-per-task"}
        unknown = sorted(str(k) for k in data if k not in known)
        if unknown:
            raise ValueError(
                f"Unknown execution setting(s) {unknown}. The settings are "
                f"{sorted(cls.model_fields)}. Put any other sbatch option "
                "under 'slurm:', for example 'slurm: {exclude: node1}'."
            )
        return data

    @field_validator("setup", mode="before")
    @classmethod
    def _one_command(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


__all__ = ["ExecutionConfig"]
