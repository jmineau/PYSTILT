"""Settings for where and how a project's receptors are run."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class ExecutionConfig(BaseModel):
    """
    Where a project's receptors run, and with what resources.

    With ``backend: local`` the receptors run on this machine as one task.
    With ``backend: slurm`` they are split among ``n_workers`` tasks of one
    Slurm job array. Either way a task runs ``cpus`` receptors at once. The
    Slurm settings are ignored by a local run, so a config can hold them and
    be run either way. None of these settings change a result.

    Examples
    --------
    >>> ExecutionConfig(backend="slurm", n_workers=100, time="02:00:00", mem="8G")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["local", "slurm"] = Field(
        "local",
        description="Where to run: on this machine (``local``) or as a Slurm job array (``slurm``).",
    )
    n_workers: int = Field(
        1,
        ge=1,
        description=(
            "Number of Slurm array tasks, which the receptors are split evenly "
            "among. A local run is one task."
        ),
    )
    cpus: int = Field(
        1,
        ge=1,
        description=(
            "Receptors each task runs at once: processes on this machine for a "
            "local run, CPUs per array task on Slurm."
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

    @field_validator("setup", mode="before")
    @classmethod
    def _one_command(cls, value: Any) -> Any:
        """Accept one setup command as a string."""
        return [value] if isinstance(value, str) else value

    @field_validator("time")
    @classmethod
    def _slurm_time(cls, value: str | int | None) -> str | int | None:
        """Raise on a time limit sbatch would not take."""
        if value is not None:
            slurm_minutes(value)  # raises on a form sbatch would not take
        return value

    @property
    def time_minutes(self) -> int | None:
        """The time limit in whole minutes, rounded up, or ``None`` when unset."""
        return None if self.time is None else slurm_minutes(self.time)


def slurm_minutes(time: str | int) -> int:
    """
    Return a Slurm time limit in whole minutes, rounded up.

    Accepts what ``sbatch --time`` does: minutes, ``MM:SS``, ``HH:MM:SS``,
    ``D-HH``, ``D-HH:MM``, and ``D-HH:MM:SS``.

    Raises
    ------
    ValueError
        If *time* has none of these forms or is not positive.
    """
    text = str(time).strip()
    days, _, clock = text.rpartition("-")
    parts = clock.split(":")
    try:
        if "-" in text and not days:
            raise ValueError
        numbers = [int(p) for p in parts]
        d = int(days) if days else 0
        if any(n < 0 for n in numbers) or d < 0:
            raise ValueError
        if days:  # D-HH[:MM[:SS]]
            hours, minutes, seconds = (numbers + [0, 0])[:3]
        elif len(numbers) == 1:  # minutes
            hours, minutes, seconds = 0, numbers[0], 0
        elif len(numbers) == 2:  # MM:SS
            hours, (minutes, seconds) = 0, numbers
        else:  # HH:MM:SS
            hours, minutes, seconds = numbers
        if len(numbers) > 3:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"Not a Slurm time limit: {time!r}. Use minutes, 'HH:MM:SS', or "
            "'D-HH:MM:SS'."
        ) from None
    total = ((d * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError(f"The time limit must be positive, not {time!r}.")
    return -(-total // 60)


__all__ = ["ExecutionConfig", "slurm_minutes"]
