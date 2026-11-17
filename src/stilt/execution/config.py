"""Settings for where and how a project's receptors are run."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class FetchConfig(BaseModel):
    """
    The Slurm resources of the job that gets the met files before a job array.

    Each setting left unset is the array's (``partition``, ``account``,
    ``qos``, ``time``, ``mem`` of :class:`ExecutionConfig`, or the ``slurm``
    option of that name, which replaces it for the array too). Set them to send
    the fetch to a data-transfer partition, where a cluster has one, and
    give ``time`` room for the downloads: a fetch job that runs out of time
    keeps the files it got, and the array then downloads the rest.

    Examples
    --------
    >>> config = ExecutionConfig(
    ...     backend="slurm",
    ...     time="02:00:00",
    ...     fetch={"partition": "notchpeak-dtn", "account": "dtn", "time": "12:00:00"},
    ... )
    >>> config.fetch.partition
    'notchpeak-dtn'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition: str | None = Field(
        None, description="Slurm partition of the fetch job. Unset is the array's."
    )
    account: str | None = Field(
        None, description="Slurm account of the fetch job. Unset is the array's."
    )
    qos: str | None = Field(
        None,
        description="Slurm quality of service of the fetch job. Unset is the array's.",
    )
    time: str | int | None = Field(
        None,
        description=(
            "Time limit of the fetch job, as for ``time``. Unset is the "
            "array's time limit per task."
        ),
    )
    mem: str | None = Field(
        None,
        description="Memory of the fetch job, such as ``4G``. Unset is the array's.",
    )

    @field_validator("time")
    @classmethod
    def _slurm_time(cls, value: str | int | None) -> str | int | None:
        """Raise on a time limit sbatch would not take."""
        if value is not None:
            slurm_minutes(value)  # raises on a form sbatch would not take
        return value


class ExecutionConfig(BaseModel):
    """
    Where a project's receptors run, and with what resources.

    With ``backend: local`` the receptors run on this machine as one task.
    With ``backend: slurm`` they are split among ``n_workers`` tasks of one
    Slurm job array. Either way a task runs ``cpus`` receptors at once. The
    Slurm settings are ignored by a local run, so a config can hold them and
    be run either way. ``timeout`` and ``keep_workdir`` apply to both. None
    of these settings change a result.

    Examples
    --------
    >>> config = ExecutionConfig(backend="slurm", n_workers=100, time="02:00:00")
    >>> config.n_workers
    100
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
        description=(
            "Slurm time limit per task, as ``HH:MM:SS``, ``D-HH:MM:SS``, or "
            "minutes. Required with ``backend: slurm``: without it a task gets "
            "the partition's default, which can be days."
        ),
    )
    mem: str | None = Field(None, description="Memory per Slurm task, such as ``8G``.")
    partition: str | None = Field(None, description="Slurm partition.")
    account: str | None = Field(None, description="Slurm account.")
    qos: str | None = Field(None, description="Slurm quality of service.")
    timeout: int | None = Field(
        None,
        ge=1,
        description=(
            "Time limit for one transport model run, in seconds. A run that "
            "exceeds it is stopped and recorded as a failed simulation, and the "
            "worker moves on. Unset waits indefinitely, so a hung run can hold "
            "a worker until its job ends."
        ),
    )
    keep_workdir: bool = Field(
        False,
        description=(
            "Keep every run's workdir, as the transport model left it, under "
            "``scratch/`` in the output directory. A failed run's is always "
            "kept."
        ),
    )
    array_parallelism: int | None = Field(
        None,
        ge=1,
        description="Most Slurm tasks running at once. Unset leaves it to Slurm.",
    )
    setup: list[str] = Field(
        default_factory=list,
        description=(
            "Shell commands run in each Slurm task before the worker, such as "
            "``module load`` lines. They run after the script exports "
            "``OMP_NUM_THREADS``, one thread per worker, so a line here can "
            "change it. The worker itself runs with the Python that submitted it."
        ),
    )
    slurm: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any other ``sbatch`` options by name, such as ``exclude: node1`` "
            "or ``constraint: skl``. ``true`` writes the bare flag. The fetch job "
            "takes ``clusters``, ``partition``, ``account``, ``qos``, ``time`` "
            "and ``mem`` from these, and no other option."
        ),
    )
    fetch: FetchConfig | Literal[False] = Field(
        default_factory=FetchConfig,
        description=(
            "Get the met files a met with ``download`` lacks once, before the "
            "receptors run: on Slurm in a job of its own that the array waits "
            "for, with these resources (:class:`FetchConfig`); locally before "
            "the workers start, when ``cpus`` is above 1. ``false`` leaves "
            "each run to download what it needs, as when something else fills "
            "the met directory."
        ),
    )

    @field_validator("fetch", mode="before")
    @classmethod
    def _fetch_true(cls, value: Any) -> Any:
        """Read ``fetch: true`` (or an empty ``fetch:``) as a fetch job with the array's resources."""
        return {} if value is True or value is None else value

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

    @model_validator(mode="after")
    def _slurm_needs_time(self) -> Self:
        """Raise when a Slurm run has no time limit."""
        if self.backend == "slurm" and self.time is None and "time" not in self.slurm:
            raise ValueError(
                "execution.time is required with backend: slurm, such as "
                "time: '02:00:00'. Without it each task gets the partition's "
                "default time limit, which can be days."
            )
        return self


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


__all__ = ["ExecutionConfig", "FetchConfig", "slurm_minutes"]
