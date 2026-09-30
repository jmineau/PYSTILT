"""Settings for one meteorology stream."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spatial import Bounds

if TYPE_CHECKING:
    from arlmet.sources import MeteorologySource


def arlmet_sources() -> dict[str, type[MeteorologySource]]:
    """Return the download sources of the installed arlmet, by name."""
    import arlmet.sources as src

    return {
        cls.name: cls
        for attr in src.__all__
        if isinstance(cls := getattr(src, attr, None), type)
        and issubclass(cls, src.MeteorologySource)
        and cls is not src.MeteorologySource
    }


#: Met fields that change no output: where the files are, not what they hold.
UNRECORDED_MET_FIELDS = frozenset({"directory", "subgrid_dir"})


class MetContent(BaseModel):
    """
    What a meteorology stream holds, apart from where its files are.

    The source or file layout and the subgrid settings decide the particles
    a run produces, so they are part of a run's identity
    (:class:`~stilt.config.TransportSettings`). :class:`MetConfig` adds the
    directories, which do not.
    """

    model_config = ConfigDict(extra="allow")

    source: str | None = Field(
        None,
        description=(
            "Name of an arlmet source to download files from NOAA archives, "
            "such as ``hrrr``, ``nam12``, ``gdas1``, or ``gfs0p25``. When set, "
            "``file_format`` and ``file_tres`` are not needed. Options for the "
            "source (such as ``domain: ak``) go in the same entry."
        ),
    )
    backend: Literal["s3", "ftp", "http"] = Field(
        "s3",
        description="Where ``source`` downloads from: ``s3``, ``ftp``, or ``http``.",
    )
    file_format: str | None = Field(
        None,
        description=(
            "``strftime`` pattern for the start of each file name, such as "
            "``%Y%m%d_%H``. Files whose names start with it are found anywhere "
            "under ``directory``. Required when ``source`` is not set."
        ),
    )
    file_tres: str | None = Field(
        None,
        description=(
            "Time covered by each file, as a pandas time string such as "
            "``6h``. Required when ``source`` is not set."
        ),
    )
    n_min: int = Field(
        1,
        description="Minimum number of files a simulation needs. Fewer fails the simulation.",
    )
    subgrid_enable: bool = Field(
        False,
        description="Crop the meteorology to ``subgrid_bounds`` before running.",
    )
    subgrid_bounds: Bounds | None = Field(
        None,
        description="Longitude/latitude box to crop the meteorology to.",
    )
    subgrid_buffer: float = Field(
        0.2,
        description="Margin added to every side of ``subgrid_bounds``, in degrees.",
    )
    subgrid_levels: int | None = Field(
        None,
        description="Number of vertical levels to keep, counted from the surface. Unset keeps all.",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        """Check the source and its options, and the fields each mode needs."""
        extra = self.source_kwargs
        if self.source is not None:
            sources = arlmet_sources()
            if self.source not in sources:
                raise ValueError(
                    f"Unknown arlmet source {self.source!r}. "
                    f"Available: {sorted(sources)}."
                )
            try:
                inspect.signature(sources[self.source]).bind(**extra)
            except TypeError as exc:
                raise ValueError(
                    f"Met source {self.source!r} does not take the options "
                    f"{sorted(extra)} ({exc})."
                ) from None
        elif extra:
            raise ValueError(
                f"Unknown met settings {sorted(extra)}. Only a met with a "
                "source takes extra options."
            )
        if self.source is None and (self.file_format is None or self.file_tres is None):
            raise ValueError(
                "file_format and file_tres are required when source is not set "
                "(archive mode). Set source to an arlmet source name to use "
                "automatic downloading instead."
            )
        if self.subgrid_enable and self.subgrid_bounds is None:
            raise ValueError("subgrid_bounds is required when subgrid_enable=True.")
        return self

    @property
    def source_kwargs(self) -> dict[str, Any]:
        """Extra fields, passed as keyword arguments to the arlmet source."""
        return dict(self.model_extra) if self.model_extra else {}

    def content(self) -> MetContent:
        """Return the content alone, without the directories a :class:`MetConfig` adds."""
        return MetContent.model_validate(
            self.model_dump(exclude=set(UNRECORDED_MET_FIELDS))
        )


class MetConfig(MetContent):
    """
    Settings for one meteorology stream.

    Give either ``source`` to download ARL files with arlmet, or
    ``file_format`` and ``file_tres`` to find them in ``directory``. With
    ``source``, other keys are options for that arlmet source (such as
    ``domain`` for ``nams``). Any other unknown key is an error.
    """

    directory: Path = Field(
        ...,
        description="Directory holding the ARL meteorology files. Downloads are saved here.",
    )
    subgrid_dir: Path | None = Field(
        None,
        description=(
            "Directory for the cropped files, shared by every simulation that "
            "uses this meteorology. Unset uses ``<directory>/subgrid``. Used "
            "only without ``source``, which crops files as it downloads them."
        ),
    )
