"""Settings for one met, a named set of meteorology files."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spatial import Bounds

if TYPE_CHECKING:
    from arlmet.sources import MeteorologySource


def arlmet_sources() -> dict[str, type[MeteorologySource]]:
    """Return the ARL archives the installed arlmet can download, by name."""
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


class MetSettings(BaseModel):
    """
    The settings of a met that are recorded with each run.

    These are the fields of :class:`MetConfig` without its two directories.
    They decide the particles a run produces, so they are saved in the run's
    ``_settings.yaml`` and are part of its hash
    (:class:`~stilt.config.TransportSettings`). The directories are left out,
    so moving the met files does not change which runs are complete.
    """

    model_config = ConfigDict(extra="allow")

    download: str | None = Field(
        None,
        description=(
            "Name of a NOAA ARL archive to download files from with arlmet, "
            "such as ``hrrr``, ``nam12``, ``gdas1``, or ``gfs0p25``. When set, "
            "``file_format`` and ``file_tres`` are not needed. Options for the "
            "archive (such as ``domain: ak``) go in the same entry."
        ),
    )
    download_from: Literal["s3", "ftp", "http"] = Field(
        "s3",
        description="Server to download from: ``s3``, ``ftp``, or ``http``.",
    )
    file_format: str | None = Field(
        None,
        description=(
            "``strftime`` pattern for the start of each file name, such as "
            "``%Y%m%d_%H``. Files whose names start with it are found anywhere "
            "under ``directory``. Required when ``download`` is not set."
        ),
    )
    file_tres: str | None = Field(
        None,
        description=(
            "Time covered by each file, as a pandas time string such as "
            "``6h``. Required when ``download`` is not set."
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
        """Check the archive and its options, and the fields each mode needs."""
        extra = self.download_options
        if self.download is not None:
            archives = arlmet_sources()
            if self.download not in archives:
                raise ValueError(
                    f"Unknown ARL archive {self.download!r} to download. "
                    f"Available: {sorted(archives)}."
                )
            try:
                inspect.signature(archives[self.download]).bind(**extra)
            except TypeError as exc:
                raise ValueError(
                    f"ARL archive {self.download!r} does not take the options "
                    f"{sorted(extra)} ({exc})."
                ) from None
        elif extra:
            raise ValueError(
                f"Unknown met settings {sorted(extra)}. Only a met with "
                "download takes extra options."
            )
        if self.download is None and (
            self.file_format is None or self.file_tres is None
        ):
            raise ValueError(
                "file_format and file_tres are required to find local files. "
                "Set download to an ARL archive name, such as hrrr, to "
                "download the files instead."
            )
        if self.subgrid_enable and self.subgrid_bounds is None:
            raise ValueError("subgrid_bounds is required when subgrid_enable=True.")
        return self

    @property
    def download_options(self) -> dict[str, Any]:
        """Extra fields, passed as keyword arguments to the arlmet archive."""
        return dict(self.model_extra) if self.model_extra else {}

    def settings(self) -> MetSettings:
        """Return the recorded settings alone, without the directories a :class:`MetConfig` adds."""
        return MetSettings.model_validate(
            self.model_dump(exclude=set(UNRECORDED_MET_FIELDS))
        )


class MetConfig(MetSettings):
    """
    Settings for one met, as written under ``mets:`` in ``config.yaml``.

    Give either ``download`` to download ARL files with arlmet, or
    ``file_format`` and ``file_tres`` to find them in ``directory``. With
    ``download``, other keys are options for that archive (such as
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
            "uses this meteorology. Each crop box gets its own folder inside "
            "it. Required when cropping your own files; not used with "
            "``download``, which crops files as it downloads them."
        ),
    )

    @model_validator(mode="after")
    def _require_subgrid_dir(self) -> Self:
        """Require ``subgrid_dir`` when local files are cropped."""
        if self.subgrid_enable and self.download is None and self.subgrid_dir is None:
            raise ValueError(
                "subgrid_dir is required when subgrid_enable=True without "
                "download. Set it to a directory for the cropped files, outside "
                "the met archive."
            )
        return self
