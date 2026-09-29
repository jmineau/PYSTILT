"""Settings for one meteorology stream."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spatial import Bounds


def _arlmet_source_names() -> frozenset[str]:
    """Return the source names the installed arlmet provides."""
    import arlmet.sources as _src
    from arlmet.sources import MeteorologySource

    return frozenset(
        getattr(_src, name).name
        for name in _src.__all__
        if isinstance(getattr(_src, name, None), type)
        and issubclass(getattr(_src, name), MeteorologySource)
        and getattr(_src, name) is not MeteorologySource
    )


#: Met fields that change no output: where the files are, not what they hold.
UNRECORDED_MET_FIELDS = frozenset({"directory", "subgrid_dir"})


class MetConfig(BaseModel):
    """
    Settings for one meteorology stream.

    Give either ``source`` to download ARL files with arlmet, or
    ``file_format`` and ``file_tres`` to find them in ``directory``. Extra
    fields are passed to the arlmet source.
    """

    model_config = ConfigDict(extra="allow")

    directory: Path = Field(
        ...,
        description="Directory holding the ARL meteorology files. Downloads are saved here.",
    )
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
    subgrid_dir: Path | None = Field(
        None,
        description=(
            "Directory for the cropped files, shared by every simulation that "
            "uses this meteorology. Unset uses ``<directory>/subgrid``. Used "
            "only without ``source``, which crops files as it downloads them."
        ),
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "MetConfig":
        """Check the source name and the fields each mode needs."""
        if self.source is not None:
            available = _arlmet_source_names()
            if self.source not in available:
                raise ValueError(
                    f"Unknown arlmet source {self.source!r}. "
                    f"Available: {sorted(available)}."
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

    def record(self) -> dict[str, Any]:
        """Return this met as stored in the project's record."""
        return self.model_dump(mode="json")

    def differences(self, recorded: dict[str, Any]) -> list[str]:
        """Return the fields that affect results and differ from ``recorded``."""
        mine = self.record()
        return sorted(
            k
            for k in mine
            if k not in UNRECORDED_MET_FIELDS and mine[k] != recorded.get(k)
        )
