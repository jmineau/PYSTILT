from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .fields import cfg_field
from .spatial import Bounds


def _arlmet_source_names() -> frozenset[str]:
    """Return the set of valid arlmet source names from the installed package."""
    import arlmet.sources as _src
    from arlmet.sources import MeteorologySource

    return frozenset(
        getattr(_src, name).name
        for name in _src.__all__
        if isinstance(getattr(_src, name, None), type)
        and issubclass(getattr(_src, name), MeteorologySource)
        and getattr(_src, name) is not MeteorologySource
    )


class MetConfig(BaseModel):
    """Meteorology file discovery, optional downloading, and optional subgridding."""

    model_config = ConfigDict(extra="allow")

    directory: Path = cfg_field(
        ...,
        description="Directory containing ARL meteorology files for this met stream.",
    )
    source: str | None = cfg_field(
        None,
        description=(
            "arlmet source name for automatic downloading from NOAA archives "
            "(e.g. 'hrrr', 'nam12', 'gdas1', 'gfs0p25'). "
            "When set, file_format and file_tres are not required. "
            "Source-specific constructor arguments (e.g. domain='ak' for nams) "
            "may be specified as additional inline fields."
        ),
    )
    backend: Literal["s3", "ftp", "http"] = cfg_field(
        "s3",
        description="Download backend when source is set. One of 's3', 'ftp', or 'http'.",
        visibility="advanced",
    )
    file_format: str | None = cfg_field(
        None,
        description=(
            "Datetime format string used to discover meteorology filenames. "
            "Required when source is not set (archive mode)."
        ),
    )
    file_tres: str | None = cfg_field(
        None,
        description=(
            "Nominal time spacing between meteorology files. "
            "Required when source is not set (archive mode)."
        ),
    )
    n_min: int = cfg_field(
        1,
        description="Minimum number of meteorology files required for a run.",
    )
    subgrid_enable: bool = cfg_field(
        False,
        description="Enable meteorology subgridding before the run.",
        visibility="advanced",
    )
    subgrid_bounds: Bounds | None = cfg_field(
        None,
        description="Bounds used for meteorology subgridding.",
        visibility="advanced",
    )
    subgrid_buffer: float = cfg_field(
        0.2,
        description="Buffer added around the receptor domain when subgridding meteorology.",
        visibility="advanced",
    )
    subgrid_levels: int | None = cfg_field(
        None,
        description="Number of vertical levels to keep when subgridding meteorology.",
        visibility="advanced",
    )
    subgrid_dir: Path | None = cfg_field(
        None,
        description=(
            "Directory to cache subgridded met files. "
            "Defaults to <directory>/subgrid when not set. "
            "Shared across all simulations that use this met stream."
        ),
        visibility="advanced",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "MetConfig":
        """Enforce mode-specific requirements."""
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
        """Extra fields passed as keyword arguments to the arlmet source constructor."""
        return dict(self.model_extra) if self.model_extra else {}
