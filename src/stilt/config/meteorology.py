from pathlib import Path

from pydantic import BaseModel, model_validator

from .fields import cfg_field
from .spatial import Bounds


class MetConfig(BaseModel):
    """Meteorology file discovery and optional subgridding."""

    directory: Path = cfg_field(
        ...,
        description="Directory containing ARL meteorology files for this met stream.",
    )
    file_format: str = cfg_field(
        ...,
        description="Datetime format string used to discover meteorology filenames.",
    )
    file_tres: str = cfg_field(
        ...,
        description="Nominal time spacing between meteorology files.",
    )
    n_min: int = cfg_field(
        1,
        description="Minimum number of meteorology files required for a run.",
    )
    subgrid_enable: Path | bool = cfg_field(
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

    @model_validator(mode="after")
    def _no_subgrid(self) -> "MetConfig":
        """Keep the public subgrid surface but fail explicitly until implemented."""
        if self.subgrid_enable is not False:
            raise NotImplementedError(
                "subgrid_enable is not yet implemented. PYSTILT will rely on "
                "future arl-met cropping and ARL-writing support here. Leave "
                "subgrid_enable unset (default False)."
            )
        if self.subgrid_bounds is not None:
            raise NotImplementedError(
                "subgrid_bounds is not yet implemented. PYSTILT will rely on "
                "future arl-met cropping and ARL-writing support here. Leave "
                "subgrid_bounds unset."
            )
        return self
