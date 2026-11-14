"""
FLEXPART's settings and its met.

Settings keep FLEXPART's own names, lowercase (``loutstep``, ``ctl``,
``lconvection``), as they appear in its ``COMMAND`` file, so a FLEXPART
user finds them where they expect.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stilt.transport import TransportConfig


class FlexpartConfig(TransportConfig):
    """
    FLEXPART 11's settings for a PYSTILT run.

    PYSTILT runs the executable of a FLEXPART checkout you built
    (``flexpart_dir``) and never ships FLEXPART, which is GPL-3. Each run
    writes FLEXPART's input files into its workdir, with particle output
    every ``loutstep`` seconds and no gridded output.
    """

    model_config = ConfigDict(extra="forbid")

    #: Where the build is changes no particle; its version is recorded instead.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"flexpart_dir"})

    flexpart_dir: Path | None = Field(
        None,
        description=(
            "Your FLEXPART 11 checkout, built in meter coordinates "
            "(``make eta=no`` in ``src/``): ``src/FLEXPART`` is run, and "
            "``options/`` gives the land-use and species files FLEXPART reads. "
            "Needed to run; a stored run's settings leave it out."
        ),
    )
    numpar: int = Field(
        1000,
        description=(
            "Number of particles released. A multipoint receptor splits them "
            "between its points."
        ),
    )
    loutstep: int = Field(
        3600,
        description="Seconds between particle outputs (``LOUTSTEP``).",
    )
    lsynctime: int = Field(
        900,
        description=(
            "Seconds between the times all particles are synchronized "
            "(``LSYNCTIME``). It must divide ``loutstep``."
        ),
    )
    ctl: float = Field(
        -5.0,
        description=(
            "Time step control (``CTL``): above 1, the boundary-layer time "
            "step is the Lagrangian time scale over ``ctl``; negative uses "
            "``lsynctime``."
        ),
    )
    ifine: int = Field(
        4,
        description=(
            "Reduction of the time step for vertical motion (``IFINE``), used "
            "only when ``ctl`` is above 1."
        ),
    )
    lconvection: Literal[0, 1] = Field(
        1, description="Convection scheme (``LCONVECTION``): 1 on, 0 off."
    )
    lturbulence: Literal[0, 1] = Field(
        1, description="Turbulence scheme (``LTURBULENCE``): 1 on, 0 off."
    )
    lsubgrid: Literal[0, 1] = Field(
        0,
        description=(
            "Raise the mixed layer for subgrid terrain (``LSUBGRID``): 1 on, 0 off."
        ),
    )
    cblflag: Literal[0, 1] = Field(
        0,
        description=(
            "Skewed rather than Gaussian turbulence in the convective boundary "
            "layer (``CBLFLAG``): 1 on, 0 off. Needs a large ``ctl`` and ``ifine``."
        ),
    )
    threads: int = Field(
        1,
        description=(
            "OpenMP threads. FLEXPART seeds each thread's random numbers with "
            "a fixed number, so the thread count changes the particles and is "
            "recorded with the run."
        ),
    )
    hnf_plume: bool = Field(
        False,
        description=(
            "Apply the near-field plume correction. FLEXPART writes none of "
            "``samt``, ``sigw``, and ``tlgr``, which it reads, so it is off."
        ),
    )

    @field_validator("seed")
    @classmethod
    def _no_seed(cls, seed: int | None) -> int | None:
        """Refuse a seed, which FLEXPART has no setting for."""
        if seed is not None:
            raise ValueError(
                "FLEXPART has no seed: it seeds each OpenMP thread with a fixed "
                "number, so a run is repeated by its thread count (threads)."
            )
        return seed

    @field_validator("numpar", "loutstep", "lsynctime", "threads")
    @classmethod
    def _positive(cls, value: int) -> int:
        """Refuse a count or a step that is not positive."""
        if value < 1:
            raise ValueError("must be 1 or more.")
        return value

    @model_validator(mode="after")
    def _steps(self) -> FlexpartConfig:
        """Check that ``lsynctime`` divides ``loutstep``, which FLEXPART requires."""
        if self.loutstep % self.lsynctime:
            raise ValueError(
                f"lsynctime ({self.lsynctime} s) must divide loutstep "
                f"({self.loutstep} s)."
            )
        return self

    @property
    def checkout(self) -> Path:
        """
        The FLEXPART checkout, ``flexpart_dir``.

        Raises
        ------
        ValueError
            If ``flexpart_dir`` is not set.
        """
        if self.flexpart_dir is None:
            raise ValueError(
                "Set flexpart_dir to your FLEXPART 11 checkout to run FLEXPART."
            )
        return Path(self.flexpart_dir)

    @property
    def executable(self) -> Path:
        """The FLEXPART executable in the checkout, ``src/FLEXPART``."""
        return self.checkout / "src" / "FLEXPART"


def flexpart_version(flexpart_dir: str | Path) -> str:
    """
    Return the version of the FLEXPART checkout in *flexpart_dir*.

    A ``version`` file in the checkout wins, as for a HYSPLIT build;
    otherwise the git commit, with ``-dirty`` when the source has
    uncommitted changes, which can change the particles.

    Raises
    ------
    FileNotFoundError
        If the checkout has neither a ``version`` file nor git history.
    """
    root = Path(flexpart_dir)
    if (root / "version").is_file():
        return (root / "version").read_text().strip()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise FileNotFoundError(
            f"{root} has no 'version' file and is not a git checkout. Put a "
            "'version' file in it holding the FLEXPART version (such as "
            "11.0), so its runs are told apart from other builds'."
        ) from None
    return f"git {commit}"


class FlexpartMetConfig(BaseModel):
    """
    The meteorology FLEXPART reads: a directory of GRIB files and what they are.

    FLEXPART reads each file as ``AVAILABLE`` lists it, by valid time; PYSTILT
    reads the valid time from each file's first message, so the names do
    not matter. Only GFS is supported so far, in GRIB2 on a regular
    latitude-longitude grid (``pgrb2``, analyses and short forecasts).

    A string in ``config.yaml`` is the ``directory``: ``gfs: /path/to/gfs``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["gfs"] = Field(
        "gfs",
        description=(
            "The weather model the files come from. ``gfs`` is NCEP's GFS on "
            "pressure levels; FLEXPART reads ECMWF's (ERA5) differently, and "
            "PYSTILT does not support it yet."
        ),
    )
    directory: Path = Field(
        ...,
        description=(
            "Directory of the GRIB files. A relative path starts from the "
            "project directory."
        ),
    )
    pattern: str = Field(
        "*",
        description="Glob that picks the GRIB files in ``directory``.",
    )

    @model_validator(mode="before")
    @classmethod
    def _folder(cls, data: Any) -> Any:
        """Take a bare string or path as the directory."""
        if isinstance(data, str | Path):
            return {"directory": data}
        return data

    def settings(self) -> dict[str, Any]:
        """
        Return what a run records of this met: its source and its grid.

        The grid, such as ``0.5x0.5 deg, 720x361``, is read from the first
        file, so GFS at 0.5 and at 0.25 degrees run in different settings
        folders. Where the files are kept is not recorded.

        Raises
        ------
        FileNotFoundError
            If ``directory`` holds no file that matches ``pattern``.
        """
        from stilt.transport.flexpart.met import grib_grid

        files = sorted(Path(self.directory).glob(self.pattern))
        if not files:
            raise FileNotFoundError(
                f"No files match {self.pattern!r} in {self.directory}."
            )
        return {"source": self.source, "grid": grib_grid(files[0])}

    @property
    def nxshift(self) -> int:
        """FLEXPART's ``NXSHIFT`` for this source: 0 for GFS, which starts at 0° E."""
        return 0


__all__ = ["FlexpartConfig", "FlexpartMetConfig", "flexpart_version"]
