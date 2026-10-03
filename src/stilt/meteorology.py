"""
Meteorology: what a met is (:class:`MetConfig`), and finding, downloading, and cropping its files (:class:`Met`).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stilt._atomic import atomic_path
from stilt.exceptions import MeteorologyError
from stilt.spatial import Bounds

if TYPE_CHECKING:
    from arlmet.archives import Archive

logger = logging.getLogger(__name__)


class MetConfig(BaseModel):
    """
    Config for one met, as written under ``mets:`` in ``config.yaml``.

    Give either ``download`` to download ARL files with arlmet, or
    ``file_format`` and ``file_tres`` to find them in ``directory``. With
    ``download``, other keys are options for that archive (such as
    ``domain`` for ``nams``). Any other unknown key is an error.

    A run records its met in its ``_settings.yaml`` without the fields in
    :attr:`UNRECORDED`: the directories, so moving the met files does not
    change which runs are complete, and ``download_from`` and ``n_min``,
    which change no particle. A project requires ``directory`` for every
    met, and ``subgrid_dir`` when it crops local files.
    """

    model_config = ConfigDict(extra="allow")

    #: Fields left out of a run's recorded settings, since they change no particle.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset(
        {"directory", "subgrid_dir", "download_from", "n_min"}
    )

    directory: Path | None = Field(
        None,
        description=(
            "Directory holding the ARL meteorology files. Downloads are saved "
            "here. A project requires it."
        ),
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
        ge=0,
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
            from arlmet.archives import ARCHIVES

            if self.download not in ARCHIVES:
                raise ValueError(
                    f"Unknown ARL archive {self.download!r} to download. "
                    f"Available: {sorted(ARCHIVES)}."
                )
            try:
                inspect.signature(ARCHIVES[self.download]).bind(**extra)
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


def _met_window(
    r_time: pd.Timestamp, n_hours: int, file_tres: str | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the first and last time a run needs meteorology for, in time order.

    A backward run released after the start of a met file also needs the
    hour after its release, because HYSPLIT interpolates the release time
    between two hours. For a release in the last hour of a file that hour is
    in the next file, which STILT-R also reads. *file_tres* is the period of
    one file; without it (downloads, whose archive files are 6 hours or
    longer) the next hour is always added, which never adds a file a release
    on a file boundary does not need.
    """
    sim_end = r_time + pd.Timedelta(hours=n_hours)
    assert isinstance(sim_end, pd.Timestamp)  # not NaT: r_time is a time
    earlier, later = min(r_time, sim_end), max(r_time, sim_end)
    if n_hours < 0 and (file_tres is None or later != later.floor(file_tres)):
        next_hour = later.floor("h") + pd.Timedelta(hours=1)
        assert isinstance(next_hour, pd.Timestamp)  # not NaT: later is a time
        later = next_hour
    return earlier, later


class Met:
    """
    The meteorology files of one met, found locally or downloaded.

    Without ``download``, files are found under ``directory`` from
    ``file_format`` and ``file_tres``. With ``download`` set to an ARL
    archive name, arlmet downloads the files, cropping them as it goes when
    subgridding is on. Local files are cropped with
    ``arlmet.extract_subset`` into :attr:`crop_dir`, which all simulations
    share.

    Parameters
    ----------
    name : str
        Name of the met in the config.
    config : MetConfig
        Its settings.
    """

    def __init__(self, name: str, config: MetConfig):
        if config.directory is None:
            raise ValueError(f"Met {name!r} has no directory.")
        self.name = name
        self.config = config
        #: ``config.directory``, made absolute.
        self.directory = config.directory.expanduser().resolve()
        self._archive: Archive | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_matched_files(paths: list[Path]) -> list[Path]:
        """Resolve symlinks and drop duplicate files, sorted by name."""
        return sorted(
            dict.fromkeys(path.resolve() for path in paths),
            key=lambda path: path.name,
        )

    def _get_archive(self) -> Archive:
        """Return the arlmet archive to download from, building it on first use."""
        if self._archive is None:
            from arlmet.archives import get_archive

            assert self.config.download is not None
            self._archive = get_archive(
                self.config.download, **self.config.download_options
            )
        return self._archive

    def _effective_bbox(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)`` of the subgrid bounds plus the buffer."""
        b = self.config.subgrid_bounds
        buf = self.config.subgrid_buffer
        if b is None:
            raise ValueError("subgrid_bounds is required to compute effective bbox.")
        return (b.xmin - buf, b.ymin - buf, b.xmax + buf, b.ymax + buf)

    @property
    def crop_dir(self) -> Path:
        """
        Directory holding this met's cropped files.

        It is a folder inside ``subgrid_dir`` named by a short hash of the
        crop box (``subgrid_bounds`` plus ``subgrid_buffer``) and
        ``subgrid_levels``. Changing any of them gives a new folder, so
        old crops are never reused for a different crop.
        """
        if self.config.subgrid_dir is None:
            raise ValueError("subgrid_dir is required to crop local files.")
        # Imported here: stilt.identity reads MetConfig from this module.
        from stilt.identity import settings_hash

        crop = {"bbox": self._effective_bbox(), "levels": self.config.subgrid_levels}
        key = settings_hash(crop)[:12]
        return self.config.subgrid_dir.expanduser().resolve() / key

    def _level_indices(self) -> list[int] | None:
        """Return the indices of the lowest ``subgrid_levels`` levels, or None to keep all."""
        if self.config.subgrid_levels is None:
            return None
        return list(range(self.config.subgrid_levels))

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _download(self, r_time: pd.Timestamp, n_hours: int) -> list[Path]:
        """Return the files for a run from the ARL archive, downloading any not yet in ``directory``."""
        t_start, t_end = _met_window(r_time, n_hours)

        bbox = self._effective_bbox() if self.config.subgrid_enable else None
        levels = self._level_indices() if self.config.subgrid_enable else None

        archive = self._get_archive()
        try:
            files = archive.fetch(
                t_start,
                t_end,
                dest_dir=self.directory,
                mirror=self.config.download_from,
                bbox=bbox,
                levels=levels,
            )
        except ImportError as exc:
            raise ImportError(
                f"{exc}\n\n"
                "Downloading meteorology requires the cloud extra. "
                "Install with: pip install pystilt[cloud]"
            ) from exc

        n_files = len(files)
        if n_files == 0 or n_files < self.config.n_min:
            raise MeteorologyError(
                f"Insufficient number of meteorological files found. "
                f"Found: {n_files}, Required: {self.config.n_min}."
            )
        return files

    def required_files(self, r_time, n_hours: int) -> list[Path]:
        """
        Return the met files that cover one simulation.

        Parameters
        ----------
        r_time : datetime-like
            Receptor time.
        n_hours : int
            Simulation length in hours, negative for backward runs.

        Returns
        -------
        list of Path

        Raises
        ------
        MeteorologyError
            Fewer than ``n_min`` files were found.
        """
        _r_time = cast(pd.Timestamp, pd.Timestamp(r_time))

        if self.config.download is not None:
            return self._download(_r_time, n_hours)

        # Local files
        file_format, file_tres = self.config.file_format, self.config.file_tres
        # MetConfig requires both when there is no download.
        assert file_format is not None and file_tres is not None
        tres = to_offset(pd.to_timedelta(file_tres)).freqstr
        earlier, later = _met_window(_r_time, n_hours, tres)
        met_times = pd.date_range(earlier.floor(tres), later, freq=tres)
        patterns = list(dict.fromkeys(t.strftime(file_format) for t in met_times))

        files: list[Path] = []
        missing: list[str] = []
        for pattern in patterns:
            # Backup copies (name~<timestamp>~, name.~1~, name~) all end in "~".
            matches = [
                p
                for p in self.directory.rglob(f"{pattern}*")
                if p.is_file() and ".lock" not in p.name and not p.name.endswith("~")
            ]
            if matches:
                files.extend(matches)
            else:
                missing.append(pattern)

        files = self._dedupe_matched_files(files)

        n_files = len(files)
        if n_files == 0 or n_files < self.config.n_min:
            detail = ""
            if missing:
                examples = ", ".join(missing[:3])
                detail = f" Patterns not found in {self.directory}: {examples}."
            raise MeteorologyError(
                f"Insufficient number of meteorological files found. "
                f"Found: {n_files}, Required: {self.config.n_min}.{detail}"
            )

        if missing:
            examples = ", ".join(missing[:3])
            logger.warning(
                "Met patterns not found (simulation may lack temporal coverage): %s",
                examples,
            )

        return files

    def readable(self, files: list[Path]) -> list[Path]:
        """
        Return the files a transport model should read in place of *files*.

        When local files are cropped, these are the cropped copies in
        :attr:`crop_dir`, made on first use. Downloaded files were cropped
        as they were downloaded. When two files share a name, the first is
        kept, since a model would otherwise read the same hours twice.
        """
        if self.config.subgrid_enable and self.config.download is None:
            files = self._crop_local_files(files)
        kept: dict[str, Path] = {}
        for path in files:
            first = kept.setdefault(path.name, path)
            if first is not path and first.resolve() != path.resolve():
                logger.warning(
                    "met %s has duplicate basename %s at %s and %s; using %s",
                    self.name,
                    path.name,
                    first,
                    path,
                    first,
                )
        return list(kept.values())

    def _crop_local_files(self, files: list[Path]) -> list[Path]:
        """
        Crop local files into :attr:`crop_dir`, reusing crops that already exist.

        Each crop is written to a temporary name and then renamed, so a
        worker never reads a half-written file. Two workers cropping the
        same file at once both finish, and the second rename wins.
        """
        from arlmet import extract_subset

        crop_dir = self.crop_dir
        crop_dir.mkdir(parents=True, exist_ok=True)
        bbox = self._effective_bbox()
        levels = self._level_indices()

        subsetted: list[Path] = []
        for src in files:
            cache_path = crop_dir / src.name
            if not cache_path.exists():
                logger.info("Subsetting %s → %s", src.name, cache_path)
                with atomic_path(cache_path) as tmp:
                    extract_subset(src, tmp, bbox=bbox, levels=levels)
            subsetted.append(cache_path)
        return subsetted


__all__ = ["Met", "MetConfig"]
