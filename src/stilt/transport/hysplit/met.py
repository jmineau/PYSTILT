"""
HYSPLIT's meteorology: the met config (:class:`MetConfig`), and finding, downloading, and cropping ARL files (:class:`Met`).
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
import os
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from stilt._paths import absolute, atomic_path
from stilt.exceptions import MeteorologyError
from stilt.identity import settings_hash
from stilt.spatial import Bounds

if TYPE_CHECKING:
    from arlmet.archives import Archive

logger = logging.getLogger(__name__)


class MetConfig(BaseModel):
    """
    HYSPLIT's config for one met, as written under ``mets:`` in ``config.yaml``.

    HYSPLIT reads ARL files. Give either ``download`` to download them with
    arlmet, or ``file_format`` and ``file_tres`` to find them in
    ``directory``. With ``download``, other keys are options for that
    archive (such as ``domain`` for ``nams``). Any other unknown key is an
    error.

    A run records which weather product its met is and how it was cropped
    (:meth:`settings`), not where the files are kept or how they are named.
    Moving the files, or renaming them, changes no result.
    """

    model_config = ConfigDict(extra="allow")

    directory: Path = Field(
        ...,
        description=(
            "Directory holding the ARL meteorology files. Downloads are saved "
            "here. In a project, a relative path starts from the project "
            "directory; ``~`` and ``$VARIABLES`` are expanded."
        ),
    )
    subgrid_dir: Path | None = Field(
        None,
        description=(
            "Directory for the cropped files, shared by every simulation that "
            "uses this meteorology. Each crop box gets its own folder inside "
            "it. Required when cropping your own files; not used with "
            "``download``, which crops files as it downloads them. A relative "
            "path starts from the project directory."
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
    subgrid_enable: bool = Field(
        False,
        description="Crop the meteorology to ``subgrid_bounds`` before running.",
    )
    subgrid_bounds: Bounds | None = Field(
        None,
        description=(
            "Longitude/latitude box to crop the meteorology to. Particles that "
            "leave it stop, so make it wide enough for the whole run, not only "
            "the footprint grid."
        ),
    )
    subgrid_levels: int | None = Field(
        None,
        description="Number of vertical levels to keep, counted from the surface. Unset keeps all.",
    )

    # The source read from the first file, kept after the first read.
    _source: str | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        """Check the archive and its options, and the fields each mode needs."""
        extra = self.download_options
        for key in extra.keys() & _REMOVED.keys():
            raise ValueError(f"{key} is gone: {_REMOVED[key]}")
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
        if self.subgrid_enable and self.download is None and self.subgrid_dir is None:
            raise ValueError(
                "subgrid_dir is required when subgrid_enable=True without "
                "download. Set it to a directory for the cropped files, outside "
                "the met archive."
            )
        return self

    def settings(self) -> dict[str, Any]:
        """
        Return what a run records of this met: its weather product and its crop.

        The product is the source id the files' headers carry, such as
        ``HRRR`` or ``NAM``: the archive's for ``download``, otherwise read
        from the first ARL file under ``directory``. The crop is
        :meth:`crop`. Nothing about where the files are kept or how they
        are named is recorded.

        Raises
        ------
        FileNotFoundError
            If ``directory`` holds no ARL file to read the product from.
        """
        return {"source": self.source(), "crop": self.crop()}

    def source(self) -> str:
        """
        Return the source id of this met's files, such as ``HRRR``, as their headers carry it.

        A local met's is read from its first file once, and kept.
        """
        if self.download is not None:
            from arlmet.archives import ARCHIVES

            return ARCHIVES[self.download].source
        if self._source is None:
            self._source = _first_source(absolute(self.directory))
        return self._source

    def crop(self) -> dict[str, Any] | None:
        """
        Return the crop as ``{"bbox": [west, south, east, north], "levels": n}``, or ``None`` without one.

        The box is ``subgrid_bounds``; ``levels`` is ``subgrid_levels``,
        ``None`` to keep all.
        """
        b = self.subgrid_bounds
        if not self.subgrid_enable or b is None:
            return None
        return {"bbox": [b.xmin, b.ymin, b.xmax, b.ymax], "levels": self.subgrid_levels}

    @property
    def download_options(self) -> dict[str, Any]:
        """Extra fields, passed as keyword arguments to the arlmet archive."""
        return dict(self.model_extra) if self.model_extra else {}


def _time(value: Any) -> pd.Timestamp:
    """Return *value* as a Timestamp, raising for a missing time."""
    time = pd.Timestamp(value)
    if not isinstance(time, pd.Timestamp):  # NaT
        raise ValueError(f"Not a time: {value!r}")
    return time


def _cover(
    window: tuple[Any, Any], hour_after: bool, file_tres: str | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the first and last time files must cover for *window*.

    With *hour_after*, the hour after the window's end is added, unless the
    end is on a file boundary. HYSPLIT needs it for a backward run, because
    it interpolates the release time between two hours: for a release in
    the last hour of a file that hour is in the next file, which STILT-R
    also reads. *file_tres* is the period of one file; without it
    (downloads, whose archive files are 6 hours or longer) the next hour is
    always added, which never adds a file a release on a file boundary does
    not need.
    """
    earlier, later = (_time(t) for t in window)
    if hour_after and (file_tres is None or later != later.floor(file_tres)):
        later = _time(later.floor("h") + pd.Timedelta(hours=1))
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
        self.name = name
        self.config = config
        #: ``config.directory``, made absolute: a relative path starts from
        #: the working directory, and ``~`` and ``$VARIABLES`` are expanded.
        self.directory = absolute(config.directory)
        self._archive: Archive | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @cached_property
    def _listing(self) -> list[tuple[tuple[str, ...], Path]]:
        """
        Every file under :attr:`directory`, walked once, as its path's parts relative to it and the path.

        Lock files and backup copies (``name~<timestamp>~``, ``name.~1~``,
        ``name~``, which all end in ``~``) are left out.
        """
        found: list[tuple[tuple[str, ...], Path]] = []
        for root, _, names in os.walk(self.directory):
            parts = Path(root).relative_to(self.directory).parts
            for name in names:
                if ".lock" not in name and not name.endswith("~"):
                    found.append(((*parts, name), Path(root) / name))
        return found

    def _matching(self, pattern: str) -> list[Path]:
        """Return the files whose path ends in one that starts with *pattern*, as ``rglob(pattern + "*")`` finds them."""
        depth = pattern.count("/") + 1
        found = []
        for parts, path in self._listing:
            if len(parts) < depth:
                continue
            tail = "/".join(parts[-depth:])
            if fnmatch.fnmatchcase(tail, pattern + "*") and path.is_file():
                found.append(path)
        return found

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

    def _bbox(self) -> tuple[float, float, float, float]:
        """Return the crop box, ``(west, south, east, north)``."""
        crop = self.config.crop()
        if crop is None:
            raise ValueError("subgrid_bounds is required to crop.")
        west, south, east, north = crop["bbox"]
        return (west, south, east, north)

    @property
    def crop_dir(self) -> Path:
        """
        Directory holding this met's cropped files.

        It is a folder inside ``subgrid_dir`` named by a short hash of the
        crop box (``subgrid_bounds``) and ``subgrid_levels``. Changing
        either gives a new folder, so old crops are never reused for a
        different crop.
        """
        if self.config.subgrid_dir is None:
            raise ValueError("subgrid_dir is required to crop local files.")
        key = settings_hash(self.config.crop() or {})[:12]
        return absolute(self.config.subgrid_dir) / key

    def _level_indices(self) -> list[int] | None:
        """Return the indices of the lowest ``subgrid_levels`` levels, or None to keep all."""
        if self.config.subgrid_levels is None:
            return None
        return list(range(self.config.subgrid_levels))

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _download(self, window: tuple[Any, Any], hour_after: bool) -> list[Path]:
        """Return the files that cover *window* from the ARL archive, downloading any not yet in ``directory``."""
        t_start, t_end = _cover(window, hour_after)

        bbox = self._bbox() if self.config.subgrid_enable else None
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
                "Downloading meteorology requires the download extra. "
                "Install with: pip install pystilt[download]"
            ) from exc

        if not files:
            raise MeteorologyError(
                f"No {self.config.download} files from {t_start} to {t_end} "
                f"were found or downloaded."
            )
        return files

    def files_for(
        self, window: tuple[Any, Any], *, hour_after: bool = False
    ) -> list[Path]:
        """
        Return the met files that cover *window*, ``(start, end)`` in time order.

        Downloads them first when ``download`` is set.

        Parameters
        ----------
        window : tuple of datetime-like
            The time to cover (:func:`run_window`).
        hour_after : bool, default False
            Also cover the hour after the window's end, unless the end is on
            a file boundary. HYSPLIT needs it for a backward run.

        Returns
        -------
        list of Path

        Raises
        ------
        MeteorologyError
            A file the run needs is missing. Every one is needed: HYSPLIT
            would stop the particles where the met runs out.
        """
        if self.config.download is not None:
            return self._download(window, hour_after)

        # Local files
        file_format, file_tres = self.config.file_format, self.config.file_tres
        # MetConfig requires both when there is no download.
        assert file_format is not None and file_tres is not None
        # pandas-stubs' to_offset takes no Timedelta (pandas does)
        # pyrefly: ignore[no-matching-overload]
        tres = to_offset(pd.to_timedelta(file_tres)).freqstr
        earlier, later = _cover(window, hour_after, tres)
        met_times = pd.date_range(earlier.floor(tres), later, freq=tres)
        patterns = list(dict.fromkeys(t.strftime(file_format) for t in met_times))

        files: list[Path] = []
        missing: list[str] = []
        for pattern in patterns:
            matches = self._matching(pattern)
            if matches:
                files.extend(matches)
            else:
                missing.append(pattern)

        if missing:
            hours = [
                f"{t:%Y-%m-%d %H:%M}"
                for t in met_times
                if t.strftime(file_format) in missing
            ]
            raise MeteorologyError(
                f"No met file in {self.directory} for {', '.join(hours)} "
                f"(each {file_tres}, named {', '.join(missing)}...). The run "
                f"covers {earlier:%Y-%m-%d %H:%M} to {later:%Y-%m-%d %H:%M}."
            )
        return self._dedupe_matched_files(files)

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
        bbox = self._bbox()
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


#: Met settings that were removed, and what to do instead.
_REMOVED = {
    "subgrid_buffer": (
        "the crop is subgrid_bounds. Widen the bounds by the buffer instead (it "
        "was in degrees; STILT-R's met_subgrid_buffer is a fraction of the "
        "footprint grid's size)."
    ),
    "n_min": (
        "a run needs every met file its hours fall in, and fails naming the "
        "hours with no file."
    ),
}


def _arl_candidates(directory: Path) -> Iterator[Path]:
    """Yield the files under *directory*, in name order, files before folders at each level."""
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except FileNotFoundError:
        return
    folders = []
    for entry in entries:
        if entry.name.endswith("~") or ".lock" in entry.name:
            continue  # backups and lock files, as Met.files_for skips them
        if entry.is_dir():
            folders.append(entry.path)
        elif entry.is_file():
            yield Path(entry.path)
    for folder in folders:
        yield from _arl_candidates(Path(folder))


def _first_source(directory: Path) -> str:
    """
    Return the source id in the header of the first ARL file under *directory*.

    Files that are not ARL files, such as a README, are skipped.

    Raises
    ------
    FileNotFoundError
        If no file under *directory* is an ARL file.
    """
    from arlmet import IndexRecord

    for path in _arl_candidates(directory):
        try:
            with path.open("rb") as f:
                return IndexRecord.from_position(f, 0).source
        except (EOFError, OSError, ValueError):  # ARLFormatError is a ValueError
            continue
    raise FileNotFoundError(
        f"No ARL file under {directory}. A met's weather product, which a "
        "run records, is read from the header of the first file there."
    )


__all__ = ["Met", "MetConfig"]
