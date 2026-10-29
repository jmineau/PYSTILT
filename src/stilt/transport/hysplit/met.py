"""
HYSPLIT's meteorology: the met config (:class:`MetConfig`), and finding, downloading, and cropping ARL files (:class:`Met`).
"""

from __future__ import annotations

import bisect
import fnmatch
import functools
import inspect
import logging
import os
from collections.abc import Iterator
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from stilt._paths import absolute, atomic_path
from stilt.exceptions import MeteorologyError
from stilt.identity import settings_hash
from stilt.meteorology import run_window
from stilt.spatial import Bounds

if TYPE_CHECKING:
    from arlmet.archives import Archive

logger = logging.getLogger(__name__)

#: The characters ``fnmatch`` reads as a wildcard in a file pattern.
_WILDCARDS = frozenset("*?[")


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
            "under ``directory``, except lock files, backup copies ending in "
            "``~``, and names ending in ``.partial`` or ``.bad``. Required "
            "when ``download`` is not set."
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


def _is_stray(name: str) -> bool:
    """
    Whether the file *name* beside a met file is not one.

    Files are found by the start of their names, so a sibling such as
    ``20240702_00-05_hrrr.partial`` would be found with the block it sits
    beside. Stray names are lock files, backup copies (``name~<timestamp>~``,
    ``name.~1~``, and ``name~`` all end in ``~``), a download cut short
    (``.partial``), and a copy set aside as damaged (``.bad``).
    """
    return ".lock" in name or name.endswith(("~", ".partial", ".bad"))


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
        #: ``config.directory``, made absolute. A relative path starts from
        #: the working directory, and ``~`` and ``$VARIABLES`` are expanded.
        self.directory = absolute(config.directory)
        self._archive: Archive | None = None
        self._matches: dict[str, list[Path]] = {}
        self._sorted_tails: dict[int, tuple[list[str], list[Path]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @cached_property
    def _listing(self) -> list[tuple[tuple[str, ...], Path]]:
        """
        Every file under :attr:`directory`, walked once, as its path's parts relative to it and the path.

        Lock files, backup copies, and names ending in ``.partial`` or ``.bad`` are left out.
        """
        found: list[tuple[tuple[str, ...], Path]] = []
        for root, names in _walk(self.directory):
            parts = Path(root).relative_to(self.directory).parts
            for name in names:
                if not _is_stray(name):
                    found.append(((*parts, name), Path(root) / name))
        return found

    def _tails(self, depth: int) -> tuple[list[str], list[Path]]:
        """Return the last *depth* parts of each file's path as ``a/b`` strings, sorted, and the files in that order."""
        if depth not in self._sorted_tails:
            tails = sorted(
                ("/".join(parts[-depth:]), path)
                for parts, path in self._listing
                if len(parts) >= depth
            )
            self._sorted_tails[depth] = ([t for t, _ in tails], [p for _, p in tails])
        return self._sorted_tails[depth]

    def _matching(self, pattern: str) -> list[Path]:
        """
        Return the files whose path ends in one that starts with *pattern*, as ``rglob(pattern + "*")`` finds them.

        Each pattern is looked up once. Without a wildcard, the files are a
        run of the sorted paths, found by bisection, so a check of many
        receptors, which looks up thousands of patterns in a folder of
        thousands of files, takes seconds.
        """
        if pattern not in self._matches:
            depth = pattern.count("/") + 1
            tails, paths = self._tails(depth)
            if _WILDCARDS.isdisjoint(pattern):
                start = bisect.bisect_left(tails, pattern)
                end = start
                while end < len(tails) and tails[end].startswith(pattern):
                    end += 1
                candidates = paths[start:end]
            else:
                match = f"{pattern}*"
                candidates = [
                    path
                    for tail, path in zip(tails, paths, strict=True)
                    if fnmatch.fnmatchcase(tail, match)
                ]
            self._matches[pattern] = [path for path in candidates if path.is_file()]
        return self._matches[pattern]

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

    @cached_property
    def _file_freq(self) -> str | None:
        """Return the period of one local file as a pandas frequency, or None for a download."""
        if self.config.download is not None or self.config.file_tres is None:
            return None
        # pandas-stubs' to_offset takes no Timedelta (pandas does)
        # pyrefly: ignore[no-matching-overload]
        return to_offset(pd.to_timedelta(self.config.file_tres)).freqstr

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
        file_format = self.config.file_format
        assert file_format is not None  # MetConfig requires it without download
        met_times, earlier, later = self._file_times(window, hour_after)
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
                f"(each {self.config.file_tres}, named {', '.join(missing)}...). "
                f"The run covers {earlier:%Y-%m-%d %H:%M} to "
                f"{later:%Y-%m-%d %H:%M}."
            )
        return self._dedupe_matched_files(files)

    def _file_times(
        self, window: tuple[Any, Any], hour_after: bool
    ) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
        """Return the times of the local files that cover *window*, and the first and last time they must cover."""
        tres = self._file_freq
        assert tres is not None  # MetConfig requires file_tres without download
        earlier, later = _cover(window, hour_after, tres)
        return pd.date_range(earlier.floor(tres), later, freq=tres), earlier, later

    def missing_hours(
        self, window: tuple[Any, Any], *, hour_after: bool = False
    ) -> list[pd.Timestamp]:
        """
        Return the times of the local files that cover *window* and are not found.

        The files :meth:`files_for` fails for, without raising. Empty when
        every file is there, and always empty with ``download``, which
        fetches the files it lacks.

        Parameters
        ----------
        window : tuple of datetime-like
            The time to cover (:func:`run_window`).
        hour_after : bool, default False
            Also cover the hour after the window's end, as :meth:`files_for`
            does.
        """
        if self.config.download is not None:
            return []
        file_format = self.config.file_format
        assert file_format is not None  # MetConfig requires it without download
        met_times, _, _ = self._file_times(window, hour_after)
        return [t for t in met_times if not self._matching(t.strftime(file_format))]

    def check_receptors(self, receptors: pd.DataFrame, n_hours: int) -> pd.DataFrame:
        """
        Return what would stop runs of *receptors* before HYSPLIT starts, found without running.

        The file check that :meth:`files_for` makes for each run, made for
        every receptor at once: the met's ``directory`` exists and holds an
        ARL file, and every file a run needs is there. With the crop on, a
        receptor outside ``subgrid_bounds`` is listed too, since HYSPLIT
        cannot start a particle outside the files. A met with ``download``
        has no files to look for before a run, so only its crop is checked.
        This looks for files, not inside them, except that it opens the first
        file the receptors need to count its time periods, which HYSPLIT
        needs to be at least two. :meth:`check` opens each file at a run, and
        a run that passes this one can still fail as ``MET_COVERAGE`` when a
        file is damaged or cut short.

        Parameters
        ----------
        receptors : pandas.DataFrame
            One row per release point, with ``receptor`` (the id), ``time``,
            ``longitude``, and ``latitude`` columns.
        n_hours : int
            The length of the runs, negative for backward.

        Returns
        -------
        pandas.DataFrame
            One row for each thing found, with ``check`` (``met_directory``,
            ``met_files``, ``met_periods``, or ``met_crop``), ``receptor`` (``NA`` when the
            finding is about the met), and ``message`` columns. Empty when
            nothing would stop a run.
        """
        found: list[tuple[str, str | None, str]] = []
        if self.config.download is None:
            problem = self._directory_problem()
            if problem is not None:
                found.append(("met_directory", None, problem))
            else:
                found += self._missing_files(receptors, n_hours)
                found += self._one_period(receptors, n_hours)
        found += self._outside_crop(receptors)
        return pd.DataFrame(found, columns=["check", "receptor", "message"])

    def _directory_problem(self) -> str | None:
        """Return why ``directory`` cannot be a local met's, or ``None`` when it holds an ARL file."""
        if not self.directory.is_dir():
            return f"{self.directory} does not exist."
        try:
            self.config.source()
        except FileNotFoundError as error:
            return str(error)
        return None

    def _missing_files(
        self, receptors: pd.DataFrame, n_hours: int
    ) -> list[tuple[str, str | None, str]]:
        """List the receptors that lack a file their run needs, and the hours of those files."""
        first = receptors.drop_duplicates("receptor")
        messages: dict[Any, str | None] = {}
        for time in first["time"].drop_duplicates():
            window = run_window(time, n_hours)
            hours = self.missing_hours(window, hour_after=n_hours < 0)
            messages[time] = (
                "no met file for " + ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in hours)
                if hours
                else None
            )
        return [
            ("met_files", str(rid), message)
            for rid, time in zip(first["receptor"], first["time"], strict=True)
            if (message := messages[time]) is not None
        ]

    def _one_period(
        self, receptors: pd.DataFrame, n_hours: int
    ) -> list[tuple[str, str | None, str]]:
        """List the met when the first file the receptors need holds one time period, which HYSPLIT cannot read."""
        file_format, tres = self.config.file_format, self._file_freq
        assert file_format is not None and tres is not None  # a local met has both
        start = _time(run_window(receptors["time"].min(), n_hours)[0]).floor(tres)
        found = self._matching(start.strftime(file_format))
        if found and _count_times(found[0]) == 1:
            return [("met_periods", None, _one_period_message(found[0]))]
        return []

    def _outside_crop(
        self, receptors: pd.DataFrame
    ) -> list[tuple[str, str | None, str]]:
        """List the receptors with a release point outside the crop box, when the met is cropped."""
        crop = self.config.crop()
        if crop is None:
            return []
        west, south, east, north = crop["bbox"]
        lon, lat = receptors["longitude"], receptors["latitude"]
        outside = receptors[(lon < west) | (lon > east) | (lat < south) | (lat > north)]
        box = f"{west:g} to {east:g} longitude, {south:g} to {north:g} latitude"
        return [
            (
                "met_crop",
                str(row.receptor),
                f"longitude {row.longitude:g} and latitude {row.latitude:g} are "
                f"outside subgrid_bounds ({box})",
            )
            for row in outside.drop_duplicates("receptor").itertuples()
        ]

    def check(
        self, files: list[Path], window: tuple[Any, Any], *, hour_after: bool = False
    ) -> None:
        """
        Raise unless *files* are whole and hold every time step *window* needs.

        A met file can be damaged and keep its name: a time step written
        partway, a missing time step, or records lost to null bytes.
        HYSPLIT then stops its particles where the damage starts, and when
        some of them have left the met's domain first, its log does not say
        why. Each file is checked once in a process
        (:meth:`arlmet.File.check`), and together their time steps must
        reach from the first time the run needs to the last, with no gap
        wider than their spacing. That also finds a file cut after a whole
        time step, which looks whole to arlmet. Files that cover the run but
        include one with a single time period are refused last, since HYSPLIT
        cannot read it.

        Parameters
        ----------
        files : list of Path
            The files the transport model will read (:meth:`readable`).
        window : tuple of datetime-like
            The time the run covers (:func:`run_window`).
        hour_after : bool, default False
            Also need the hour after the window, as :meth:`files_for` does.

        Raises
        ------
        MeteorologyError
            Naming the damaged file and its first problem, a file that holds
            one time period, or the times the files do not hold.
        """
        held: set[pd.Timestamp] = set()
        single: list[Path] = []
        for path in files:
            times, problems = _inspect(path)
            if problems:
                more = f" ({len(problems) - 1} more)" if len(problems) > 1 else ""
                raise MeteorologyError(
                    f"The met file {path} is damaged: {problems[0]}{more}"
                )
            if len(times) == 1:
                single.append(path)
            held.update(times)

        earlier, later = _cover(window, hour_after, self._file_freq)
        needs = f"The run needs {earlier:%Y-%m-%d %H:%M} to {later:%Y-%m-%d %H:%M}."
        steps = sorted(held)
        if not steps or steps[0] > earlier or steps[-1] < later:
            holds = (
                f"{steps[0]:%Y-%m-%d %H:%M} to {steps[-1]:%Y-%m-%d %H:%M}"
                if steps
                else "no time steps"
            )
            raise MeteorologyError(f"The met files hold {holds}. {needs}")
        pairs = list(pairwise(steps))
        spacing = min((b - a for a, b in pairs), default=pd.Timedelta(0))
        for a, b in pairs:
            if b - a > spacing and a < later and b > earlier:
                raise MeteorologyError(
                    f"The met files hold no time step between "
                    f"{a:%Y-%m-%d %H:%M} and {b:%Y-%m-%d %H:%M}. {needs}"
                )
        if single:
            raise MeteorologyError(_one_period_message(single[0]))

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


def _inspect(path: Path) -> tuple[tuple[pd.Timestamp, ...], tuple[str, ...]]:
    """Return the time steps of the ARL file at *path* and what is wrong with it, read once per version of the file."""
    stat = path.stat()
    return _inspect_file(str(path), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=4096)
def _inspect_file(
    path: str, size: int, mtime_ns: int
) -> tuple[tuple[pd.Timestamp, ...], tuple[str, ...]]:
    """
    Open the ARL file at *path* and check it (:meth:`arlmet.File.check`).

    *size* and *mtime_ns* are only part of the cache key, so a file
    rewritten in place is read again. A check reads every record's header,
    seconds for a large file read from disk, so a process checks each file
    once, however many runs read it.
    """
    del size, mtime_ns
    from arlmet import ARLFormatError, File

    try:
        with File(path) as met:
            return tuple(met.times), tuple(met.check())
    except ARLFormatError as exc:
        return (), (str(exc),)


def _count_times(path: Path) -> int | None:
    """Return the number of time periods in the ARL file at *path*, or ``None`` when it cannot be read as one."""
    from arlmet import ARLFormatError, File

    try:
        with File(path) as met:
            return len(met.times)
    except (ARLFormatError, OSError):
        return None


def _one_period_message(path: Path) -> str:
    """Return why the met file at *path*, which holds one time period, cannot be used."""
    return (
        f"The met file {path} holds one time period. HYSPLIT stops on a file "
        "with one ('Only one time period of meteo data'), so each file must "
        "hold at least two, such as the 6-hour files of NOAA's HRRR archive."
    )


def _walk(directory: Path) -> Iterator[tuple[str, list[str]]]:
    """
    Yield each folder under *directory*, itself first, with the names in it that are not folders.

    Like :func:`os.walk`, it does not go into a symlink to a folder and skips
    a folder it cannot read. Unlike it, it tells a folder from a file without
    following symlinks, so it opens no symlink's target: an archive of
    symlinks, such as one link per met file beside the files in
    ``year/month`` folders, is listed from its folders alone. A symlink to a
    folder is listed with the names, and a pattern that matches it finds
    nothing, since only files are kept (:meth:`Met._matching`).
    """
    folders = [str(directory)]
    while folders:
        folder = folders.pop(0)
        names: list[str] = []
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        folders.append(entry.path)
                    else:
                        names.append(entry.name)
        except OSError:
            continue
        yield folder, names


def _arl_candidates(directory: Path) -> Iterator[Path]:
    """Yield the files under *directory*, in name order, files before folders at each level."""
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except FileNotFoundError:
        return
    folders = []
    for entry in entries:
        if _is_stray(entry.name):
            continue  # as Met.files_for skips them
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
        If *directory* does not exist, or no file under it is an ARL file.
    """
    from arlmet import IndexRecord

    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} does not exist.")
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
