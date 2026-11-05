"""
HYSPLIT's meteorology: the met config (:class:`MetConfig`), and finding, downloading, and cropping ARL files (:class:`Met`).
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from collections.abc import Iterable, Iterator
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from stilt._paths import absolute, atomic_path
from stilt._transient import is_transient
from stilt.exceptions import MeteorologyError
from stilt.identity import settings_hash
from stilt.meteorology import run_window
from stilt.spatial import Bounds

from .catalog import Catalog, catalog_path, is_stray

if TYPE_CHECKING:
    from arlmet.archives import Archive

logger = logging.getLogger(__name__)


class MetConfig(BaseModel):
    """
    HYSPLIT's config for one met, as written under ``mets:`` in ``config.yaml``.

    HYSPLIT reads ARL files. A met of files you have needs only their folder,
    so a string is the ``directory``: ``hrrr: /path/to/hrrr``. Each file's own
    header says which times it holds, whatever it is named. With ``download``,
    arlmet downloads them, and other keys are options for that archive (such
    as ``domain`` for ``nams``). Any other unknown key is an error.

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
            "such as ``hrrr``, ``nam12``, ``gdas1``, or ``gfs0p25``. Options "
            "for the archive (such as ``domain: ak``) go in the same entry."
        ),
    )
    download_from: Literal["s3", "ftp", "http"] = Field(
        "s3",
        description="Server to download from: ``s3``, ``ftp``, or ``http``.",
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

    @model_validator(mode="before")
    @classmethod
    def _folder(cls, data: Any) -> Any:
        """Read a string as the met's folder: ``hrrr: /path/to/hrrr``."""
        return {"directory": data} if isinstance(data, str | os.PathLike) else data

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


def _download_cover(
    window: tuple[Any, Any], step: pd.Timedelta
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the first and last time downloaded files must cover for *window*.

    HYSPLIT needs the met step at or after the window's end, so the end is
    taken up to the next time step of the archive (*step*, from
    ``Archive.step``), as the rule for local files has it. A run that ends
    after the last step of a file gets the next file, such as the next
    day's with 3-hourly steps in daily files, and one that ends on a step
    gets that step's file and no more.
    """
    earlier, later = (_time(t) for t in window)
    return earlier, _time(later.ceil(f"{int(step.total_seconds())}s"))


class Met:
    """
    The meteorology files of one met, found locally or downloaded.

    Without ``download``, files are found under ``directory`` by what their
    headers say they hold (:class:`~stilt.transport.hysplit.catalog.Catalog`).
    With ``download`` set to an ARL archive name, arlmet downloads the files,
    cropping them as it goes when subgridding is on. Local files are cropped with
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
        #: What the files under :attr:`directory` hold, read from their headers.
        self.catalog = Catalog(
            self.directory,
            catalog_path(
                self.directory,
                None if config.subgrid_dir is None else absolute(config.subgrid_dir),
            ),
        )
        self._archive: Archive | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _download(self, window: tuple[Any, Any]) -> list[Path]:
        """Return the files that cover *window* from the ARL archive, downloading any not yet in ``directory``."""
        archive = self._get_archive()
        t_start, t_end = _download_cover(window, archive.step)

        bbox = self._bbox() if self.config.subgrid_enable else None
        levels = self._level_indices() if self.config.subgrid_enable else None

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

    def files_for(self, window: tuple[Any, Any]) -> list[Path]:
        """
        Return the met files that cover *window*, ``(start, end)`` in time order.

        HYSPLIT steps through a run with the two met steps around each of
        its times, so in either direction it needs the met step at or after
        the window's end: the release of a backward run, the end of a
        forward one. The files are the ones whose headers say they hold the
        steps from the one at or before the start to that one
        (:meth:`Catalog.cover`), whatever they are named. Downloads the files
        first when ``download`` is set.

        Parameters
        ----------
        window : tuple of datetime-like
            The time to cover (:func:`run_window`).

        Returns
        -------
        list of Path

        Raises
        ------
        MeteorologyError
            A file the run needs is missing, or files of two products hold
            the same time. Every file is needed: HYSPLIT would stop the
            particles where the met runs out.
        """
        if self.config.download is not None:
            return self._download(window)

        earlier, later = (_time(t) for t in window)
        cover = self.catalog.cover(earlier, later)
        if cover.clash is not None:
            raise MeteorologyError(cover.clash)
        if cover.gaps:
            raise MeteorologyError(
                f"No met file in {self.directory} for {_spans(cover.gaps)}. "
                f"The run covers {earlier:%Y-%m-%d %H:%M} to "
                f"{later:%Y-%m-%d %H:%M}."
            )
        return cover.files

    def missing(
        self, window: tuple[Any, Any]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return the stretches of *window* that no local file holds.

        The stretches :meth:`files_for` fails for, without raising. Empty
        when every file is there, and always empty with ``download``, which
        fetches the files it lacks.

        Parameters
        ----------
        window : tuple of datetime-like
            The time to cover (:func:`run_window`).

        Returns
        -------
        list of tuple of pandas.Timestamp
            Each stretch's first and last time.
        """
        if self.config.download is not None:
            return []
        earlier, later = (_time(t) for t in window)
        return self.catalog.cover(earlier, later).gaps

    def check_receptors(self, receptors: pd.DataFrame, n_hours: int) -> pd.DataFrame:
        """
        Return what would stop runs of *receptors* before HYSPLIT starts, found without running.

        The file check that :meth:`files_for` makes for each run, made for
        every receptor at once: the met's ``directory`` exists and holds an
        ARL file, and every file a run needs is there, holds at least two
        time periods, and ends on a whole time step. With the crop on, a
        receptor outside ``subgrid_bounds`` is listed too, since HYSPLIT
        cannot start a particle outside the files. A met with ``download``
        has no files to look for before a run, so only its crop is checked.

        This brings the table of what the files hold up to date first
        (:meth:`Catalog.refresh`), so it is the place to build it before a
        batch: a first read of a large archive takes minutes. It reads each
        file's first two index records, not the rest. :meth:`check` reads
        each file at a run, and a run that passes this one can still fail as
        ``MET_COVERAGE`` when a file is damaged in the middle.

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
            ``met_files``, ``met_periods``, ``met_cut``, or ``met_crop``), ``receptor`` (``NA`` when the
            finding is about the met), and ``message`` columns. Empty when
            nothing would stop a run.
        """
        found: list[tuple[str, str | None, str]] = []
        if self.config.download is None:
            problem = self._directory_problem()
            if problem is not None:
                found.append(("met_directory", None, problem))
            else:
                self.catalog.refresh()
                found += self._file_problems(receptors, n_hours)
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

    def _file_problems(
        self, receptors: pd.DataFrame, n_hours: int
    ) -> list[tuple[str, str | None, str]]:
        """List the receptors that lack a file their run needs, and the files they need that cannot be read to the end."""
        first = receptors.drop_duplicates("receptor")
        messages: dict[Any, str | None] = {}
        single: dict[Path, None] = {}
        cut: dict[Path, None] = {}
        for time in first["time"].drop_duplicates():
            earlier, later = (_time(t) for t in run_window(time, n_hours))
            cover = self.catalog.cover(earlier, later)
            messages[time] = cover.clash or (
                f"no met file for {_spans(cover.gaps)}" if cover.gaps else None
            )
            single.update(dict.fromkeys(cover.single))
            cut.update(dict.fromkeys(cover.cut))
        found: list[tuple[str, str | None, str]] = [
            ("met_files", str(rid), message)
            for rid, time in zip(first["receptor"], first["time"], strict=True)
            if (message := messages[time]) is not None
        ]
        if single:
            more = f" {len(single) - 1} more do too." if len(single) > 1 else ""
            found.append(
                ("met_periods", None, _one_period_message(next(iter(single))) + more)
            )
        found += [
            (
                "met_cut",
                None,
                f"The met file {path} ends partway through a time step, so it "
                "is damaged. Get it again.",
            )
            for path in cut
        ]
        return found

    def _outside_crop(
        self, receptors: pd.DataFrame
    ) -> list[tuple[str, str | None, str]]:
        """List the receptors with a release point outside the crop box, when the met is cropped."""
        bounds = self.config.subgrid_bounds
        if not self.config.subgrid_enable or bounds is None:
            return []
        west, south, east, north = bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax
        outside = receptors[
            ~bounds.contains(receptors["longitude"], receptors["latitude"])
        ]
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

    def check(self, files: list[Path], window: tuple[Any, Any]) -> None:
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
        time step, which looks whole to arlmet. HYSPLIT needs the met step at
        or before the window's start and the one at or after its end, and
        nothing outside them. Files that cover the run but include one with a
        single time period are refused last, since HYSPLIT cannot read it.

        Parameters
        ----------
        files : list of Path
            The files the transport model will read (:meth:`readable`).
        window : tuple of datetime-like
            The time the run covers (:func:`run_window`).

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

        earlier, later = (_time(t) for t in window)
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

    def fetch(self, windows: Iterable[tuple[Any, Any]]) -> pd.DataFrame:
        """
        Get the met files that cover every one of *windows*, downloading any a download lacks.

        Each window is asked for as :meth:`files_for` is when a run needs it,
        so the files are the ones those runs would use. Windows that overlap
        are asked for together, one request to the archive for a stretch of
        runs; when that fails, each of its windows is asked for on its own,
        so a failure names only the windows it belongs to. A met without
        ``download`` is only checked: its files are looked for, never made.
        Only a file that is not there is a failure to report. Any other
        error stops the fetch and is raised, whether a retry may get past it
        (a busy archive, a connection that failed) or not (a denied request,
        a full disk), so that ``stilt fetch`` can exit with 3 or 4.

        Parameters
        ----------
        windows : iterable of tuple of datetime-like
            The times to cover, each as :func:`run_window` returns it.

        Returns
        -------
        pandas.DataFrame
            ``file`` and ``message`` columns: a row for each file, with
            ``message`` empty, then a row for each window whose files could
            not be had, with ``file`` empty and the reason in ``message``.
        """
        files: dict[Path, None] = {}
        problems: dict[str, None] = {}

        def get(window: tuple[Any, Any]) -> str | None:
            """Get the files for *window*, or return why they could not be had."""
            try:
                files.update(dict.fromkeys(self.files_for(window)))
            except MeteorologyError as error:
                return str(error)
            except (ValueError, FileNotFoundError) as error:
                # fsspec's HTTP filesystem raises FileNotFoundError for a busy
                # server too: the cause says, and a retry may get past it.
                if is_transient(error):
                    raise
                first, last = (_time(t) for t in window)
                return f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}: {error}"
            return None

        for whole, parts in _overlapping(windows):
            reason = get(whole)
            if reason is not None and len(parts) == 1:
                problems[reason] = None
            elif reason is not None:
                for part in parts:
                    if (reason := get(part)) is not None:
                        problems[reason] = None
        table = [(str(path), None) for path in files]
        table += [(None, reason) for reason in problems]
        return pd.DataFrame(table, columns=["file", "message"]).astype("string")


#: Met settings that were removed, and what to do instead.
_REMOVED = {
    "file_format": (
        "delete it. Each met file's header says which times it holds, so its "
        "name no longer matters."
    ),
    "file_tres": (
        "delete it. Each met file's header says which times it holds, so how "
        "much a file covers is read from the file."
    ),
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


def _spans(gaps: list[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    """Return stretches of time as text."""
    return ", ".join(f"{a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M}" for a, b in gaps)


def _overlapping(
    windows: Iterable[tuple[Any, Any]],
) -> list[
    tuple[tuple[pd.Timestamp, pd.Timestamp], list[tuple[pd.Timestamp, pd.Timestamp]]]
]:
    """
    Group *windows* that overlap, in time order.

    Returns, for each group, the one window that spans it and the distinct
    windows in it. Windows that only touch stay apart.
    """
    ordered = sorted(
        {
            (min(a, b), max(a, b))
            for a, b in ((_time(first), _time(last)) for first, last in windows)
        }
    )
    groups: list[list[tuple[pd.Timestamp, pd.Timestamp]]] = []
    end: pd.Timestamp | None = None
    for window in ordered:
        if end is not None and window[0] < end:
            groups[-1].append(window)
            end = max(end, window[1])
        else:
            groups.append([window])
            end = window[1]
    return [((parts[0][0], max(b for _, b in parts)), parts) for parts in groups]


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


def _one_period_message(path: Path) -> str:
    """Return why the met file at *path*, which holds one time period, cannot be used."""
    return (
        f"The met file {path} holds one time period. HYSPLIT stops on a file "
        "with one ('Only one time period of meteo data'), so each file must "
        "hold at least two, such as the 6-hour files of NOAA's HRRR archive."
    )


def _arl_candidates(directory: Path) -> Iterator[Path]:
    """Yield the files under *directory*, in name order, files before folders at each level."""
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except FileNotFoundError:
        return
    folders = []
    for entry in entries:
        if is_stray(entry.name):
            continue  # as the catalog skips them
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
