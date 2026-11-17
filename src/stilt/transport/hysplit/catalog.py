"""
What the ARL files under a folder hold, read from their own headers (:class:`Catalog`).

An ARL file says which times it holds: its first index record gives the first
time, the grid and the size of a time step, and its second gives the spacing.
So a met needs no pattern of file names. A folder's files are read once
(:func:`arlmet.probe`), kept in a table beside the crops, and a run asks the
table which files hold the met steps it needs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stilt._paths import write_parquet

logger = logging.getLogger(__name__)

#: Threads reading files during a refresh. Reading a header is waiting on the
#: filesystem, not work for a core: 16 read 16,600 HRRR files on NFS in about
#: two minutes, one took 35.
_THREADS = 16

#: How long, in seconds, a process waits for another that is building a folder's
#: first table before building it too: long enough for the largest archive.
_BUILD_WAIT = 600.0

#: The table's columns. ``first`` is ``NaT``, and ``n_steps`` 0, for a file
#: that is not ARL, kept so that it is not opened again until it changes.
_COLUMNS = [
    "path",
    "resolved",
    "size",
    "mtime_ns",
    "first",
    "step_seconds",
    "n_steps",
    "source",
    "grid",
    "whole",
]


def is_stray(name: str) -> bool:
    """
    Whether the file *name* beside a met file is not one.

    A header cannot tell: a crop being written can read as whole time steps
    until it is renamed into place. Stray names are lock files, backup copies
    (``name~<timestamp>~``, ``name.~1~``, and ``name~`` all end in ``~``), a
    download cut short (``.partial``), a copy set aside as damaged
    (``.bad``), and a crop PYSTILT is still writing (``.name.<id>.tmp``).
    """
    return (
        ".lock" in name
        or name.endswith(("~", ".partial", ".bad"))
        or (name.startswith(".") and name.endswith(".tmp"))
    )


def walk(directory: Path) -> Iterator[tuple[str, list[str]]]:
    """
    Yield each folder under *directory*, itself first, with the names in it that are not folders.

    Like :func:`os.walk`, it does not go into a symlink to a folder and skips
    a folder it cannot read. Unlike it, it tells a folder from a file without
    following symlinks, so it opens no symlink's target: an archive of
    symlinks, such as one link per met file beside the files in
    ``year/month`` folders, is listed from its folders alone.
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


def catalog_path(directory: Path, subgrid_dir: Path | None) -> Path:
    """
    Return where the table of *directory* is kept.

    In ``catalogs/`` inside *subgrid_dir*, the place a met already writes to
    and shares, else under the user's cache directory. The name is a hash of
    the folder, so a folder moved is read again.
    """
    if subgrid_dir is not None:
        parent = subgrid_dir / "catalogs"
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        parent = Path(cache) / "pystilt" / "met_catalogs"
    key = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    return parent / f"{key}.parquet"


def _empty() -> pd.DataFrame:
    """Return a table with no files."""
    return pd.DataFrame(
        {
            "path": pd.Series(dtype=str),
            "resolved": pd.Series(dtype=str),
            "size": pd.Series(dtype="int64"),
            "mtime_ns": pd.Series(dtype="int64"),
            "first": pd.Series(dtype="datetime64[ns]"),
            "step_seconds": pd.Series(dtype="float64"),
            "n_steps": pd.Series(dtype="int64"),
            "source": pd.Series(dtype=str),
            "grid": pd.Series(dtype=str),
            "whole": pd.Series(dtype=bool),
        }
    )


def _read_header(path: str, size: int, mtime_ns: int) -> dict:
    """Return the row of the file at *path*, one with no steps when it is not an ARL file."""
    from arlmet import ARLFormatError, probe

    row = {
        "path": path,
        # The name HYSPLIT is given and a crop is named by: the file, not a link.
        "resolved": os.path.realpath(path),
        "size": size,
        "mtime_ns": mtime_ns,
        "first": pd.NaT,
        "step_seconds": np.nan,
        "n_steps": 0,
        "source": "",
        "grid": "",
        "whole": False,
    }
    try:
        found = probe(path)
    except (ARLFormatError, EOFError, OSError, ValueError):
        return row
    return {
        **row,
        "first": found.first,
        "step_seconds": np.nan if found.step is None else found.step.total_seconds(),
        "n_steps": found.n_steps,
        "source": found.source,
        "grid": repr(found.grid),
        "whole": found.whole,
    }


def _stat(path: str) -> tuple[str, os.stat_result] | None:
    """Return *path* and its stat, or ``None`` for a broken link or anything but a file."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (path, info) if stat.S_ISREG(info.st_mode) else None


@dataclass
class Cover:
    """
    The files that hold a stretch of time, as :meth:`Catalog.cover` finds them.

    Attributes
    ----------
    files : list of Path
        The files, in time order, with symlinks resolved.
    firsts, lasts : list of pandas.Timestamp
        The first and last time each of :attr:`files` holds.
    gaps : list of tuple of pandas.Timestamp
        The stretches of the time asked for that no file holds.
    clash : str or None
        Why the files cannot be used: two products hold the same time.
    single : list of Path
        The files with one time period, which HYSPLIT cannot read.
    cut : list of Path
        The files that end partway through a time step.
    """

    files: list[Path] = field(default_factory=list)
    firsts: list[pd.Timestamp] = field(default_factory=list)
    lasts: list[pd.Timestamp] = field(default_factory=list)
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    clash: str | None = None
    single: list[Path] = field(default_factory=list)
    cut: list[Path] = field(default_factory=list)


class Catalog:
    """
    The table of what the ARL files under one folder hold.

    Parameters
    ----------
    directory : Path
        The folder, absolute. Files are found anywhere under it, except
        lock files, backup copies, and names ending in ``.partial`` or
        ``.bad`` (:func:`is_stray`).
    path : Path
        Where the table is kept (:func:`catalog_path`).
    """

    def __init__(self, directory: Path, path: Path):
        self.directory = directory
        self.path = path
        self._table: pd.DataFrame | None = None
        # Whether this process has listed the folder, so it does so only once.
        self._refreshed = False
        # The files found still as the table has them.
        self._confirmed: set[str] = set()

    @property
    def table(self) -> pd.DataFrame:
        """The files that hold time steps, in time order, read or built on first use."""
        if self._table is None:
            kept = self._load()
            self._table = self._build() if kept is None else _usable(kept)
        return self._table

    def _build(self) -> pd.DataFrame:
        """
        Make a folder's first table, one process at a time.

        A job array starts many tasks at once, and each reading a whole
        archive would load its filesystem for nothing. The first makes a lock
        folder and the rest wait for its table. A lock left by a killed job
        only delays them: after :data:`_BUILD_WAIT` they build it themselves.
        """
        lock = self.path.with_name(f"{self.path.name}.lock")
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.mkdir()
        except FileExistsError:
            deadline = time.monotonic() + _BUILD_WAIT
            while lock.exists() and time.monotonic() < deadline:
                time.sleep(5.0)
            kept = self._load()
            return self.refresh() if kept is None else _usable(kept)
        except OSError:  # nowhere to keep it: refresh says so
            return self.refresh()
        try:
            return self.refresh()
        finally:
            lock.rmdir()

    def _load(self) -> pd.DataFrame | None:
        """Return the table kept on disk, or ``None`` when there is none to read."""
        try:
            kept = pd.read_parquet(self.path)
        except (OSError, ValueError):  # missing, or written by another version
            return None
        return kept if list(kept.columns) == _COLUMNS else None

    def refresh(self) -> pd.DataFrame:
        """
        List the folder, read the files that are new or changed, and keep the table.

        A file is known by its path, size and modified time. A file reached
        by two paths, such as a link beside its target, is one file. When the
        table cannot be written, the refresh still serves this process.

        Returns
        -------
        pandas.DataFrame
            The files that hold time steps, in time order.
        """
        kept = self._load()
        known = {
            (row["path"], row["size"], row["mtime_ns"]): row
            for row in ([] if kept is None else kept.to_dict("records"))
        }
        listed = [
            os.path.join(folder, name)
            for folder, names in walk(self.directory)
            for name in names
            if not is_stray(name)
        ]
        rows, to_read, seen = [], [], set()
        with ThreadPoolExecutor(_THREADS) as pool:
            for found in pool.map(_stat, sorted(listed)):
                if found is None:
                    continue
                path, info = found
                if (info.st_dev, info.st_ino) in seen:
                    continue
                seen.add((info.st_dev, info.st_ino))
                row = known.get((path, info.st_size, info.st_mtime_ns))
                if row is None:
                    to_read.append((path, info.st_size, info.st_mtime_ns))
                else:
                    rows.append(row)
            rows += pool.map(lambda args: _read_header(*args), to_read)
        table = pd.DataFrame(rows, columns=_COLUMNS) if rows else _empty()
        table["first"] = pd.to_datetime(table["first"])
        if to_read or kept is None or len(kept) != len(table):
            try:
                write_parquet(_as_arrow(table), self.path)
            except OSError as error:
                logger.warning(
                    "Could not keep the met catalog at %s (%s): each process "
                    "will read the files again.",
                    self.path,
                    error,
                )
        self._table = _usable(table)
        self._refreshed = True
        self._confirmed.clear()
        return self._table

    def _unchanged(self, rows: list[Any]) -> bool:
        """Whether each file in *rows* is still as they have it."""
        for row in rows:
            if row.resolved in self._confirmed:
                continue
            try:
                info = os.stat(row.resolved)
            except OSError:
                return False
            if info.st_size != row.size or info.st_mtime_ns != row.mtime_ns:
                return False
            self._confirmed.add(row.resolved)
        return True

    def cover(self, earlier: pd.Timestamp, later: pd.Timestamp) -> Cover:
        """
        Return the files that hold the met steps of a run from *earlier* to *later*.

        HYSPLIT interpolates between the met steps either side of each of its
        times, so a run needs the last step at or before *earlier*, the first
        at or after *later*, and every step between. The table is trusted
        until it fails: when it has a gap, or a file it chose has changed,
        the folder is listed once in this process and the table is asked again.

        Where files overlap, a file wholly inside another of the same product
        is dropped, so a short crop never hides the whole file. Files of
        different products holding the same time are a :attr:`Cover.clash`: a
        run records one source, and which it means cannot be told from here.
        """
        chosen = self._chosen(earlier, later)
        kept = _longest(chosen)
        if not self._refreshed and (
            _gaps(kept, earlier, later) or not self._unchanged(kept)
        ):
            self.refresh()
            chosen = self._chosen(earlier, later)
            kept = _longest(chosen)
        return Cover(
            files=[Path(row.resolved) for row in kept],
            firsts=[row.first for row in kept],
            lasts=[row.last for row in kept],
            gaps=_gaps(kept, earlier, later),
            clash=_clash(chosen),
            single=[Path(row.resolved) for row in kept if row.n_steps == 1],
            cut=[Path(row.resolved) for row in kept if not row.whole],
        )

    def _chosen(self, earlier: pd.Timestamp, later: pd.Timestamp) -> list[Any]:
        """Return the rows of the files that hold any met step of a run from *earlier* to *later*, in time order."""
        table = self.table
        first = table["first"].to_numpy()
        last = table["last"].to_numpy()
        start, end = _bounds(
            first, last, table["step_seconds"].to_numpy(), earlier, later
        )
        where = np.flatnonzero((last >= start) & (first <= end))
        return list(table.iloc[where].itertuples(index=False))


def _as_arrow(table: pd.DataFrame):
    """Return *table* as a pyarrow table, ready to write."""
    import pyarrow as pa

    return pa.Table.from_pandas(table[_COLUMNS], preserve_index=False)


def _usable(table: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of *table* that hold time steps, in time order, with each file's last step."""
    held = table[table["n_steps"] > 0].sort_values(["first", "path"]).copy()
    step = pd.to_timedelta(held["step_seconds"].fillna(0), unit="s")
    held["last"] = held["first"] + (held["n_steps"] - 1) * step
    return held.reset_index(drop=True)


def _bounds(
    first: np.ndarray,
    last: np.ndarray,
    steps: np.ndarray,
    earlier: pd.Timestamp,
    later: pd.Timestamp,
) -> tuple[np.datetime64, np.datetime64]:
    """
    Return the last step at or before *earlier* and the first step at or after *later*.

    Each must be within a step of the time, or there is a gap at that end, and
    an older file than the gap is no use. Without such a step, the bound is the
    time itself, and :func:`_gaps` reports what is missing.
    """
    start = earlier.to_datetime64()
    end = later.to_datetime64()
    at_second = np.timedelta64(1, "s")
    spacing = np.nan_to_num(steps).astype("timedelta64[s]")

    before = (first <= start) & (last + spacing >= start)
    if before.any():
        since = (np.minimum(last[before], start) - first[before]) / at_second
        size = np.where(np.isnan(steps[before]), 1.0, steps[before])
        done = np.floor(since / size) * size
        start = (first[before] + done.astype("timedelta64[s]")).max()

    after = (last >= end) & (first - spacing <= end)
    if after.any():
        until = (np.maximum(first[after], end) - first[after]) / at_second
        size = np.where(np.isnan(steps[after]), 1.0, steps[after])
        done = np.ceil(until / size) * size
        end = (first[after] + done.astype("timedelta64[s]")).min()
    return start, end


def _clash(chosen: list[Any]) -> str | None:
    """Return why *chosen* cannot be used together: files of different products hold the same time."""
    widest = None
    for row in chosen:
        if (
            widest is not None
            and row.first <= widest.last
            and (row.source, row.grid) != (widest.source, widest.grid)
        ):
            return (
                "Two files of different products or grids hold "
                f"{row.first:%Y-%m-%d %H:%M}: {widest.resolved} and "
                f"{row.resolved}. Keep each product in its own folder, "
                "and give each its own met."
            )
        if widest is None or row.last > widest.last:
            widest = row
    return None


def _longest(chosen: list[Any]) -> list[Any]:
    """Return *chosen*, in time order, without a file wholly inside another of the same product."""
    kept: list[Any] = []
    for row in sorted(chosen, key=lambda r: (-r.n_steps, r.path)):
        if not any(
            (other.source, other.grid) == (row.source, row.grid)
            and other.first <= row.first
            and other.last >= row.last
            for other in kept
        ):
            kept.append(row)
    return sorted(kept, key=lambda r: (r.first, r.path))


def _gaps(
    kept: list[Any], earlier: pd.Timestamp, later: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the stretches from *earlier* to *later* with no met step either side, from the files *kept*."""
    if not kept:
        return [(earlier, later)]
    known = [row.step_seconds for row in kept if row.step_seconds == row.step_seconds]
    step = pd.Timedelta(seconds=min(known)) if known else None
    gaps = []
    if kept[0].first > earlier:
        gaps.append((earlier, min(kept[0].first, later)))
    reach = kept[0].last
    for row in kept[1:]:
        if step is not None and row.first - reach > step:
            gaps.append((reach, row.first))
        reach = max(reach, row.last)
    if reach < later:
        gaps.append((max(reach, earlier), later))
    return gaps
