"""
The GRIB files of a FLEXPART met: their valid times, and the ones a run needs.
"""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

from stilt.exceptions import MeteorologyError


def _first_message_keys(path: str | Path, keys: tuple[str, ...]) -> list:
    """Return *keys* of the first GRIB message in *path*, read with eccodes."""
    try:
        import eccodes
    except ImportError as err:
        raise ImportError(
            'FLEXPART runs read GRIB met with eccodes: pip install "pystilt[grib]"'
        ) from err
    with open(path, "rb") as f:
        handle = eccodes.codes_grib_new_from_file(f)
        if handle is None:
            raise ValueError(f"{path} holds no GRIB message.")
        try:
            return [eccodes.codes_get(handle, key) for key in keys]
        finally:
            eccodes.codes_release(handle)


def grib_valid_time(path: str | Path) -> dt.datetime:
    """Return the valid time of the first GRIB message in *path*."""
    date, time = _first_message_keys(path, ("validityDate", "validityTime"))
    return dt.datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")


def grib_grid(path: str | Path) -> str:
    """Return the grid of the first GRIB message in *path*, such as ``0.5x0.5 deg, 720x361``."""
    di, dj, ni, nj = _first_message_keys(
        path,
        (
            "iDirectionIncrementInDegrees",
            "jDirectionIncrementInDegrees",
            "Ni",
            "Nj",
        ),
    )
    return f"{di:g}x{dj:g} deg, {ni}x{nj}"


@functools.lru_cache(maxsize=16)
def _catalog(files: tuple[tuple[str, int, int], ...]) -> dict[dt.datetime, Path]:
    """Return the files by valid time; *files* are (path, size, mtime) so a changed file is read again."""
    by_time: dict[dt.datetime, Path] = {}
    for name, _, _ in files:
        valid = grib_valid_time(name)
        if valid in by_time:
            raise MeteorologyError(
                f"{by_time[valid].name} and {Path(name).name} are both for {valid}."
            )
        by_time[valid] = Path(name)
    return dict(sorted(by_time.items()))


def catalog(directory: str | Path, pattern: str = "*") -> dict[dt.datetime, Path]:
    """
    Return the GRIB files in *directory* that match *pattern*, by valid time.

    Each file is read once per process, and again only if it changes.

    Raises
    ------
    MeteorologyError
        If two files have the same valid time.
    """
    paths = sorted(p for p in Path(directory).glob(pattern) if p.is_file())
    key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    return _catalog(key)


def files_for(
    available: dict[dt.datetime, Path], window: tuple[dt.datetime, dt.datetime]
) -> dict[dt.datetime, Path]:
    """
    Return the files a run over *window* reads: from the last at or before its start to the first at or after its end.

    FLEXPART interpolates in time between files, so it needs one at or
    beyond each end.

    Raises
    ------
    MeteorologyError
        If no file is at or before the start or at or after the end, or two
        files in between are further apart than the closest two files in
        the directory (a missing file).
    """
    start, end = window
    times = list(available)
    before = [t for t in times if t <= start]
    after = [t for t in times if t >= end]
    if not before or not after:
        span = f"{times[0]} to {times[-1]}" if times else "no files"
        raise MeteorologyError(
            f"The GRIB files ({span}) do not cover the run from {start} to {end}."
        )
    chosen = [t for t in times if before[-1] <= t <= after[0]]
    step = min((b - a for a, b in zip(times, times[1:], strict=False)), default=None)
    gaps = [b - a for a, b in zip(chosen, chosen[1:], strict=False)]
    if step is not None and gaps and max(gaps) > step:
        raise MeteorologyError(
            f"The GRIB files for the run from {start} to {end} have a gap of "
            f"{max(gaps)}, where the closest are {step} apart: a file is missing."
        )
    return {t: available[t] for t in chosen}


__all__ = ["catalog", "files_for", "grib_grid", "grib_valid_time"]
