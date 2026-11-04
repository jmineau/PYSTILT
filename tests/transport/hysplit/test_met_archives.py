"""
Tests that a met finds the files a run needs for every archive in arlmet.

``Met`` chooses files by its own rules: from ``file_format``, ``file_tres`` and
the spacing read from a file for local files, and from ``Archive.step`` for a
download. arlmet's ``Archive.paths_for_range`` knows each archive's real file
layout. The archives differ in the two facts those rules turn on, the *step*
(1, 3 or 6 hours) and the *file span* (6 hours, a day, a week of the month, a
month), so these tests run the rules against every archive in
``arlmet.archives.ARCHIVES`` rather than one made-up layout.

What a run needs is worked out here from the archive alone: every met step from
the one at or before the run's start to the one at or after its end (HYSPLIT
interpolates between the steps on either side of each of its times), and the
file each step is in. Nothing here copies ``Met``'s own arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from arlmet.archives import ARCHIVES, Archive, get_archive

from stilt.exceptions import MeteorologyError
from stilt.transport.hysplit import Met, MetConfig

from ...fixtures.factories import write_arl_file

#: How to find the files of an archive on disk: the ``file_format`` and
#: ``file_tres`` a user writes for files named as NOAA names them.
#: ``test_the_local_formats_name_the_archives_files`` checks each against the
#: archive's own file names. A monthly file has the ``file_tres`` of a day: its
#: format names the same file for every day of the month, and files are found
#: once whatever the number of days.
LOCAL_FILES = {
    "hrrr": ("%Y%m%d_%H", "6h"),
    "hrrr.v1": ("hysplit.%Y%m%d.%Hz.hrrra", "6h"),
    "nams": ("%Y%m%d_hysplit.t00z.namsa", "1D"),
    "nam12": ("%Y%m%d_nam12", "1D"),
    "gdas0p5": ("%Y%m%d_gdas0p5", "1D"),
    "gfs0p25": ("%Y%m%d_gfs0p25", "1D"),
    "narr": ("NARR%Y%m", "1D"),
    "reanalysis": ("RP%Y%m.gbl", "1D"),
}

#: Archives whose files no ``file_format`` can name, and why.
NOT_FOUND_LOCALLY = {
    "gdas1": (
        "its weekly files are named by the week of the month (w1 to w5) and a "
        "lowercase month, which no strftime code writes"
    ),
}

#: Where the runs end. Each is a boundary of the files of some archive, the
#: first of its month or its week: the day a leap February ends, a new year, a
#: month after a February of 28 days, the second and the fifth week of a month
#: (days 8 and 29), and the first day after a month of 31 days.
CENTERS = {
    "leap-february": "2020-03-01",
    "new-year": "2021-01-01",
    "short-february": "2021-03-01",
    "second-week": "2021-06-08",
    "fifth-week": "2021-06-29",
    "31-day-month": "2021-08-01",
}

#: The lengths of the runs: 3 hours reaches into at most one file before the
#: one it ends in, 30 hours into several 6-hour files.
LENGTHS = [pd.Timedelta(hours=3), pd.Timedelta(hours=30)]

#: The archives, as the registry has them now, so one added to arlmet is
#: tested without a change here.
ARCHIVE_NAMES = sorted(ARCHIVES)
LOCAL_NAMES = [name for name in ARCHIVE_NAMES if name in LOCAL_FILES]


# ---------------------------------------------------------------------------
# What the archive says
# ---------------------------------------------------------------------------


def _file_of(archive: Archive, time: pd.Timestamp) -> str:
    """Return the name of the archive's file that holds the met step at *time*."""
    (path,) = archive.paths_for_range(time, time)
    return path.split("/")[-1]


def _needed(archive: Archive, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Return the files of the met steps a run from *start* to *end* reads, in time order."""
    steps = pd.date_range(
        start.floor(archive.step), end.ceil(archive.step), freq=archive.step
    )
    return list(dict.fromkeys(_file_of(archive, step) for step in steps))


def _steps_around(center: str, freq: pd.Timedelta) -> pd.DatetimeIndex:
    """Return the times *freq* apart in the three days either side of *center*."""
    middle = pd.Timestamp(center)
    return pd.date_range(
        middle - pd.Timedelta(days=3),
        middle + pd.Timedelta(days=3),
        freq=freq,
        inclusive="left",
    )


def _ends(center: str) -> pd.DatetimeIndex:
    """Return the ends of the runs around *center*: every 30 minutes, a day either side."""
    middle = pd.Timestamp(center)
    return pd.date_range(
        middle - pd.Timedelta(days=1), middle + pd.Timedelta(days=1), freq="30min"
    )


# ---------------------------------------------------------------------------
# A met of each kind
# ---------------------------------------------------------------------------


def _download_met(tmp_path: Path, name: str) -> Met:
    """Return a met that downloads *name*, from the real archive with no network."""
    archive = get_archive(name)

    def fetch(start, end, **kwargs):
        return [
            tmp_path / p.split("/")[-1] for p in archive.paths_for_range(start, end)
        ]

    archive.fetch = fetch  # type: ignore[method-assign]
    met = Met(name, MetConfig(directory=tmp_path, download=name))
    met._archive = archive
    return met


def _local_met(tmp_path: Path, name: str, center: str) -> Met:
    """
    Return a met of local files named as *name*'s archive names them.

    Each file holds the archive's steps for the three days either side of
    *center* that fall in it, so a run around *center* has every step it needs.
    """
    archive = get_archive(name)
    held: dict[str, list[pd.Timestamp]] = {}
    for step in _steps_around(center, archive.step):
        held.setdefault(_file_of(archive, step), []).append(step)
    for file_name, times in held.items():
        write_arl_file(tmp_path / file_name, times=times)
    file_format, file_tres = LOCAL_FILES[name]
    return Met(
        name,
        MetConfig(directory=tmp_path, file_format=file_format, file_tres=file_tres),
    )


def _met(kind: str, tmp_path: Path, name: str, center: str) -> Met:
    return (
        _download_met(tmp_path, name)
        if kind == "download"
        else _local_met(tmp_path, name, center)
    )


#: ``(archive, kind)`` for every way a met gets the files of an archive.
WAYS = [
    (name, kind)
    for name in ARCHIVE_NAMES
    for kind in ("download", "local")
    if kind == "download" or name in LOCAL_FILES
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_archive_is_found_locally_or_has_a_reason_it_is_not():
    """A new archive in arlmet fails here until its local format is written down."""
    assert set(LOCAL_FILES) | set(NOT_FOUND_LOCALLY) == set(ARCHIVES)
    assert not set(LOCAL_FILES) & set(NOT_FOUND_LOCALLY)


@pytest.mark.parametrize("name", LOCAL_NAMES)
def test_the_local_formats_name_the_archives_files(name):
    """``file_format`` of each file's period starts the name the archive gives that file."""
    archive = get_archive(name)
    file_format, file_tres = LOCAL_FILES[name]
    times = [
        time
        for center in CENTERS.values()
        for time in _steps_around(center, pd.Timedelta(hours=1))
    ]
    for time in times:
        wanted = time.floor(file_tres).strftime(file_format)
        assert _file_of(archive, time).startswith(wanted), (time, wanted)


@pytest.mark.parametrize("center", CENTERS.values(), ids=CENTERS.keys())
@pytest.mark.parametrize(("name", "kind"), WAYS)
def test_a_run_gets_the_files_of_every_met_step_it_reads(tmp_path, name, kind, center):
    """
    Every run end around a file boundary, every 30 minutes, gets the archive's own files.

    The files are those of the steps from the one at or before the run's start
    to the one at or after its end, and nothing more: no file past the last
    step of the file a run ends in. Local files must also pass the check that
    they hold the steps.
    """
    archive = get_archive(name)
    met = _met(kind, tmp_path, name, center)
    wrong = []
    for length in LENGTHS:
        for end in _ends(center):
            window = (end - length, end)
            wanted = _needed(archive, *window)
            try:
                found = met.files_for(window)
                if kind == "local":
                    met.check(found, window)
            except MeteorologyError as exc:
                wrong.append(f"{window[0]} to {end}: {exc}")
                continue
            if [p.name for p in found] != wanted:
                wrong.append(
                    f"{window[0]} to {end}: {[p.name for p in found]}, not {wanted}"
                )
    assert not wrong, f"{len(wrong)} runs: " + "\n".join(wrong[:5])


@pytest.mark.parametrize(
    ("steps", "minutes", "last_file"),
    [
        # The end is `steps` met steps and `minutes` past the first step of
        # the next file; `last_file` is the file of the run's last met step.
        pytest.param(-2, 1, "before", id="mid-file"),
        pytest.param(-1, 0, "before", id="on-the-last-step-of-the-file"),
        pytest.param(-1, 1, "after", id="a-minute-past-the-last-step"),
        pytest.param(0, -1, "after", id="a-minute-before-the-next-file"),
        pytest.param(0, 0, "after", id="on-the-first-step-of-the-next-file"),
        pytest.param(0, 1, "after", id="a-minute-past-the-first-step"),
    ],
)
@pytest.mark.parametrize(("name", "kind"), WAYS)
def test_a_run_ending_at_the_edge_of_a_file_reads_the_file_it_should(
    tmp_path, name, kind, steps, minutes, last_file
):
    """
    Where a run ends in its file decides whether the next file is read.

    Spelled out for each case, so the expectation does not come from the
    rule: a run that ends on or before the last step of a file reads no
    further, and one that ends past it needs the next file's first step.
    """
    archive = get_archive(name)
    anchor = pd.Timestamp("2021-06-29 12:00")
    edge = next(
        t
        for t in pd.date_range(anchor, periods=24 * 40, freq="1h")
        if _file_of(archive, t) != _file_of(archive, anchor)
    )
    end = edge + steps * archive.step + pd.Timedelta(minutes=minutes)
    met = _met(kind, tmp_path, name, "2021-06-29")

    found = [p.name for p in met.files_for((end - pd.Timedelta(hours=3), end))]

    before, after = _file_of(archive, edge - archive.step), _file_of(archive, edge)
    assert found[-1] == (before if last_file == "before" else after)
    if last_file == "before":
        assert after not in found
