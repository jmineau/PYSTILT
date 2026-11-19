"""Tests for HYSPLIT's met config and its files (runtraj.models.hysplit.met)."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from runtraj.exceptions import MeteorologyError
from runtraj.meteorology import run_window
from runtraj.models.hysplit import Met, MetConfig
from runtraj.spatial import Bounds

from ...fixtures.factories import write_arl_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_met(tmp_path: Path) -> Met:
    return Met("hrrr", MetConfig(directory=tmp_path))


# ---------------------------------------------------------------------------
# MetConfig validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Met download (via arlmet)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Met archive subsetting via arlmet.extract_subset
# ---------------------------------------------------------------------------


BOUNDS = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)


SIX_HOURS = pd.date_range("2023-01-01 12:00", periods=6, freq="h")


# ---------------------------------------------------------------------------
# The files a run reads
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backward and forward runs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Insufficient files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The crop box: longitudes count modulo 360 (#326)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Which files are met files: anything with a header, anywhere under the folder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Met steps longer than an hour (#283)
#
# HYSPLIT needs the met step at or before a run's start and the one at or
# after its end. For daily files of 3-hourly steps (NAM12, GDAS 0.5 deg), the
# step after a 22:00 end is the next day's 00:00, and a 21:00 end needs
# nothing past 21:00.
# ---------------------------------------------------------------------------


def _stepped_files(
    tmp_path: Path, names: list[str], file_format: str, tres: str, step: str
) -> list[Path]:
    """Write one ARL file per name in *names*, each holding the steps of *step* in its *tres*."""
    files = []
    for name in names:
        first = pd.Timestamp(dt.datetime.strptime(name, file_format))
        end = first + pd.to_timedelta(tres)
        times = pd.date_range(first, end, freq=step, inclusive="left")
        files.append(write_arl_file(tmp_path / name, times=times))
    return files


#: Daily files of 3-hourly steps, as NAM12's: 00:00 to 21:00 each day.
_NAM = {"names": ["20150101", "20150102"], "file_format": "%Y%m%d", "tres": "1D"}


@pytest.mark.parametrize(
    ("step", "end", "files"),
    [
        # the issue's two backward runs: after the last step, and on it
        ("3h", "2015-01-01 22:00", ["20150101", "20150102"]),
        ("3h", "2015-01-01 21:00", ["20150101"]),
        ("3h", "2015-01-01 21:01", ["20150101", "20150102"]),
        ("3h", "2015-01-01 20:59", ["20150101"]),
        ("3h", "2015-01-02 00:00", ["20150101", "20150102"]),
        # daily files of 6-hourly steps, as GFS 0.25 deg: 00:00 to 18:00
        ("6h", "2015-01-01 18:00", ["20150101"]),
        ("6h", "2015-01-01 19:00", ["20150101", "20150102"]),
    ],
)
@pytest.mark.parametrize("n_hours", [-3, 3])
def test_files_for_reads_the_next_file_only_past_the_last_met_step(
    tmp_path, step, end, files, n_hours
):
    """A backward run released at *end* and a forward run ending then get the same files."""
    _stepped_files(tmp_path, _NAM["names"], _NAM["file_format"], _NAM["tres"], step)
    met = _make_met(tmp_path)
    r_time = (
        pd.Timestamp(end) if n_hours < 0 else pd.Timestamp(end) - pd.Timedelta(hours=3)
    )
    window = run_window(r_time, n_hours)

    found = met.files_for(window)

    assert [p.name for p in found] == files
    met.check(found, window)


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_check_passes_a_run_that_ends_on_the_last_met_step(tmp_path, n_hours):
    """A run released at 21:00 (or ending then) runs on that day's file alone, as HYSPLIT runs it (#283)."""
    files = _stepped_files(
        tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h"
    )
    r_time = dt.datetime(2015, 1, 1, 21) if n_hours < 0 else dt.datetime(2015, 1, 1, 18)

    _make_met(tmp_path).check(files, run_window(r_time, n_hours))


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_check_fails_a_run_past_the_last_met_step_without_the_next_file(
    tmp_path, n_hours
):
    """A run released at 22:00 needs the next day's 00:00 step."""
    files = _stepped_files(
        tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h"
    )
    r_time = dt.datetime(2015, 1, 1, 22) if n_hours < 0 else dt.datetime(2015, 1, 1, 19)
    met = _make_met(tmp_path)

    with pytest.raises(
        MeteorologyError,
        match="hold 2015-01-01 00:00 to 2015-01-01 21:00. The run needs "
        "2015-01-01 19:00 to 2015-01-01 22:00",
    ):
        met.check(files, run_window(r_time, n_hours))


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_missing_lists_the_step_a_run_past_the_last_step_needs(tmp_path, n_hours):
    """`stilt check` asks for the next day's file a 22:00 end needs, and none for 21:00."""
    _stepped_files(tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h")
    met = _make_met(tmp_path)

    def missing(end):
        r_time = pd.Timestamp(end)
        if n_hours > 0:
            r_time -= pd.Timedelta(hours=n_hours)
        return met.missing(run_window(r_time, n_hours))

    assert missing("2015-01-01 22:00") == [
        (pd.Timestamp("2015-01-01 21:00"), pd.Timestamp("2015-01-01 22:00"))
    ]
    assert missing("2015-01-01 21:00") == []


#: 6-hour files of hourly steps, as NOAA's HRRR archive.
_HRRR = {
    "names": ["20210115_00", "20210115_06"],
    "file_format": "%Y%m%d_%H",
    "tres": "6h",
}


@pytest.mark.parametrize(
    ("end", "files"),
    [
        # on the file's last hour: HYSPLIT reads nothing past 05:00
        ("2021-01-15 05:00", ["20210115_00"]),
        ("2021-01-15 05:30", ["20210115_00", "20210115_06"]),
        ("2021-01-15 04:59", ["20210115_00"]),
        ("2021-01-15 06:00", ["20210115_00", "20210115_06"]),
    ],
)
@pytest.mark.parametrize("n_hours", [-3, 3])
def test_files_for_hourly_steps_drop_the_file_after_the_last_hour(
    tmp_path, end, files, n_hours
):
    """With hourly steps only a run that ends exactly on a file's last hour changes: it no longer needs the next file."""
    _stepped_files(tmp_path, _HRRR["names"], _HRRR["file_format"], _HRRR["tres"], "h")
    met = _make_met(tmp_path)
    r_time = (
        pd.Timestamp(end) if n_hours < 0 else pd.Timestamp(end) - pd.Timedelta(hours=3)
    )
    window = run_window(r_time, n_hours)

    found = met.files_for(window)

    assert [p.name for p in found] == files
    met.check(found, window)


def test_the_met_step_skips_a_partial_download(tmp_path):
    """A cut-short download beside a met file, with other steps, is not read for the step."""
    _stepped_files(tmp_path, _NAM["names"], _NAM["file_format"], _NAM["tres"], "3h")
    write_arl_file(
        tmp_path / "20150101.partial",
        times=pd.date_range("2015-01-01", periods=24, freq="h"),
    )
    met = _make_met(tmp_path)

    files = met.files_for(run_window(dt.datetime(2015, 1, 1, 22), -3))

    assert [p.name for p in files] == _NAM["names"]


# ---------------------------------------------------------------------------
# What a run records of the met
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Checking the files a run reads
# ---------------------------------------------------------------------------


def test_a_global_met_grid_holds_every_longitude(tmp_path):
    """On a grid that goes all the way round, no longitude is outside the domain (#440)."""
    import numpy as np
    from arlmet import File, Grid, Projection
    from arlmet.vertical import PressureAxis
    from runtraj.models.hysplit.met import _on_grid

    grid = Grid(
        projection=Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-2.0,
            sync_lon=0.0,
        ),
        nx=360,
        ny=5,
    )
    path = tmp_path / "global.arl"
    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=PressureAxis(levels=[0.0]),
    ) as f:
        rs = f.create_recordset(pd.Timestamp("2023-01-01"))
        rs.create_datarecord(
            "PRSS", level=0, forecast=0, data=np.ones((5, 360), np.float32)
        )
    points = pd.DataFrame(
        {"longitude": [-0.5, 179.9, -179.9], "latitude": [0.0, 2.0, 3.0]}
    )

    assert _on_grid(path, points).tolist() == [True, True, False]
