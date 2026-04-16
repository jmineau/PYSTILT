"""Integration tests for HYSPLIT release-point assignment behavior."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from stilt.config import STILTParams
from stilt.hysplit.runner import HYSPLITRunner
from stilt.meteorology import MetStream
from stilt.receptor import Receptor

from .conftest import integration


def _release_time_rows(particles):
    """Return the rows closest to release time, sorted by particle index."""
    latest_time = particles["time"].max()
    return particles.loc[particles["time"] == latest_time].sort_values("indx")


def _nearest_release_assignments(release_rows, receptor):
    """Return the nearest explicit release-point index for each particle row."""
    release_points = np.column_stack((receptor.longitudes, receptor.latitudes))
    particle_points = release_rows[["long", "lati"]].to_numpy()
    distances = np.sum(
        (particle_points[:, None, :] - release_points[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1).tolist()


@integration
def test_hysplit_multipoint_release_points_follow_control_order(tmp_path, met_dir):
    """Multipoint particles are assigned to explicit points in CONTROL order.

    This test characterizes the compiled HYSPLIT binary directly rather than
    PYSTILT's later ``xhgt`` reconstruction. It uses a divisible particle count
    so each explicit release point should receive the same-size contiguous
    ``indx`` block.
    """

    receptor = Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=[-112.0, -111.8, -111.6],
        latitude=[40.5, 40.5, 40.5],
        altitude=[100.0, 500.0, 900.0],
    )
    params = STILTParams(
        n_hours=-1,
        numpar=12,
        hnf_plume=False,
        rm_dat=True,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met_files = MetStream(
        "hrrr",
        directory=met_dir,
        file_format="%Y%m%d.%Hz.hrrra",
        file_tres="6h",
    ).required_files(r_time=receptor.time, n_hours=params.n_hours)

    runner = HYSPLITRunner(
        receptor=receptor,
        params=params,
        met_files=met_files,
        directory=Path(tmp_path) / "hysplit_assignment",
    )
    runner.prepare()
    result = runner.execute(timeout=120, rm_dat=True)

    release_rows = _release_time_rows(result.particles)
    assert release_rows["indx"].tolist() == list(range(1, params.numpar + 1))

    nearest_release = _nearest_release_assignments(release_rows, receptor)

    assert nearest_release == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]

    grouped_heights = [
        release_rows.iloc[start : start + 4]["zagl"].to_numpy(dtype=float)
        for start in (0, 4, 8)
    ]
    expected_heights = receptor.altitudes.tolist()
    for block, expected in zip(grouped_heights, expected_heights, strict=False):
        assert abs(float(block.mean()) - expected) < 100.0


@integration
def test_hysplit_multipoint_release_points_follow_control_order_nondivisible(
    tmp_path, met_dir
):
    """Nondivisible particle counts still use contiguous blocks in point order."""

    receptor = Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=[-112.0, -111.8, -111.6],
        latitude=[40.5, 40.5, 40.5],
        altitude=[100.0, 500.0, 900.0],
    )
    params = STILTParams(
        n_hours=-1,
        numpar=10,
        hnf_plume=False,
        rm_dat=True,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met_files = MetStream(
        "hrrr",
        directory=met_dir,
        file_format="%Y%m%d.%Hz.hrrra",
        file_tres="6h",
    ).required_files(r_time=receptor.time, n_hours=params.n_hours)

    runner = HYSPLITRunner(
        receptor=receptor,
        params=params,
        met_files=met_files,
        directory=Path(tmp_path) / "hysplit_assignment_nondivisible",
    )
    runner.prepare()
    result = runner.execute(timeout=120, rm_dat=True)

    release_rows = _release_time_rows(result.particles)
    assert release_rows["indx"].tolist() == list(range(1, params.numpar + 1))
    assert _nearest_release_assignments(release_rows, receptor) == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
    ]


@integration
def test_hysplit_column_release_spans_vertical_line_without_endpoint_chunking(
    tmp_path, met_dir
):
    """Column releases span the requested vertical range rather than endpoint chunks."""

    receptor = Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=-112.0,
        latitude=40.5,
        altitude=[5.0, 1000.0],
    )
    params = STILTParams(
        n_hours=-1,
        numpar=12,
        hnf_plume=False,
        rm_dat=True,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met_files = MetStream(
        "hrrr",
        directory=met_dir,
        file_format="%Y%m%d.%Hz.hrrra",
        file_tres="6h",
    ).required_files(r_time=receptor.time, n_hours=params.n_hours)

    runner = HYSPLITRunner(
        receptor=receptor,
        params=params,
        met_files=met_files,
        directory=Path(tmp_path) / "hysplit_column_assignment",
    )
    runner.prepare()
    result = runner.execute(timeout=120, rm_dat=True)

    release_rows = _release_time_rows(result.particles)
    assert release_rows["indx"].tolist() == list(range(1, params.numpar + 1))
    assert (
        release_rows["zagl"].between(receptor.bottom - 50.0, receptor.top + 50.0).all()
    )
    assert release_rows["zagl"].nunique() == params.numpar
    corr = np.corrcoef(
        release_rows["indx"].to_numpy(dtype=float),
        release_rows["zagl"].to_numpy(dtype=float),
    )[0, 1]
    assert float(corr) > 0.98
