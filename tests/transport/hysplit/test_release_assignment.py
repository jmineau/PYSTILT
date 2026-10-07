"""Integration tests for HYSPLIT release-point assignment behavior."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from stilt.meteorology import run_window
from stilt.receptors import ColumnReceptor, MultiPointReceptor
from stilt.transport.hysplit import (
    HysplitConfig,
    Met,
    read_particle_dat,
    run_hycs_std,
    write_inputs,
)

from ...conftest import integration
from ...fixtures.factories import make_met_config
from ...fixtures.particles import finished


def _raw_particles(receptor, config, met, workdir, timeout):
    """Run hycs_std and return its particles as HYSPLIT wrote them, before release heights are added."""
    window = run_window(receptor.time, config.n_hours)
    files = met.files_for(window, hour_after=config.n_hours < 0)
    write_inputs(workdir, receptor, config, files)
    run_hycs_std(workdir, timeout)
    return read_particle_dat(workdir / "PARTICLE_STILT.DAT", config.varsiwant)


def _release_time_rows(particles):
    """Return the rows closest to release time, sorted by particle index."""
    latest_time = particles["time"].max()
    return particles.loc[particles["time"] == latest_time].sort_values("particle")


def _nearest_release_assignments(release_rows, receptor):
    """Return the nearest explicit release-point index for each particle row."""
    release_points = np.column_stack((receptor.longitudes, receptor.latitudes))
    particle_points = release_rows[["lon", "lat"]].to_numpy()
    distances = np.sum(
        (particle_points[:, None, :] - release_points[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1).tolist()


@integration
def test_hysplit_multipoint_release_points_follow_control_order(tmp_path, met_dir):
    """
    Multipoint particles are assigned to explicit points in CONTROL order.

    This test characterizes the compiled HYSPLIT binary directly rather than
    PYSTILT's later ``xhgt`` reconstruction. It uses a divisible particle count
    so each explicit release point should receive the same-size contiguous
    ``particle`` block.
    """

    receptor = MultiPointReceptor(
        time=dt.datetime(2021, 1, 15, 6, 0),
        longitudes=[-112.0, -111.8, -111.6],
        latitudes=[40.5, 40.5, 40.5],
        altitudes=[100.0, 500.0, 900.0],
    )
    params = HysplitConfig(
        n_hours=-1,
        numpar=12,
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met = Met("hrrr", make_met_config(met_dir, file_tres="6h"))
    result = _raw_particles(
        receptor, params, met, Path(tmp_path) / "hysplit_assignment", timeout=120
    )

    release_rows = _release_time_rows(result)
    assert release_rows["particle"].tolist() == list(range(1, params.numpar + 1))

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
    expected_heights = list(receptor.altitudes)
    for block, expected in zip(grouped_heights, expected_heights, strict=False):
        assert abs(float(block.mean()) - expected) < 100.0


@integration
def test_hysplit_multipoint_release_points_follow_control_order_nondivisible(
    tmp_path, met_dir
):
    """Nondivisible particle counts still use contiguous blocks in point order."""

    receptor = MultiPointReceptor(
        time=dt.datetime(2021, 1, 15, 6, 0),
        longitudes=[-112.0, -111.8, -111.6],
        latitudes=[40.5, 40.5, 40.5],
        altitudes=[100.0, 500.0, 900.0],
    )
    params = HysplitConfig(
        n_hours=-1,
        numpar=10,
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met = Met("hrrr", make_met_config(met_dir, file_tres="6h"))
    result = _raw_particles(
        receptor,
        params,
        met,
        Path(tmp_path) / "hysplit_assignment_nondivisible",
        timeout=120,
    )

    release_rows = _release_time_rows(result)
    assert release_rows["particle"].tolist() == list(range(1, params.numpar + 1))
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

    receptor = ColumnReceptor(
        time=dt.datetime(2021, 1, 15, 6, 0),
        longitude=-112.0,
        latitude=40.5,
        bottom=5.0,
        top=1000.0,
    )
    params = HysplitConfig(
        n_hours=-1,
        numpar=12,
        # Seed the release so the column heights are reproducible run-to-run
        # (matches the fidelity reference); otherwise the stochastic draw flaked.
        krand=2,
        seed=42,
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    met = Met("hrrr", make_met_config(met_dir, file_tres="6h"))
    result = _raw_particles(
        receptor, params, met, Path(tmp_path) / "hysplit_column_assignment", timeout=120
    )

    release_rows = _release_time_rows(result)
    assert release_rows["particle"].tolist() == list(range(1, params.numpar + 1))
    assert (
        release_rows["zagl"].between(receptor.bottom - 50.0, receptor.top + 50.0).all()
    )
    heights = release_rows["zagl"].to_numpy(dtype=float)
    assert len(np.unique(heights)) == params.numpar
    span = receptor.top - receptor.bottom
    sorted_heights = np.sort(heights)
    central_band = heights[
        (heights >= receptor.bottom + 0.25 * span)
        & (heights <= receptor.top - 0.25 * span)
    ]

    # A true column release should populate the interior of the requested
    # vertical span rather than splitting into two endpoint-heavy clusters. With
    # the seeded release the heights are reproducible, so these are stable checks
    # rather than coin flips. The max-gap bound is the primary guard (chunking
    # leaves a large empty interior); the central-band count is a secondary floor
    # — kept at ``numpar // 3`` because ``numpar // 2`` sat at the mean of a
    # spanning draw and flaked (e.g. 5 of 12) before the release was seeded.
    assert len(central_band) >= params.numpar // 3
    assert float(np.max(np.diff(sorted_heights))) < 0.35 * span


@integration
def test_close_spaced_slant_release_heights_are_recovered(tmp_path, met_dir):
    """
    The case that used to fail silently.

    Ten release points 173 m apart (a 30-degree viewing zenith over a 3 km
    column) climbing 300 m per level. The bundled HYSPLIT build writes no t=0
    row, and in the first minute the wind carries particles hundreds of
    metres, past several neighbouring release points. Matching on horizontal
    position put the assigned release height off by ~240 m RMS; matching on
    height recovers it to within the vertical drift.
    """

    n_levels = 10
    altitudes = np.linspace(300.0, 3000.0, n_levels)
    metres_per_deg_lon = 111_320.0 * np.cos(np.radians(40.766))
    receptor = MultiPointReceptor(
        time=dt.datetime(2021, 1, 15, 6, 0),
        longitudes=-111.8479 + np.arange(n_levels) * 173.0 / metres_per_deg_lon,
        latitudes=np.full(n_levels, 40.766),
        altitudes=altitudes,
    )
    params = HysplitConfig(n_hours=-1, numpar=200, hnf_plume=False)
    met = Met("hrrr", make_met_config(met_dir, file_tres="6h"))
    particles = _raw_particles(
        receptor, params, met, Path(tmp_path) / "close_slant", timeout=300
    )
    data = finished(particles, receptor, params)

    release = _release_time_rows(data).drop_duplicates("particle")
    # Each group's actual height should sit at the altitude it was assigned.
    errors = release.groupby("xhgt")["zagl"].mean() - sorted(set(release["xhgt"]))
    assert float(np.sqrt((errors**2).mean())) < 60.0
    # and every level received particles
    assert set(release["xhgt"]) == set(altitudes.tolist())
