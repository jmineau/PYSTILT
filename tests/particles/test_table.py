"""Tests for runtraj.particles.table: particle files and release heights."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from runtraj.identity import run_settings
from runtraj.particles import (
    particles_metadata,
    read_particles,
    write_particles,
)
from runtraj.releases import ColumnRelease, PointRelease
from runtraj.transport import ModelInfo
from runtraj.transport.hysplit import HysplitConfig

from ..fixtures.particles import finished

#: A release that gives no release height of its own: a column.
_NO_POINT = ColumnRelease(
    time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, bottom=0, top=1000
)


def _particles_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [-60, -120],
            "particle": [1, 1],
            "lon": [-111.9, -112.0],
            "lat": [40.7, 40.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def _params(tmp_path) -> HysplitConfig:
    return HysplitConfig(n_hours=-24, numpar=10)


def _settings(params: HysplitConfig) -> dict:
    """The run settings a particle file records, for *params* and a test met."""
    met = {"source": "HRRR", "crop": None}
    return run_settings(params, met, ModelInfo(name="hysplit", sha256="0" * 64), None)


def test_parquet_roundtrip_preserves_naive_utc_from_tz_aware_release(tmp_path):
    """
    A tz-aware release time becomes naive UTC and stays so through parquet.

    Then the release, trajectory and footprint time axes align without
    pandas raising on mixed tz comparisons.
    """
    aware_release = PointRelease(
        time=pd.Timestamp("2023-01-01 12:00:00+00:00"),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    # Release normalizes tz-aware input to naive UTC.
    assert aware_release.time.tzinfo is None

    traj = finished(_particles_basic(), aware_release)

    path = tmp_path / "traj.parquet"
    params = _params(tmp_path)
    write_particles(path, traj, aware_release, _settings(params), [Path("/tmp/met1")])
    loaded = read_particles(path)

    release = particles_metadata(path).release
    assert release.time.tzinfo is None
    assert release.time == aware_release.time
    assert loaded["time"].dt.tz is None
    expected = pd.Timestamp("2023-01-01 12:00") + pd.to_timedelta(
        traj["age"].to_numpy(), unit="min"
    )
    np.testing.assert_array_equal(loaded["time"].to_numpy(), expected.to_numpy())


_FIRST, _SECOND = "202301011200_-111.85_40.77_5", "202301011300_-111.85_40.77_5"


# ---------------------------------------------------------------------------
# Multipoint release-height recovery
#
# HYSPLIT does not say which starting location a particle came from. Builds
# without release-time rows first report a particle a timestep after release,
# by which point the wind has carried it hundreds of metres -- further than
# the points of a slant column are apart. These tests pin down the three
# cases in ``_multipoint_release_heights``.
# ---------------------------------------------------------------------------


def _finish(particles, release):
    return finished(particles, release)


def test_finished_particles_keep_the_uncorrected_foot(point_release):
    """A run's particles are stored as the model left them; the footprint applies the correction."""
    traj = finished(_particles_basic(), point_release)
    assert "foot_no_hnf_dilution" not in traj.columns
    assert traj["foot"].tolist() == _particles_basic()["foot"].tolist()


def test_footprint_calculate_from_trajectory(point_release, tmp_path):
    """calc_footprint works directly on a particle table and its release."""
    from runtraj.spatial import Grid

    from stilt.footprint import calc_footprint

    traj = finished(_particles_basic(), point_release)
    grid = Grid(xmin=-115.0, xmax=-110.0, ymin=38.0, ymax=43.0, xres=0.1, yres=0.1)
    # particles are at [-111.9, -112.0] x [40.7, 40.6] - inside the grid
    result = calc_footprint(traj, point_release, grid)
    assert float(result.sum()) > 0


def test_calc_footprint_regenerates_a_footprint_on_a_new_grid(tmp_path):
    """calc_footprint makes a footprint on any grid from stored particles."""
    from runtraj.spatial import Grid

    from stilt.footprint import calc_footprint

    rng = np.random.default_rng(0)
    n = 20
    particles = pd.DataFrame(
        {
            "age": [-60] * n + [-120] * n,
            "particle": list(range(1, n + 1)) * 2,
            "lon": rng.uniform(-113.9, -113.1, n * 2),
            "lat": rng.uniform(39.1, 39.9, n * 2),
            "zagl": rng.uniform(5, 100, n * 2),
            "foot": rng.uniform(0.0, 1e-3, n * 2),
        }
    )
    release = PointRelease(
        time="2023-01-01 12:00:00", longitude=-113.5, latitude=39.5, altitude=5.0
    )
    traj = finished(particles, release)
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    from stilt.transforms import FirstOrderLifetime

    fp = calc_footprint(traj, release, grid, name="coarse")
    assert fp.stilt.name == "coarse"
    assert fp.stilt.release == release
    assert fp.stilt.grid == grid
    assert float(fp.sum()) > 0

    decayed = calc_footprint(
        traj, release, grid, transforms=[FirstOrderLifetime(lifetime_hours=0.5)]
    )
    assert float(decayed.sum()) < float(fp.sum())
    assert decayed.stilt.config.transforms == [FirstOrderLifetime(lifetime_hours=0.5)]


def test_check_particles_names_the_missing_columns():
    from runtraj.particles import PARTICLE_SCHEMA, check_particles

    table = pd.DataFrame({name: [1.0] for name in PARTICLE_SCHEMA.names})
    check_particles(table)
    with pytest.raises(ValueError, match="no 'foot' column"):
        check_particles(table, need=("foot",))
    with pytest.raises(ValueError, match="no 'lon', 'lat' columns"):
        check_particles(table.drop(columns=["lon", "lat"]))


def test_a_footprint_needs_the_particle_columns(point_release):
    from runtraj.spatial import Grid

    from stilt.footprint import calc_footprint

    grid = Grid(xmin=-113.0, xmax=-111.0, ymin=40.0, ymax=41.0, xres=0.1, yres=0.1)
    particles = pd.DataFrame(
        {"particle": [1], "age": [-1], "lon": [-112.0], "lat": [40.5], "zagl": [5.0]}
    )
    with pytest.raises(ValueError, match="no 'foot' column"):
        calc_footprint(particles, point_release, grid)


#: foot of one minute below a 500 m layer of air of mean density 1 kg/m3.
