"""Tests for stilt.particles.table: particle files and release heights."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stilt.identity import run_settings, transport_from_settings
from stilt.particles import (
    particles_from_table,
    particles_metadata,
    read_particles,
    write_particles,
)
from stilt.particles.table import add_release_heights, needs_height_to_match
from stilt.releases import ColumnRelease, MultiPointRelease, PointRelease
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig

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


def _particles_release_rows(
    indices: list[int],
    longs: list[float],
    lats: list[float],
    zagl: list[float] | None = None,
    age: float = -1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [age] * len(indices),
            "particle": indices,
            "lon": longs,
            "lat": lats,
            "zagl": zagl if zagl is not None else [10.0] * len(indices),
            "foot": [1e-5] * len(indices),
            "dens": [1.2] * len(indices),
            "samt": [1.0] * len(indices),
            "sigw": [0.1] * len(indices),
            "tlgr": [10.0] * len(indices),
            "mlht": [500.0] * len(indices),
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


def test_read_particles_builds_time_when_only_time_is_asked_for(
    point_release, tmp_path
):
    """``time`` is not stored; asking for it reads ``age`` to build it."""
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_release)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_release, _settings(params), [])
    loaded = read_particles(path, columns=["time"])
    expected = pd.Timestamp(point_release.time) + pd.to_timedelta(
        traj["age"].to_numpy(), unit="min"
    )
    np.testing.assert_array_equal(loaded["time"].to_numpy(), expected.to_numpy())


def _file_table(release_id: str, ages: list[int]) -> pa.Table:
    """One release's rows as a particle file stores them, with the ``date`` folder column the output adds."""
    n = len(ages)
    return pa.table(
        {
            "release": pa.DictionaryArray.from_arrays(
                pa.array(np.zeros(n, dtype=np.int32)), pa.array([release_id])
            ),
            "particle": pa.array(np.arange(1, n + 1, dtype=np.int32)),
            "age": pa.array(ages, pa.int32()),
            "lon": pa.array(np.linspace(-112.0, -111.0, n)),
            "date": pa.array([dt.date(2023, 1, 1)] * n, pa.date32()),
        }
    )


_FIRST, _SECOND = "202301011200_-111.85_40.77_5", "202301011300_-111.85_40.77_5"


def _two_files() -> pa.Table:
    """Two releases' files read together: two dictionaries of one entry each, as the output reads them."""
    return pa.concat_tables(
        [_file_table(_SECOND, [-60, -120, -180]), _file_table(_FIRST, [-30, -90])]
    )


def test_particles_from_table_gives_every_row_its_release_and_time():
    got = particles_from_table(_two_files())

    assert list(got.columns) == ["release", "particle", "age", "lon", "time"]
    assert got["release"].tolist() == [_SECOND] * 3 + [_FIRST] * 2
    assert not isinstance(got["release"].dtype, pd.CategoricalDtype)
    assert all(isinstance(r, str) for r in got["release"].unique())
    assert got["age"].dtype == got["particle"].dtype == "float64"
    start = {_FIRST: pd.Timestamp("2023-01-01 12:00"), _SECOND: "2023-01-01 13:00"}
    want = [
        pd.Timestamp(start[r]) + pd.Timedelta(minutes=a)
        for r, a in zip(got["release"], got["age"], strict=True)
    ]
    assert got["time"].dtype.kind == "M"  # naive UTC
    assert got["time"].tolist() == want
    assert got["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")


def test_particles_from_table_reads_a_release_column_of_plain_strings():
    table = _two_files()
    plain = table.set_column(0, "release", table["release"].cast(pa.string()))
    assert plain.schema.field("release").type == pa.string()
    pd.testing.assert_frame_equal(
        particles_from_table(plain), particles_from_table(table)
    )


def test_particles_from_table_without_time_or_age():
    table = _two_files()
    assert "time" not in particles_from_table(table, time=False).columns
    no_age = table.drop_columns("age")
    got = particles_from_table(no_age)
    assert list(got.columns) == ["release", "particle", "lon"]
    assert got["release"].tolist() == [_SECOND] * 3 + [_FIRST] * 2


def test_particles_from_table_of_no_rows_is_empty():
    got = particles_from_table(_file_table(_FIRST, []))
    assert got.empty
    assert "date" not in got.columns


def test_particles_from_table_rejects_an_id_it_cannot_time():
    with pytest.raises(ValueError, match="is not of the form"):
        particles_from_table(_file_table("not_a_release_id", [-60]))


def test_write_and_read_particles_round_trip(point_release, tmp_path):
    """A particle file holds everything needed to read it back."""
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_release)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_release, _settings(params), [Path("/tmp/met1")])

    loaded = read_particles(path)
    assert len(loaded) == 2
    assert "release" not in loaded.columns
    pd.testing.assert_frame_equal(loaded[traj.columns], traj, check_dtype=False)
    meta = particles_metadata(path)
    assert meta.release == point_release
    assert meta.settings == _settings(params)
    # Rebuilt from the record: what changes a particle, as the run recorded it.
    assert transport_from_settings(meta.settings).settings() == params.settings()
    assert meta.met_files == [Path("/tmp/met1")]


def test_a_file_without_settings_is_refused_for_its_metadata(point_release, tmp_path):
    """A particle file written before files recorded settings still reads, but has no metadata."""
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_release)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_release, _settings(params), [])
    table = pq.ParquetFile(path).read()
    meta = dict(table.schema.metadata)
    del meta[b"stilt:settings"]
    pq.write_table(table.replace_schema_metadata(meta), path)

    assert len(read_particles(path)) == len(traj)
    with pytest.raises(ValueError, match="records no run settings"):
        particles_metadata(path)


def test_write_particles_is_atomic_on_failure(point_release, tmp_path, monkeypatch):
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_release)
    path = tmp_path / "traj.parquet"
    tmp = path.with_suffix(".parquet.tmp")

    def _broken_write(table, write_path, **kwargs):
        del table, kwargs
        Path(write_path).write_bytes(b"partial parquet")
        raise RuntimeError("write failed")

    monkeypatch.setattr("stilt.particles.table.pq.write_table", _broken_write)

    with pytest.raises(RuntimeError, match="write failed"):
        write_particles(path, traj, point_release, _settings(params), [])

    assert not path.exists()
    assert not tmp.exists()


def test_finished_particles_column_release_assigns_release_height(
    column_release, tmp_path
):
    particles = _particles_basic().assign(particle=[1, 2])
    traj = finished(particles, column_release)
    assert "release_height" in traj.columns
    assert traj["release_height"].tolist() == pytest.approx([16.25, 38.75])


def test_finished_particles_column_release_spans_column_monotonically(tmp_path):
    release = ColumnRelease(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        bottom=5.0,
        top=1000.0,
    )
    particles = _particles_release_rows(
        indices=list(range(1, 13)),
        longs=[-111.85] * 12,
        lats=[40.77] * 12,
    )
    traj = finished(particles, release)

    expected = [
        ((i - 0.5) * (release.top - release.bottom) / 12) + release.bottom
        for i in range(1, 13)
    ]
    assert traj["release_height"].tolist() == pytest.approx(expected)
    assert traj["release_height"].is_monotonic_increasing


def test_finished_particles_multipoint_release_assigns_release_height_from_release_locations(
    tmp_path,
):
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-112.0, -111.8, -111.6],
        latitudes=[40.5, 40.5, 40.5],
        altitudes=[100.0, 500.0, 900.0],
    )
    particles = _particles_release_rows(
        indices=list(range(1, 13)),
        longs=[
            -112.0012,
            -112.0012,
            -112.0012,
            -112.0011,
            -111.8048,
            -111.8052,
            -111.8044,
            -111.8049,
            -111.6142,
            -111.6136,
            -111.6140,
            -111.6138,
        ],
        lats=[
            40.4981,
            40.4981,
            40.4981,
            40.4980,
            40.5000,
            40.5009,
            40.5002,
            40.4996,
            40.4997,
            40.4994,
            40.4995,
            40.4996,
        ],
        zagl=[
            98.0,
            104.0,
            91.0,
            110.0,
            512.0,
            489.0,
            503.0,
            497.0,
            880.0,
            905.0,
            921.0,
            893.0,
        ],
    )
    traj = finished(particles, release)

    assert "release_height" in traj.columns
    assert traj["release_height"].tolist() == pytest.approx(
        [
            100.0,
            100.0,
            100.0,
            100.0,
            500.0,
            500.0,
            500.0,
            500.0,
            900.0,
            900.0,
            900.0,
            900.0,
        ]
    )


def test_finished_particles_multipoint_nondivisible_particle_blocks_follow_release_locations(
    tmp_path,
):
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-112.0, -111.8, -111.6],
        latitudes=[40.5, 40.5, 40.5],
        altitudes=[100.0, 500.0, 900.0],
    )
    particles = _particles_release_rows(
        indices=list(range(1, 11)),
        longs=[
            -112.0012,
            -112.0012,
            -112.0012,
            -112.0011,
            -111.8048,
            -111.8052,
            -111.8044,
            -111.8049,
            -111.6142,
            -111.6136,
        ],
        lats=[
            40.4981,
            40.4981,
            40.4981,
            40.4980,
            40.5000,
            40.5009,
            40.5002,
            40.4996,
            40.4997,
            40.4994,
        ],
        zagl=[98.0, 104.0, 91.0, 110.0, 512.0, 489.0, 503.0, 497.0, 880.0, 905.0],
    )
    traj = finished(particles, release)

    assert traj["release_height"].tolist() == pytest.approx(
        [100.0, 100.0, 100.0, 100.0, 500.0, 500.0, 500.0, 500.0, 900.0, 900.0]
    )


# ---------------------------------------------------------------------------
# Multipoint release-height recovery
#
# HYSPLIT does not say which starting location a particle came from. Builds
# without release-time rows first report a particle a timestep after release,
# by which point the wind has carried it hundreds of metres -- further than
# the points of a slant column are apart. These tests pin down the three
# cases in ``_multipoint_release_heights``.
# ---------------------------------------------------------------------------


def _slant(spacing_deg: float = 0.002, n: int = 4, **kwargs) -> MultiPointRelease:
    """A slant-like release: points ~170 m apart, climbing 300 m per level."""
    return MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85 + i * spacing_deg for i in range(n)],
        latitudes=[40.77] * n,
        altitudes=[300.0 * (i + 1) for i in range(n)],
        **kwargs,
    )


def _finish(particles, release):
    return finished(particles, release)


def test_multipoint_close_points_are_matched_on_height_not_position():
    release = _slant()
    # Every particle has been blown ~600 m east (0.007 deg), well past the
    # neighbouring release points, but has barely moved vertically.
    true_level = [0, 0, 1, 1, 2, 2, 3, 3]
    particles = _particles_release_rows(
        indices=list(range(1, 9)),
        longs=[release.longitudes[k] + 0.007 for k in true_level],
        lats=[40.77] * 8,
        zagl=[
            release.altitudes[k] + dz
            for k, dz in zip(true_level, [-12, 9, 15, -6, 4, -18, 11, -3], strict=True)
        ],
    )
    data = _finish(particles, release)
    assert data["release_height"].tolist() == [
        300.0,
        300.0,
        600.0,
        600.0,
        900.0,
        900.0,
        1200.0,
        1200.0,
    ]


def test_multipoint_horizontal_matching_would_have_failed_that_case():
    """Guards the premise: nearest-horizontal really does get this wrong."""
    release = _slant()
    lons = np.asarray(release.longitudes)
    drifted = lons[:4] + 0.007
    nearest = np.argmin(np.abs(drifted[:, None] - lons[None, :]), axis=1)
    assert nearest.tolist() != [0, 1, 2, 3]


def test_multipoint_release_time_rows_are_used_when_present():
    release = _slant()
    true_level = [0, 1, 2, 3]
    t0 = _particles_release_rows(
        indices=[1, 2, 3, 4],
        longs=[release.longitudes[k] for k in true_level],
        lats=[40.77] * 4,
        zagl=[5000.0] * 4,  # deliberately useless: position must decide
        age=0,
    )
    later = _particles_release_rows(
        indices=[1, 2, 3, 4],
        longs=[release.longitudes[k] + 0.007 for k in true_level],
        lats=[40.77] * 4,
        zagl=[5000.0] * 4,
        age=-1,
    )
    data = _finish(pd.concat([later, t0], ignore_index=True), release)
    by_particle = data.drop_duplicates("particle").set_index("particle")[
        "release_height"
    ]
    assert by_particle.loc[[1, 2, 3, 4]].tolist() == [300.0, 600.0, 900.0, 1200.0]


def test_multipoint_release_height_is_constant_along_each_trajectory():
    release = _slant(n=2)
    rows = [
        _particles_release_rows(
            [1, 2],
            list(release.longitudes),
            [40.77] * 2,
            zagl=[310.0 + 40 * step, 590.0 - 40 * step],
            age=-(step + 1),
        )
        for step in range(3)
    ]
    data = _finish(pd.concat(rows, ignore_index=True), release)
    assert (data.groupby("particle")["release_height"].nunique() == 1).all()
    assert data.drop_duplicates("particle").set_index("particle")[
        "release_height"
    ].to_dict() == {
        1: 300.0,
        2: 600.0,
    }


def test_multipoint_msl_release_matches_on_height_above_sea_level():
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85, -111.848],
        latitudes=[40.77, 40.77],
        altitudes=[1800.0, 2400.0],
        altitude_ref="msl",
    )
    particles = _particles_release_rows(
        [1, 2], [-111.843, -111.841], [40.77] * 2, zagl=[905.0, 292.0]
    )
    particles["zsfc"] = [1500.0, 1500.0]  # 905+1500=2405, 292+1500=1792
    data = _finish(particles, release)
    assert data["release_height"].tolist() == [2400.0, 1800.0]


def test_multipoint_same_altitude_close_points_warn():
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85, -111.848],  # ~170 m apart
        latitudes=[40.77, 40.77],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.85, -111.848], [40.77] * 2, zagl=[500.0, 500.0]
    )
    with pytest.warns(UserWarning, match="cannot be reliably matched"):
        _finish(particles, release)


def test_multipoint_same_altitude_wide_points_do_not_warn(recwarn):
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-112.0, -111.8],  # ~17 km apart
        latitudes=[40.5, 40.5],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.995, -111.795], [40.5] * 2, zagl=[500.0, 500.0]
    )
    data = _finish(particles, release)
    assert data["release_height"].tolist() == [500.0, 500.0]
    assert not [w for w in recwarn if "reliably matched" in str(w.message)]


def test_multipoint_release_time_rows_silence_the_warning(recwarn):
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85, -111.848],
        latitudes=[40.77, 40.77],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.85, -111.848], [40.77] * 2, zagl=[500.0, 500.0], age=0
    )
    _finish(particles, release)
    assert not [w for w in recwarn if "reliably matched" in str(w.message)]


def test_finished_particles_keep_the_uncorrected_foot(point_release):
    """A run's particles are stored as the model left them; the footprint applies the correction."""
    traj = finished(_particles_basic(), point_release)
    assert "foot_no_hnf_dilution" not in traj.columns
    assert traj["foot"].tolist() == _particles_basic()["foot"].tolist()


def test_footprint_calculate_from_trajectory(point_release, tmp_path):
    """calc_footprint works directly on a particle table and its release."""
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

    traj = finished(_particles_basic(), point_release)
    grid = Grid(xmin=-115.0, xmax=-110.0, ymin=38.0, ymax=43.0, xres=0.1, yres=0.1)
    # particles are at [-111.9, -112.0] x [40.7, 40.6] - inside the grid
    result = calc_footprint(traj, point_release, grid)
    assert float(result.sum()) > 0


def _particles_two_lengths() -> pd.DataFrame:
    """Two particles: particle 1 reaches -120 min, particle 2 only -30 min."""
    return pd.DataFrame(
        {
            "age": [-60, -120, -30],
            "particle": [1, 1, 2],
            "lon": [-111.9, -112.0, -111.8],
            "lat": [40.7, 40.6, 40.75],
            "zagl": [10.0, 20.0, 15.0],
            "foot": [1e-5, 2e-5, 1e-5],
            "dens": [1.2, 1.2, 1.2],
            "samt": [1.0, 1.0, 1.0],
            "sigw": [0.1, 0.1, 0.1],
            "tlgr": [10.0, 10.0, 10.0],
            "mlht": [500.0, 500.0, 500.0],
        }
    )


def test_endpoints_returns_far_end_per_particle(point_release, tmp_path):
    """
    ``endpoints()`` returns each particle's row with the largest ``|time|``.

    There is no duration filtering: a particle that left the domain early is
    a real endpoint.
    """
    particles = finished(_particles_two_lengths(), point_release)
    # As read_particles gives them.
    particles["time"] = point_release.time + pd.to_timedelta(
        particles["age"], unit="min"
    )

    ep = particles.stilt.endpoints().sort_values("particle").reset_index(drop=True)
    assert list(ep.columns) == list(particles.columns)
    # Both particles kept, each at its far end (largest |time|): p1 at -120, p2 at -30.
    assert ep["particle"].tolist() == [1, 2]
    assert ep["age"].tolist() == [-120, -30]
    assert ep.loc[0, ["lon", "lat", "zagl"]].tolist() == [-112.0, 40.6, 20.0]
    assert ep.loc[1, ["lon", "lat", "zagl"]].tolist() == [-111.8, 40.75, 15.0]
    assert ep["time"].tolist() == [
        point_release.time + pd.Timedelta(minutes=-120),
        point_release.time + pd.Timedelta(minutes=-30),
    ]


def test_calc_footprint_regenerates_a_footprint_on_a_new_grid(tmp_path):
    """calc_footprint makes a footprint on any grid from stored particles."""
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

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


def test_stored_settings_this_version_does_not_have_are_dropped(
    point_release, tmp_path
):
    """Files written before a setting was removed still load (#44)."""
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_release)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_release, _settings(params), [])
    table = pq.ParquetFile(path).read()
    meta = dict(table.schema.metadata)
    stored = json.loads(meta[b"stilt:settings"])
    meta[b"stilt:settings"] = json.dumps({**stored, "gone": 1}).encode()
    meta[b"stilt:is_error"] = b"true"  # written by earlier versions; ignored
    pq.write_table(table.replace_schema_metadata(meta), path)

    rebuilt = transport_from_settings(particles_metadata(path).settings)
    assert rebuilt.settings() == params.settings()


def test_check_particles_names_the_missing_columns():
    from stilt.particles import PARTICLE_SCHEMA, check_particles

    table = pd.DataFrame({name: [1.0] for name in PARTICLE_SCHEMA.names})
    check_particles(table)
    with pytest.raises(ValueError, match="no 'foot' column.*footprint_particles"):
        check_particles(table, need=("foot",))
    with pytest.raises(ValueError, match="no 'lon', 'lat' columns"):
        check_particles(table.drop(columns=["lon", "lat"]))


def test_a_footprint_needs_the_particle_columns(point_release):
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

    grid = Grid(xmin=-113.0, xmax=-111.0, ymin=40.0, ymax=41.0, xres=0.1, yres=0.1)
    particles = pd.DataFrame(
        {"particle": [1], "age": [-1], "lon": [-112.0], "lat": [40.5], "zagl": [5.0]}
    )
    with pytest.raises(ValueError, match="no 'foot' column"):
        calc_footprint(particles, point_release, grid)


def test_a_column_release_row_says_which_slab_a_particle_stands_for():
    """With t = 0 rows, release_height is the centre of the slab the release falls in, not the indx order."""
    from stilt.particles import add_release_heights

    column = ColumnRelease(
        time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, bottom=0, top=1000
    )
    released = [900.0, 10.0, 600.0, 300.0]  # four slabs of 250 m, out of indx order
    rows = pd.DataFrame(
        {
            "particle": [1, 2, 3, 4] * 2,
            "age": [0] * 4 + [-1] * 4,
            "lon": [-111.85] * 8,
            "lat": [40.77] * 8,
            "zagl": released + [z + 3.0 for z in released],
        }
    )
    with_release = add_release_heights(rows, column)
    by_particle = with_release.drop_duplicates("particle").set_index("particle")[
        "release_height"
    ]
    assert by_particle.to_dict() == {1: 875.0, 2: 125.0, 3: 625.0, 4: 375.0}

    # Without the release rows, particle indx is in slab indx.
    first_step = add_release_heights(rows[rows.age == -1], column)
    assert first_step["release_height"].tolist() == [125.0, 375.0, 625.0, 875.0]


@pytest.mark.parametrize(
    ("longitudes", "latitudes", "altitudes", "expected"),
    [
        ([-111.85, -111.848], [40.77, 40.77], [1800.0, 2400.0], True),
        ([-111.85, -111.848], [40.77, 40.77], [2000.0, 2000.0], False),
        ([-111.85, -111.848, -111.846], [40.77] * 3, [1800.0, 1800.0, 2400.0], False),
        ([-111.85, -111.80], [40.77, 40.77], [1800.0, 2400.0], False),
        ([-111.85], [40.77], [1800.0], False),
    ],
)
def test_needs_height_to_match_is_true_for_close_points_at_different_altitudes(
    longitudes, latitudes, altitudes, expected
):
    assert needs_height_to_match(longitudes, latitudes, altitudes) is expected


@pytest.mark.filterwarnings("ignore:MultiPointRelease release points")
@pytest.mark.parametrize(
    ("longitudes", "altitudes"),
    [
        ([-111.85, -111.848], [2000.0, 2000.0]),  # one altitude
        ([-111.85, -111.80], [1800.0, 2400.0]),  # 4 km apart
    ],
)
def test_msl_multipoint_release_heights_do_not_read_zsfc_when_it_is_not_needed(
    longitudes, altitudes
):
    release = MultiPointRelease(
        time="2023-01-01 12:00:00",
        longitudes=longitudes,
        latitudes=[40.77, 40.77],
        altitudes=altitudes,
        altitude_ref="msl",
    )
    assert not needs_height_to_match(longitudes, [40.77, 40.77], altitudes)
    particles = _particles_release_rows(
        [1, 2], [lon + 0.003 for lon in longitudes], [40.77] * 2, zagl=[300.0, 900.0]
    )
    without = add_release_heights(particles, release)["release_height"]
    with_zsfc = add_release_heights(particles.assign(zsfc=1500.0), release)[
        "release_height"
    ]
    assert without.tolist() == with_zsfc.tolist()


#: foot of one minute below a 500 m layer of air of mean density 1 kg/m3.
