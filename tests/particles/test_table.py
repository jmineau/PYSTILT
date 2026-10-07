"""Tests for stilt.particles.table: particle files, release heights, the near-field correction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from stilt.identity import run_settings, transport_from_settings
from stilt.meteorology import MetConfig
from stilt.particles import (
    correct_near_field,
    particles_metadata,
    read_particles,
    write_particles,
)
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig

from ..fixtures.particles import finished, point_at

#: A receptor that gives no release height of its own: a column.
_NO_POINT = ColumnReceptor(
    time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, bottom=0, top=1000
)


def _particles_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60, -120],
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
    time: float = -1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [time] * len(indices),
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


def _params(tmp_path, hnf_plume=False) -> HysplitConfig:
    return HysplitConfig(
        n_hours=-24,
        numpar=10,
        hnf_plume=hnf_plume,
    )


def _settings(params: HysplitConfig) -> dict:
    """The run settings a particle file records, for *params* and a test met."""
    met = MetConfig(file_format="%Y%m%d_%H", file_tres="1h")
    return run_settings(params, met, ModelInfo(name="hysplit", version="v5.1.0"), None)


def test_parquet_roundtrip_preserves_naive_utc_from_tz_aware_receptor(tmp_path):
    """
    A tz-aware receptor time must normalize to naive UTC and stay naive
    through the trajectory parquet round-trip so the receptor/trajectory/
    footprint time axes align without pandas raising on mixed tz comparisons."""
    aware_receptor = PointReceptor(
        time=pd.Timestamp("2023-01-01 12:00:00+00:00"),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    # Receptor normalizes tz-aware input to naive UTC.
    assert aware_receptor.time.tzinfo is None

    traj = finished(
        _particles_basic(), aware_receptor, _params(tmp_path, hnf_plume=False)
    )

    path = tmp_path / "traj.parquet"
    params = _params(tmp_path, hnf_plume=False)
    write_particles(path, traj, aware_receptor, _settings(params), [Path("/tmp/met1")])
    loaded = read_particles(path)

    receptor = particles_metadata(path).receptor
    assert receptor.time.tzinfo is None
    assert receptor.time == aware_receptor.time
    assert loaded["datetime"].dt.tz is None
    expected = pd.Timestamp("2023-01-01 12:00") + pd.to_timedelta(
        traj["time"].to_numpy(), unit="min"
    )
    np.testing.assert_array_equal(loaded["datetime"].to_numpy(), expected.to_numpy())


def test_write_and_read_particles_round_trip(point_receptor, tmp_path):
    """A particle file holds everything needed to read it back."""
    params = _params(tmp_path, hnf_plume=False)
    traj = finished(_particles_basic(), point_receptor, params)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_receptor, _settings(params), [Path("/tmp/met1")])

    loaded = read_particles(path)
    assert len(loaded) == 2
    assert "receptor" not in loaded.columns
    pd.testing.assert_frame_equal(loaded[traj.columns], traj, check_dtype=False)
    meta = particles_metadata(path)
    assert meta.receptor == point_receptor
    assert meta.settings == _settings(params)
    # Rebuilt from the record: what changes a particle, as the run recorded it.
    assert transport_from_settings(meta.settings).settings() == params.settings()
    assert meta.met_files == [Path("/tmp/met1")]


def test_a_file_without_settings_is_refused_for_its_metadata(point_receptor, tmp_path):
    """A particle file written before files recorded settings still reads, but has no metadata."""
    params = _params(tmp_path, hnf_plume=False)
    traj = finished(_particles_basic(), point_receptor, params)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_receptor, _settings(params), [])
    table = pq.ParquetFile(path).read()
    meta = dict(table.schema.metadata)
    del meta[b"stilt:settings"]
    pq.write_table(table.replace_schema_metadata(meta), path)

    assert len(read_particles(path)) == len(traj)
    with pytest.raises(ValueError, match="records no run settings"):
        particles_metadata(path)


def test_write_particles_is_atomic_on_failure(point_receptor, tmp_path, monkeypatch):
    params = _params(tmp_path, hnf_plume=False)
    traj = finished(_particles_basic(), point_receptor, params)
    path = tmp_path / "traj.parquet"
    tmp = path.with_suffix(".parquet.tmp")

    def _broken_write(table, write_path, **kwargs):
        del table, kwargs
        Path(write_path).write_bytes(b"partial parquet")
        raise RuntimeError("write failed")

    monkeypatch.setattr("stilt.particles.table.pq.write_table", _broken_write)

    with pytest.raises(RuntimeError, match="write failed"):
        write_particles(path, traj, point_receptor, _settings(params), [])

    assert not path.exists()
    assert not tmp.exists()


def test_correct_near_field_requires_columns():
    p = pd.DataFrame({"time": [-60], "particle": [1], "foot": [1e-5]})
    with pytest.raises(ValueError, match="needs the particle columns"):
        correct_near_field(p, point_at(5.0), 0.5)


def test_correct_near_field_adds_reference_column():
    out = correct_near_field(
        _particles_basic().drop(columns=["zagl"]).assign(xhgt=[5.0, 5.0]),
        _NO_POINT,
        0.5,
    )
    assert "foot_no_hnf_dilution" in out.columns
    assert out["foot_no_hnf_dilution"].iloc[0] == pytest.approx(1e-5)


def test_correct_near_field_grows_outward_from_release_when_forward():
    """A forward run accumulates sigma from the release point, not the far end."""
    backward = _particles_basic().drop(columns=["zagl"]).assign(xhgt=[5.0, 5.0])
    forward = backward.assign(time=-backward["time"])

    back_out = correct_near_field(backward, _NO_POINT, 0.5)
    fwd_out = correct_near_field(forward, _NO_POINT, 0.5)

    # sigma depends on |time| only, so mirroring the track must not change foot
    assert fwd_out["foot"].to_numpy() == pytest.approx(back_out["foot"].to_numpy())
    # foot scales as 1 / plume, so it shrinks as the plume grows away from release
    assert fwd_out["foot"].iloc[0] > fwd_out["foot"].iloc[1]


def test_finished_particles_column_receptor_assigns_xhgt(column_receptor, tmp_path):
    particles = _particles_basic().assign(particle=[1, 2])
    traj = finished(particles, column_receptor, _params(tmp_path, hnf_plume=False))
    assert "xhgt" in traj.columns
    assert traj["xhgt"].tolist() == pytest.approx([16.25, 38.75])


def test_finished_particles_column_receptor_spans_column_monotonically(tmp_path):
    receptor = ColumnReceptor(
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
    traj = finished(
        particles, receptor, HysplitConfig(n_hours=-24, numpar=12, hnf_plume=False)
    )

    expected = [
        ((i - 0.5) * (receptor.top - receptor.bottom) / 12) + receptor.bottom
        for i in range(1, 13)
    ]
    assert traj["xhgt"].tolist() == pytest.approx(expected)
    assert traj["xhgt"].is_monotonic_increasing


def test_finished_particles_multipoint_receptor_assigns_xhgt_from_release_locations(
    tmp_path,
):
    receptor = MultiPointReceptor(
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
    traj = finished(
        particles, receptor, HysplitConfig(n_hours=-24, numpar=12, hnf_plume=False)
    )

    assert "xhgt" in traj.columns
    assert traj["xhgt"].tolist() == pytest.approx(
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
    receptor = MultiPointReceptor(
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
    traj = finished(
        particles, receptor, HysplitConfig(n_hours=-24, numpar=10, hnf_plume=False)
    )

    assert traj["xhgt"].tolist() == pytest.approx(
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


def _slant(spacing_deg: float = 0.002, n: int = 4, **kwargs) -> MultiPointReceptor:
    """A slant-like receptor: points ~170 m apart, climbing 300 m per level."""
    return MultiPointReceptor(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85 + i * spacing_deg for i in range(n)],
        latitudes=[40.77] * n,
        altitudes=[300.0 * (i + 1) for i in range(n)],
        **kwargs,
    )


def _finish(particles, receptor):
    config = HysplitConfig(n_hours=-24, numpar=len(particles), hnf_plume=False)
    return finished(particles, receptor, config)


def test_multipoint_close_points_are_matched_on_height_not_position():
    receptor = _slant()
    # Every particle has been blown ~600 m east (0.007 deg), well past the
    # neighbouring release points, but has barely moved vertically.
    true_level = [0, 0, 1, 1, 2, 2, 3, 3]
    particles = _particles_release_rows(
        indices=list(range(1, 9)),
        longs=[receptor.longitudes[k] + 0.007 for k in true_level],
        lats=[40.77] * 8,
        zagl=[
            receptor.altitudes[k] + dz
            for k, dz in zip(true_level, [-12, 9, 15, -6, 4, -18, 11, -3], strict=True)
        ],
    )
    data = _finish(particles, receptor)
    assert data["xhgt"].tolist() == [
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
    receptor = _slant()
    lons = np.asarray(receptor.longitudes)
    drifted = lons[:4] + 0.007
    nearest = np.argmin(np.abs(drifted[:, None] - lons[None, :]), axis=1)
    assert nearest.tolist() != [0, 1, 2, 3]


def test_multipoint_release_time_rows_are_used_when_present():
    receptor = _slant()
    true_level = [0, 1, 2, 3]
    t0 = _particles_release_rows(
        indices=[1, 2, 3, 4],
        longs=[receptor.longitudes[k] for k in true_level],
        lats=[40.77] * 4,
        zagl=[5000.0] * 4,  # deliberately useless: position must decide
        time=0,
    )
    later = _particles_release_rows(
        indices=[1, 2, 3, 4],
        longs=[receptor.longitudes[k] + 0.007 for k in true_level],
        lats=[40.77] * 4,
        zagl=[5000.0] * 4,
        time=-1,
    )
    data = _finish(pd.concat([later, t0], ignore_index=True), receptor)
    by_particle = data.drop_duplicates("particle").set_index("particle")["xhgt"]
    assert by_particle.loc[[1, 2, 3, 4]].tolist() == [300.0, 600.0, 900.0, 1200.0]


def test_multipoint_xhgt_is_constant_along_each_trajectory():
    receptor = _slant(n=2)
    rows = [
        _particles_release_rows(
            [1, 2],
            list(receptor.longitudes),
            [40.77] * 2,
            zagl=[310.0 + 40 * step, 590.0 - 40 * step],
            time=-(step + 1),
        )
        for step in range(3)
    ]
    data = _finish(pd.concat(rows, ignore_index=True), receptor)
    assert (data.groupby("particle")["xhgt"].nunique() == 1).all()
    assert data.drop_duplicates("particle").set_index("particle")["xhgt"].to_dict() == {
        1: 300.0,
        2: 600.0,
    }


def test_multipoint_msl_receptor_matches_on_height_above_sea_level():
    receptor = MultiPointReceptor(
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
    data = _finish(particles, receptor)
    assert data["xhgt"].tolist() == [2400.0, 1800.0]


def test_multipoint_same_altitude_close_points_warn():
    receptor = MultiPointReceptor(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85, -111.848],  # ~170 m apart
        latitudes=[40.77, 40.77],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.85, -111.848], [40.77] * 2, zagl=[500.0, 500.0]
    )
    with pytest.warns(UserWarning, match="cannot be reliably matched"):
        _finish(particles, receptor)


def test_multipoint_same_altitude_wide_points_do_not_warn(recwarn):
    receptor = MultiPointReceptor(
        time="2023-01-01 12:00:00",
        longitudes=[-112.0, -111.8],  # ~17 km apart
        latitudes=[40.5, 40.5],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.995, -111.795], [40.5] * 2, zagl=[500.0, 500.0]
    )
    data = _finish(particles, receptor)
    assert data["xhgt"].tolist() == [500.0, 500.0]
    assert not [w for w in recwarn if "reliably matched" in str(w.message)]


def test_multipoint_release_time_rows_silence_the_warning(recwarn):
    receptor = MultiPointReceptor(
        time="2023-01-01 12:00:00",
        longitudes=[-111.85, -111.848],
        latitudes=[40.77, 40.77],
        altitudes=[500.0, 500.0],
    )
    particles = _particles_release_rows(
        [1, 2], [-111.85, -111.848], [40.77] * 2, zagl=[500.0, 500.0], time=0
    )
    _finish(particles, receptor)
    assert not [w for w in recwarn if "reliably matched" in str(w.message)]


def test_finished_particles_with_hnf_plume(point_receptor, tmp_path):
    """hnf_plume=True runs plume-dilution correction and adds reference column."""
    traj = finished(
        _particles_basic(), point_receptor, _params(tmp_path, hnf_plume=True)
    )
    assert "foot_no_hnf_dilution" in traj.columns


def test_correct_near_field_raises_without_a_release_height():
    """A receptor other than a point, and no xhgt column, raises ValueError."""
    p = _particles_basic()

    with pytest.raises(ValueError, match="release height"):
        correct_near_field(p, _NO_POINT, 0.5)


def test_footprint_calculate_from_trajectory(point_receptor, tmp_path):
    """calc_footprint works directly on a particle table and its receptor."""
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

    traj = finished(
        _particles_basic(), point_receptor, _params(tmp_path, hnf_plume=False)
    )
    grid = Grid(xmin=-115.0, xmax=-110.0, ymin=38.0, ymax=43.0, xres=0.1, yres=0.1)
    # particles are at [-111.9, -112.0] x [40.7, 40.6] - inside the grid
    result = calc_footprint(traj, point_receptor, grid)
    assert float(result.sum()) > 0


def _particles_two_lengths() -> pd.DataFrame:
    """Two particles: particle 1 reaches -120 min, particle 2 only -30 min."""
    return pd.DataFrame(
        {
            "time": [-60, -120, -30],
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


def test_endpoints_returns_far_end_per_particle(point_receptor, tmp_path):
    """
    endpoints() returns one row per particle at its largest-|time| row, with no
    duration filtering: a particle that left the domain early is a real endpoint."""
    particles = finished(_particles_two_lengths(), point_receptor, _params(tmp_path))
    # As read_particles gives them.
    particles["datetime"] = point_receptor.time + pd.to_timedelta(
        particles["time"], unit="min"
    )

    ep = particles.stilt.endpoints().sort_values("particle").reset_index(drop=True)
    assert list(ep.columns) == list(particles.columns)
    # Both particles kept, each at its far end (largest |time|): p1 at -120, p2 at -30.
    assert ep["particle"].tolist() == [1, 2]
    assert ep["time"].tolist() == [-120, -30]
    assert ep.loc[0, ["lon", "lat", "zagl"]].tolist() == [-112.0, 40.6, 20.0]
    assert ep.loc[1, ["lon", "lat", "zagl"]].tolist() == [-111.8, 40.75, 15.0]
    assert ep["datetime"].tolist() == [
        point_receptor.time + pd.Timedelta(minutes=-120),
        point_receptor.time + pd.Timedelta(minutes=-30),
    ]


def test_calc_footprint_regenerates_a_footprint_on_a_new_grid(tmp_path):
    """calc_footprint makes a footprint on any grid from stored particles."""
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

    rng = np.random.default_rng(0)
    n = 20
    particles = pd.DataFrame(
        {
            "time": [-60] * n + [-120] * n,
            "particle": list(range(1, n + 1)) * 2,
            "lon": rng.uniform(-113.9, -113.1, n * 2),
            "lat": rng.uniform(39.1, 39.9, n * 2),
            "zagl": rng.uniform(5, 100, n * 2),
            "foot": rng.uniform(0.0, 1e-3, n * 2),
        }
    )
    receptor = PointReceptor(
        time="2023-01-01 12:00:00", longitude=-113.5, latitude=39.5, altitude=5.0
    )
    traj = finished(
        particles, receptor, HysplitConfig(n_hours=-2, numpar=n, hnf_plume=False)
    )
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    from stilt.transforms import FirstOrderLifetime

    fp = calc_footprint(traj, receptor, grid, name="coarse")
    assert fp.stilt.name == "coarse"
    assert fp.stilt.receptor == receptor
    assert fp.stilt.grid == grid
    assert float(fp.sum()) > 0

    decayed = calc_footprint(
        traj, receptor, grid, transforms=[FirstOrderLifetime(lifetime_hours=0.5)]
    )
    assert float(decayed.sum()) < float(fp.sum())
    assert decayed.stilt.config.transforms == [FirstOrderLifetime(lifetime_hours=0.5)]


def test_stored_settings_this_version_does_not_have_are_dropped(
    point_receptor, tmp_path
):
    """Files written before a setting was removed still load (#44)."""
    params = _params(tmp_path)
    traj = finished(_particles_basic(), point_receptor, params)
    path = tmp_path / "traj.parquet"
    write_particles(path, traj, point_receptor, _settings(params), [])
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
    with pytest.raises(ValueError, match="no 'foot' column"):
        check_particles(table, need=("foot",))
    with pytest.raises(ValueError, match="no 'lon', 'lat' columns"):
        check_particles(table.drop(columns=["lon", "lat"]))


def test_a_footprint_needs_the_particle_columns(point_receptor):
    from stilt.footprint import calc_footprint
    from stilt.spatial import Grid

    grid = Grid(xmin=-113.0, xmax=-111.0, ymin=40.0, ymax=41.0, xres=0.1, yres=0.1)
    particles = pd.DataFrame(
        {"particle": [1], "time": [-1], "lon": [-112.0], "lat": [40.5], "zagl": [5.0]}
    )
    with pytest.raises(ValueError, match="no 'foot' column"):
        calc_footprint(particles, point_receptor, grid)


def test_a_column_release_row_says_which_slab_a_particle_stands_for():
    """With t = 0 rows, xhgt is the centre of the slab the release falls in, not the indx order."""
    from stilt.particles import add_release_heights

    column = ColumnReceptor(
        time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, bottom=0, top=1000
    )
    released = [900.0, 10.0, 600.0, 300.0]  # four slabs of 250 m, out of indx order
    rows = pd.DataFrame(
        {
            "particle": [1, 2, 3, 4] * 2,
            "time": [0] * 4 + [-1] * 4,
            "lon": [-111.85] * 8,
            "lat": [40.77] * 8,
            "zagl": released + [z + 3.0 for z in released],
        }
    )
    with_release = add_release_heights(rows, column)
    by_particle = with_release.drop_duplicates("particle").set_index("particle")["xhgt"]
    assert by_particle.to_dict() == {1: 875.0, 2: 125.0, 3: 625.0, 4: 375.0}

    # Without the release rows, particle indx is in slab indx.
    first_step = add_release_heights(rows[rows.time == -1], column)
    assert first_step["xhgt"].tolist() == [125.0, 375.0, 625.0, 875.0]


def test_hnf_correction_invariants():
    """
    Mathematical invariants of the HNF near-field plume-dilution correction.

    The HNF correction replaces the HYSPLIT-raw foot with a Gaussian near-field
    value (0.02897 / (plume * dens) * samt * 60) when the plume has not yet
    grown to fill the mixing layer.  This can be larger or smaller than the raw
    value.  The invariants that must always hold:

    1. Corrected foot is always positive.
    2. When plume >= pbl_mixing, foot is left unchanged (identity path).
    3. The raw values are preserved in `foot_no_hnf_dilution`.
    """
    rng = np.random.default_rng(77)
    n = 50
    raw_foot = rng.uniform(1e-5, 1e-3, n)
    # Force some particles to have large plume (sigma >> mlht) so the identity
    # path is exercised.  Large sigw + long time → large plume.
    sigw = np.concatenate(
        [
            rng.uniform(0.01, 0.5, n // 2),  # small sigma → near-field path
            rng.uniform(5.0, 20.0, n // 2),  # large sigma → identity path
        ]
    )
    particles = pd.DataFrame(
        {
            "time": np.concatenate(
                [
                    rng.uniform(-0.5, -0.1, n // 2),  # short time → small plume
                    rng.uniform(-6.0, -5.0, n // 2),  # long time → large plume
                ]
            ),
            "particle": [float(i + 1) for i in range(n)],
            "lon": [-112.0] * n,
            "lat": [40.5] * n,
            "zagl": [5.0] * n,
            "foot": raw_foot,
            "mlht": [500.0] * n,  # pbl_mixing = 0.5 * 500 = 250 m
            "dens": [1.2] * n,
            "samt": [60.0] * n,
            "sigw": sigw,
            "tlgr": [100.0] * n,
        }
    )

    result = correct_near_field(particles.copy(), point_at(5.0), 0.5)

    # Invariant 1: corrected foot is always positive.
    assert np.all(result["foot"].values > 0), (
        "HNF-corrected foot values must be positive"
    )

    # Invariant 2: identity path — particles where plume >= pbl_mixing
    # are unchanged.  Reconstruct plume = r_zagl + sigma to find them.
    abs_time_s = np.abs(particles["time"] * 60)
    tlgr = particles["tlgr"]
    sigma = (
        particles["samt"]
        * np.sqrt(2)
        * particles["sigw"]
        * np.sqrt(tlgr * abs_time_s + tlgr**2 * np.exp(-abs_time_s / tlgr) - 1)
    )
    plume = 5.0 + sigma  # r_zagl=5 + sigma (single timestep, cumsum is sigma itself)
    pbl_mixing = 0.5 * particles["mlht"]
    identity_mask = (plume >= pbl_mixing).to_numpy()

    np.testing.assert_allclose(
        result.loc[identity_mask, "foot"].values,
        raw_foot[identity_mask],
        rtol=1e-12,
        err_msg="Particles with plume >= pbl_mixing must have foot unchanged",
    )

    # Invariant 3: raw values preserved in foot_no_hnf_dilution.
    np.testing.assert_array_equal(
        result["foot_no_hnf_dilution"].values,
        raw_foot,
        err_msg="foot_no_hnf_dilution must equal the original foot values",
    )
