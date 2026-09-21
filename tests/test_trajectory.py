"""Tests for trajectory model and plume-dilution helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from stilt.config import STILTParams
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.trajectory import Trajectories, calc_plume_dilution


def _particles_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60, -120],
            "indx": [1, 1],
            "long": [-111.9, -112.0],
            "lati": [40.7, 40.6],
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
            "indx": indices,
            "long": longs,
            "lati": lats,
            "zagl": zagl if zagl is not None else [10.0] * len(indices),
            "foot": [1e-5] * len(indices),
            "dens": [1.2] * len(indices),
            "samt": [1.0] * len(indices),
            "sigw": [0.1] * len(indices),
            "tlgr": [10.0] * len(indices),
            "mlht": [500.0] * len(indices),
        }
    )


def _params(tmp_path, hnf_plume=False) -> STILTParams:
    return STILTParams(
        n_hours=-24,
        numpar=10,
        hnf_plume=hnf_plume,
    )


def test_from_particles_adds_datetime(point_receptor, tmp_path):
    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )

    assert "datetime" in traj.data.columns
    assert pd.api.types.is_datetime64_any_dtype(traj.data["datetime"])


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

    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=aware_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    assert traj.data["datetime"].dt.tz is None

    path = tmp_path / "traj.parquet"
    traj.to_parquet(path)
    loaded = Trajectories.from_parquet(path)

    assert loaded.receptor.time.tzinfo is None
    assert loaded.receptor.time == aware_receptor.time
    assert loaded.data["datetime"].dt.tz is None
    pd.testing.assert_series_equal(
        loaded.data["datetime"].reset_index(drop=True),
        traj.data["datetime"].reset_index(drop=True),
    )


def test_to_from_parquet_roundtrip(point_receptor, tmp_path):
    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
        is_error=True,
    )
    path = tmp_path / "traj.parquet"
    traj.to_parquet(path)

    loaded = Trajectories.from_parquet(path)
    assert len(loaded.data) == 2
    assert loaded.receptor.id == point_receptor.id
    assert loaded.is_error is True


def test_to_parquet_is_atomic_on_failure(point_receptor, tmp_path, monkeypatch):
    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    path = tmp_path / "traj.parquet"
    tmp = path.with_suffix(".parquet.tmp")

    def _broken_write(table, write_path, **kwargs):
        del table, kwargs
        Path(write_path).write_bytes(b"partial parquet")
        raise RuntimeError("write failed")

    monkeypatch.setattr("stilt.trajectory.pq.write_table", _broken_write)

    with pytest.raises(RuntimeError, match="write failed"):
        traj.to_parquet(path)

    assert not path.exists()
    assert not tmp.exists()


def test_to_parquet_falls_back_when_zstd_is_unavailable(
    point_receptor,
    tmp_path,
    monkeypatch,
):
    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    path = tmp_path / "traj.parquet"
    compressions: list[str | None] = []

    def _fake_write(table, write_path, **kwargs):
        del table
        compression = kwargs["compression"]
        compressions.append(compression)
        if compression == "zstd":
            raise pa.ArrowNotImplementedError("codec not supported")
        Path(write_path).write_bytes(b"fallback parquet")

    monkeypatch.setattr("stilt.trajectory.pq.write_table", _fake_write)

    traj.to_parquet(path)

    assert compressions == ["zstd", "snappy"]
    assert path.read_bytes() == b"fallback parquet"


def test_calc_plume_dilution_requires_columns():
    p = pd.DataFrame({"time": [-60], "indx": [1], "foot": [1e-5]})
    with pytest.raises(ValueError, match="hnf_plume requires"):
        calc_plume_dilution(particles=p, r_zagl=5.0, veght=0.5)


def test_calc_plume_dilution_adds_reference_column():
    out = calc_plume_dilution(
        particles=_particles_basic().drop(columns=["zagl"]).assign(xhgt=[5.0, 5.0]),
        r_zagl=None,
        veght=0.5,
    )
    assert "foot_no_hnf_dilution" in out.columns
    assert out["foot_no_hnf_dilution"].iloc[0] == pytest.approx(1e-5)


def test_from_particles_column_receptor_assigns_xhgt(column_receptor, tmp_path):
    particles = _particles_basic().assign(indx=[1, 2])
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=column_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    assert "xhgt" in traj.data.columns
    assert traj.data["xhgt"].tolist() == pytest.approx([16.25, 38.75])


def test_from_particles_column_receptor_spans_column_monotonically(tmp_path):
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
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=receptor,
        params=STILTParams(n_hours=-24, numpar=12, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )

    expected = [
        ((i - 0.5) * (receptor.top - receptor.bottom) / 12) + receptor.bottom
        for i in range(1, 13)
    ]
    assert traj.data["xhgt"].tolist() == pytest.approx(expected)
    assert traj.data["xhgt"].is_monotonic_increasing


def test_from_particles_multipoint_receptor_assigns_xhgt_from_release_locations(
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
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=receptor,
        params=STILTParams(n_hours=-24, numpar=12, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )

    assert "xhgt" in traj.data.columns
    assert traj.data["xhgt"].tolist() == pytest.approx(
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


def test_from_particles_multipoint_nondivisible_particle_blocks_follow_release_locations(
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
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=receptor,
        params=STILTParams(n_hours=-24, numpar=10, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )

    assert traj.data["xhgt"].tolist() == pytest.approx(
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


def _from_particles(particles, receptor):
    return Trajectories.from_particles(
        particles=particles,
        receptor=receptor,
        params=STILTParams(n_hours=-24, numpar=len(particles), hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    ).data


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
    data = _from_particles(particles, receptor)
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
    drifted = np.array([receptor.longitudes[k] + 0.007 for k in [0, 1, 2, 3]])
    nearest = np.argmin(np.abs(drifted[:, None] - receptor.longitudes[None, :]), axis=1)
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
    data = _from_particles(pd.concat([later, t0], ignore_index=True), receptor)
    by_particle = data.drop_duplicates("indx").set_index("indx")["xhgt"]
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
    data = _from_particles(pd.concat(rows, ignore_index=True), receptor)
    assert (data.groupby("indx")["xhgt"].nunique() == 1).all()
    assert data.drop_duplicates("indx").set_index("indx")["xhgt"].to_dict() == {
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
    data = _from_particles(particles, receptor)
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
        _from_particles(particles, receptor)


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
    data = _from_particles(particles, receptor)
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
    _from_particles(particles, receptor)
    assert not [w for w in recwarn if "reliably matched" in str(w.message)]


def test_from_particles_with_hnf_plume(point_receptor, tmp_path):
    """hnf_plume=True runs plume-dilution correction and adds reference column."""
    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=True),
        met_files=[Path("/tmp/met1")],
    )
    assert "foot_no_hnf_dilution" in traj.data.columns


def test_calc_plume_dilution_raises_when_no_xhgt_and_no_rzagl():
    """r_zagl=None with no xhgt column raises ValueError."""
    p = _particles_basic()
    from stilt.trajectory import calc_plume_dilution

    with pytest.raises(ValueError, match="r_zagl must be provided"):
        calc_plume_dilution(particles=p, r_zagl=None, veght=0.5)


def test_footprint_calculate_from_trajectory(point_receptor, tmp_path):
    """Footprint.calculate works directly on Trajectories.data and receptor."""
    from stilt.config import FootprintConfig, Grid
    from stilt.footprint import Footprint

    traj = Trajectories.from_particles(
        particles=_particles_basic(),
        receptor=point_receptor,
        params=_params(tmp_path, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    config = FootprintConfig(
        grid=Grid(xmin=-115.0, xmax=-110.0, ymin=38.0, ymax=43.0, xres=0.1, yres=0.1)
    )
    # particles are at [-111.9, -112.0] x [40.7, 40.6] - inside the grid
    result = Footprint.calculate(traj.data, receptor=traj.receptor, config=config)
    assert result is not None or result is None  # just ensure it doesn't crash


def _particles_two_lengths() -> pd.DataFrame:
    """Two particles: particle 1 reaches -120 min, particle 2 only -30 min."""
    return pd.DataFrame(
        {
            "time": [-60, -120, -30],
            "indx": [1, 1, 2],
            "long": [-111.9, -112.0, -111.8],
            "lati": [40.7, 40.6, 40.75],
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
    traj = Trajectories.from_particles(
        particles=_particles_two_lengths(),
        receptor=point_receptor,
        params=_params(tmp_path),
        met_files=[Path("/tmp/met1")],
    )

    ep = traj.endpoints().sort_values("indx").reset_index(drop=True)
    assert list(ep.columns) == [
        "indx",
        "time",
        "lati",
        "long",
        "zagl",
        "endpoint_age_min",
        "run_time",
    ]
    # Both particles kept, each at its far end (largest |time|): p1 at -120, p2 at -30.
    assert ep["indx"].tolist() == [1, 2]
    assert ep["endpoint_age_min"].tolist() == [-120, -30]
    assert ep.loc[0, ["long", "lati", "zagl"]].tolist() == [-112.0, 40.6, 20.0]
    assert ep.loc[1, ["long", "lati", "zagl"]].tolist() == [-111.8, 40.75, 15.0]
    assert ep["time"].tolist() == [
        point_receptor.time + pd.Timedelta(minutes=-120),
        point_receptor.time + pd.Timedelta(minutes=-30),
    ]
    assert (ep["run_time"] == point_receptor.time).all()


def test_trajectories_footprint_regenerates_on_new_grid(tmp_path):
    """Trajectories.footprint() calculates a footprint on an arbitrary grid."""
    from stilt.config import FootprintConfig, Grid
    from stilt.footprint import Footprint

    rng = np.random.default_rng(0)
    n = 20
    particles = pd.DataFrame(
        {
            "time": [-60] * n + [-120] * n,
            "indx": list(range(1, n + 1)) * 2,
            "long": rng.uniform(-113.9, -113.1, n * 2),
            "lati": rng.uniform(39.1, 39.9, n * 2),
            "zagl": rng.uniform(5, 100, n * 2),
            "foot": rng.uniform(0.0, 1e-3, n * 2),
        }
    )
    receptor = PointReceptor("2023-01-01 12:00:00", -113.5, 39.5, 5.0)
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=receptor,
        params=STILTParams(n_hours=-2, numpar=n, hnf_plume=False),
        met_files=[Path("/tmp/met1")],
    )
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    )
    fp = traj.footprint(config, name="coarse")
    assert isinstance(fp, Footprint)
    assert fp.name == "coarse"
    assert fp.receptor == receptor
    assert fp.config.grid == config.grid
    assert float(fp.data.sum()) > 0
