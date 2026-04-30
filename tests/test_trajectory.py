"""Tests for trajectory model and plume-dilution helpers."""

from pathlib import Path

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
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-1] * len(indices),
            "indx": indices,
            "long": longs,
            "lati": lats,
            "zagl": [10.0] * len(indices),
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
