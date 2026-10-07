"""Tests for footprint files: NetCDF and the geometry a footprint records."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.footprint import read_footprint
from stilt.footprint.config import FootprintConfig
from stilt.receptors import PointReceptor
from stilt.spatial import Grid
from stilt.transforms import AveragingKernel

from ..fixtures.footprints import as_footprint, geometry_footprint, make_footprint


def test_netcdf_roundtrip_preserves_name(tmp_path):
    foot = make_footprint(n_times=1)
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)
    assert loaded.stilt.name == "slv"
    assert loaded.stilt.grid.xres == pytest.approx(0.1)


def test_from_netcdf_forwards_chunks_to_xarray(tmp_path, monkeypatch):
    foot = make_footprint(n_times=1)
    path = tmp_path / "chunked_foot.nc"
    foot.stilt.to_netcdf(path)
    seen_kwargs = {}
    real_open_dataset = xr.open_dataset

    def fake_open_dataset(path_arg, **kwargs):
        seen_kwargs.update(kwargs)
        return real_open_dataset(path_arg)

    monkeypatch.setattr("stilt.footprint.io.xr.open_dataset", fake_open_dataset)

    loaded = read_footprint(path, chunks={"time": 1})

    assert loaded.stilt.name == "slv"
    assert seen_kwargs["chunks"] == {"time": 1}


def test_netcdf_writes_cf_grid_mapping_and_coordinates(tmp_path):
    foot = make_footprint(n_times=1)
    path = tmp_path / "cf_foot.nc"

    foot.stilt.to_netcdf(path)

    ds = xr.open_dataset(path)
    try:
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "crs" in ds
        assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
        assert ds["foot"].attrs["grid_mapping"] == "crs"
        assert ds["lon"].attrs["standard_name"] == "longitude"
        assert ds["lon"].attrs["units"] == "degrees_east"
        assert ds["lat"].attrs["standard_name"] == "latitude"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds["time"].attrs["standard_name"] == "time"
        assert "stilt_receptor" in ds["foot"].attrs
        assert "stilt_footprint" in ds["foot"].attrs
        assert ds["receptor"].item() == "202301011200_-111.85_40.77_5"
        assert "receptor_time" not in ds.coords
        assert "receptor_longitude" not in ds.coords
        assert "receptor_latitude" not in ds.coords
        assert "receptor_altitude" not in ds.coords
    finally:
        ds.close()


def test_netcdf_roundtrip_prefers_stored_name_attr(tmp_path):
    foot = make_footprint(n_times=1)
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    original = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    renamed = sim_dir / "202301011200_-111.85_40.77_5_wrong_foot.nc"
    foot.stilt.to_netcdf(original)

    ds = xr.open_dataset(original)
    ds.load()
    ds.close()
    ds["foot"].attrs["stilt_name"] = "stored"
    ds.to_netcdf(renamed)

    loaded = read_footprint(renamed)
    assert loaded.stilt.name == "stored"


def test_netcdf_roundtrip_preserves_transforms(tmp_path):
    foot = make_footprint(n_times=1)
    kernel = AveragingKernel(levels=[0.0, 1000.0], values=[0.1, 0.9], coordinate="xhgt")
    config = FootprintConfig(grid=foot.stilt.grid, transforms=[kernel])
    foot = as_footprint(foot, foot.stilt.receptor, config, "slv")
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)

    assert len(loaded.stilt.config.transforms) == 1
    transform = loaded.stilt.config.transforms[0]
    assert isinstance(transform, AveragingKernel)
    assert transform == kernel


def test_netcdf_with_unimportable_transform_still_loads(tmp_path):
    # A footprint written where a user transform class was importable must
    # still open on a machine without that package; the transform is kept as
    # its settings mapping.
    foot = make_footprint(n_times=1)
    config = FootprintConfig(
        grid=foot.stilt.grid,
        transforms=[AveragingKernel(levels=[0.0, 1000.0], values=[0.1, 0.9])],
    )
    foot = as_footprint(foot, foot.stilt.receptor, config, "slv")
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    original = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    patched = sim_dir / "202301011200_-111.85_40.77_5_user_foot.nc"
    foot.stilt.to_netcdf(original)

    missing_kind = "no_such_pkg_for_stilt_tests.transforms.MyKernel"
    ds = xr.open_dataset(original)
    ds.load()
    ds.close()
    settings = json.loads(ds["foot"].attrs["stilt_footprint"])
    settings["transforms"] = [
        {"kind": missing_kind, "levels": [0.0, 1000.0], "values": [0.1, 0.9]}
    ]
    ds["foot"].attrs["stilt_footprint"] = json.dumps(settings)
    ds.to_netcdf(patched)

    recorded = {"kind": missing_kind, "levels": [0.0, 1000.0], "values": [0.1, 0.9]}
    loaded = read_footprint(patched)
    with pytest.warns(UserWarning, match="could not be imported"):
        assert loaded.stilt.config.transforms == [recorded]

    # The mapping survives another write/read unchanged. The accessor read the
    # settings once above, so writing does not warn again.
    rewritten = sim_dir / "202301011200_-111.85_40.77_5_again_foot.nc"
    loaded.stilt.to_netcdf(rewritten)
    again = read_footprint(rewritten)
    with pytest.warns(UserWarning):
        assert again.stilt.config.transforms == [recorded]


def test_netcdf_roundtrip_no_name(tmp_path):
    """Footprint with no name (unnamed) roundtrips as empty string."""
    foot = make_footprint(n_times=1)
    foot.attrs["stilt_name"] = ""
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_foot.nc"
    foot.stilt.to_netcdf(path)
    loaded = read_footprint(path)
    assert loaded.stilt.name == ""


def test_netcdf_roundtrip_with_timezone_aware_time(tmp_path):
    receptor_time = pd.Timestamp("2023-01-01 12:00:00+00:00")
    receptor = PointReceptor(
        time=receptor_time,
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.8, ymin=39.0, ymax=39.2, xres=0.1, yres=0.1)
    )
    data = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={
            "time": [receptor_time],
            "lat": np.array([39.05, 39.15]),
            "lon": np.array([-113.95, -113.85]),
        },
        attrs={"units": "ppm (umol-1 m2 s)"},
    )
    foot = as_footprint(data, receptor, config, "slv")

    path = tmp_path / "timezone_aware_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)
    assert tuple(loaded.dims) == ("time", "lat", "lon")
    assert loaded.shape == (1, 2, 2)
    assert float(loaded.sum()) == pytest.approx(4.0)
    # Time coord and receptor.time must come back as naive UTC.
    assert loaded.stilt.receptor.time.tzinfo is None
    loaded_time = pd.Timestamp(loaded.time.values[0])
    assert loaded_time.tzinfo is None
    assert loaded_time == pd.Timestamp("2023-01-01 12:00:00")


# ---------------------------------------------------------------------------
# Geometry hash: recorded with geometry-derived footprints

# ---------------------------------------------------------------------------


def test_a_footprint_records_its_geometry_and_hash():
    foot, fc, mesh = geometry_footprint()
    assert foot.stilt.config == fc
    assert foot.stilt.geometry_hash == mesh.hash
    assert make_footprint().stilt.geometry_hash is None


def test_netcdf_roundtrip_keeps_geometry_and_hash(tmp_path):
    foot, fc, mesh = geometry_footprint()
    path = foot.stilt.to_netcdf(tmp_path / "geo_foot.nc")
    loaded = read_footprint(path)
    assert loaded.stilt.config.geometry == fc.geometry
    assert loaded.stilt.geometry_hash == mesh.hash
    # a grid-only footprint carries no geometry attrs at all
    plain = make_footprint().stilt.to_netcdf(tmp_path / "plain_foot.nc")
    with xr.open_dataset(plain) as ds:
        settings = json.loads(ds["foot"].attrs["stilt_footprint"])
    assert settings["geometry"] is None and settings["geometry_hash"] is None
    assert read_footprint(plain).stilt.geometry_hash is None


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def test_threads_default_to_the_cpus_this_process_may_use_at_most_eight(monkeypatch):
    """In a Slurm job the process may use fewer CPUs than the node has (#190)."""
    import os

    from stilt.footprint import io

    monkeypatch.setattr(os, "cpu_count", lambda: 112)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(4)), raising=False
    )
    assert io._threads(None) == 4
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(16)), raising=False
    )
    assert io._threads(None) == io.MAX_THREADS == 8
    assert io._threads(32) == 32  # an explicit count is kept


def test_threads_count_the_machine_where_affinity_is_unknown(monkeypatch):
    import os

    from stilt.footprint import io

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert io._threads(None) == 2
