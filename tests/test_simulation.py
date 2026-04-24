"""Tests for stilt.simulation (SimID and Simulation behavior)."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from stilt.config import (
    FirstOrderLifetimeTransformSpec,
    FootprintConfig,
    Grid,
    MetConfig,
    STILTParams,
)
from stilt.errors import HYSPLITTimeoutError
from stilt.footprint import Footprint
from stilt.meteorology import MetSource
from stilt.simulation import SimID, Simulation
from stilt.storage import FsspecStore
from stilt.trajectory import Trajectories


def _params(tmp_path=None, **kwargs) -> STILTParams:
    data = {
        "n_hours": -24,
        "numpar": 10,
        "hnf_plume": False,
    }
    data.update(kwargs)
    return STILTParams(**data)


def _met_config(tmp_path, **kwargs) -> MetConfig:
    data = {
        "directory": tmp_path / "met",
        "file_format": "%Y%m%d_%H",
        "file_tres": "1h",
    }
    data.update(kwargs)
    return MetConfig(**data)


def _sim(
    tmp_path,
    point_receptor,
    met_kwargs=None,
    store=None,
    **param_overrides,
) -> Simulation:
    sid = str(SimID.from_parts("hrrr", point_receptor))
    sim_dir = tmp_path / "simulations" / "by-id" / sid
    sim_dir.mkdir(parents=True, exist_ok=True)
    mc = _met_config(tmp_path, **(met_kwargs or {}))
    met = MetSource(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
        n_min=mc.n_min,
        subgrid_enable=mc.subgrid_enable,
        subgrid_bounds=mc.subgrid_bounds,
        subgrid_buffer=mc.subgrid_buffer,
        subgrid_levels=mc.subgrid_levels,
    )
    return Simulation(
        directory=sim_dir,
        receptor=point_receptor,
        params=_params(tmp_path, **param_overrides),
        meteorology=met,
        store=store,
    )


def _write_remote_trajectories(storage_root, point_receptor, *, is_error=False) -> Path:
    sid = str(SimID.from_parts("hrrr", point_receptor))
    sim_dir = storage_root / "simulations" / "by-id" / sid
    sim_dir.mkdir(parents=True, exist_ok=True)
    path = (
        sim_dir / f"{sid}_error.parquet"
        if is_error
        else sim_dir / f"{sid}_traj.parquet"
    )
    particles = pd.DataFrame(
        {
            "time": [-60],
            "indx": [1],
            "long": [-111.9],
            "lati": [40.7],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    traj = Trajectories.from_particles(
        particles=particles,
        receptor=point_receptor,
        params=_params(storage_root),
        met_files=[storage_root / "metfile"],
        is_error=is_error,
    )
    traj.to_parquet(path)
    return path


def test_simid_roundtrip():
    value = "hrrr_202301011200_-111.85_40.77_5"
    sid = SimID(value)
    assert sid.met == "hrrr"
    assert sid.time == pd.Timestamp("2023-01-01 12:00")
    assert str(sid) == value


def test_simid_invalid_raises():
    with pytest.raises(ValueError):
        SimID("20230101")


def test_simid_from_receptor(point_receptor):
    sid = SimID.from_parts("hrrr", point_receptor)
    assert sid.location == point_receptor.id.location
    assert sid.met == "hrrr"


def test_simid_is_pathlike(point_receptor, tmp_path):
    sid = SimID.from_parts("hrrr", point_receptor)
    p = tmp_path / "simulations" / "by-id" / sid
    assert p.name == str(sid)


def test_simulation_status_none_when_dir_missing(point_receptor, tmp_path):
    mc = _met_config(tmp_path)
    met = MetSource(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
    )
    sid = str(SimID.from_parts("hrrr", point_receptor))
    sim = Simulation(
        directory=tmp_path / sid,
        receptor=point_receptor,
        params=_params(tmp_path),
        meteorology=met,
    )
    sim.directory.rmdir()
    assert sim.status is None


def test_simulation_without_directory_uses_canonical_temp_sim_id(
    point_receptor, tmp_path
):
    mc = _met_config(tmp_path)
    met = MetSource(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
    )

    sim = Simulation(
        directory=None,
        receptor=point_receptor,
        params=_params(tmp_path),
        meteorology=met,
    )

    assert sim.id == SimID.from_parts("hrrr", point_receptor)
    assert sim.directory.name == str(sim.id)


def test_simulation_status_complete_when_traj_present(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    sim.trajectories_path.touch()
    assert sim.status == "complete"


def test_meteorology_requires_explicit_subgrid_path(point_receptor, tmp_path):
    with pytest.raises(
        NotImplementedError,
        match="subgrid_enable is not yet implemented",
    ):
        _sim(tmp_path, point_receptor, met_kwargs={"subgrid_enable": True})


def test_meteorology_requires_bounds_when_subgrid_enabled(point_receptor, tmp_path):
    with pytest.raises(
        NotImplementedError,
        match="subgrid_enable is not yet implemented",
    ):
        _sim(
            tmp_path,
            point_receptor,
            met_kwargs={"subgrid_enable": tmp_path / "subgrid"},
        )


def test_simulation_met_files_stage_into_compute_dir(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    sim.meteorology.directory.mkdir(parents=True, exist_ok=True)
    source = sim.meteorology.directory / point_receptor.time.strftime(
        sim.meteorology.file_format
    )
    source.touch()

    staged = sim.met_files

    assert staged == [sim.met_dir / source.name]
    assert staged[0].exists()
    assert sim.source_met_files == [source]


def test_simulation_run_trajectories_uses_source_met_files_in_metadata(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)
    source_dir = tmp_path / "archive" / "hrrr"
    source_dir.mkdir(parents=True)
    source_file = source_dir / point_receptor.time.strftime("%Y%m%d_%H")
    source_file.touch()
    sim.meteorology = MetSource(
        "hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )

    seen: dict[str, list[Path]] = {}

    class _Result:
        def __init__(self):
            self.stdout = "ok"
            self.particles = pd.DataFrame(
                {
                    "time": [-60],
                    "indx": [1],
                    "long": [-111.9],
                    "lati": [40.7],
                    "zagl": [10.0],
                    "foot": [1e-5],
                }
            )
            self.error_particles = None

    class _FakeRunner:
        def __init__(self, **kwargs):
            seen["runner_met_files"] = kwargs["met_files"]

        def prepare(self):
            return None

        def execute(self, timeout, rm_dat):
            return _Result()

    def _fake_from_particles(particles, *, receptor, params, met_files, is_error=False):
        seen["traj_met_files"] = met_files
        return Trajectories(
            receptor=receptor,
            params=params,
            met_files=met_files,
            data=particles.assign(datetime=pd.to_datetime(point_receptor.time)),
            is_error=is_error,
        )

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    monkeypatch.setattr(
        "stilt.simulation.Trajectories.from_particles", _fake_from_particles
    )

    sim.run_trajectories(timeout=1, rm_dat=False, write=False)

    assert seen["runner_met_files"] == [sim.met_dir / source_file.name]
    assert seen["traj_met_files"] == [source_file]


def test_run_trajectories_timeout_maps_to_domain_error(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)

    class _FakeMet:
        def required_files(self, **kwargs):
            return []

        def stage_files_for_simulation(self, **kwargs):
            return []

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(self, timeout, rm_dat):
            raise HYSPLITTimeoutError("boom")

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    monkeypatch.setattr(sim, "meteorology", _FakeMet())

    with pytest.raises(HYSPLITTimeoutError):
        sim.run_trajectories(timeout=1, rm_dat=False)


def test_run_trajectories_sets_main_and_error_trajectories(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)

    class _FakeMet:
        def required_files(self, **kwargs):
            return []

        def stage_files_for_simulation(self, **kwargs):
            return []

    class _Result:
        def __init__(self):
            self.stdout = "ok"
            self.particles = pd.DataFrame(
                {
                    "time": [-60],
                    "indx": [1],
                    "long": [-111.9],
                    "lati": [40.7],
                    "zagl": [10.0],
                    "foot": [1e-5],
                }
            )
            self.error_particles = pd.DataFrame(
                {
                    "time": [-60],
                    "indx": [1],
                    "long": [-111.9],
                    "lati": [40.7],
                    "zagl": [10.0],
                    "foot": [2e-5],
                }
            )

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(self, timeout, rm_dat):
            return _Result()

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    monkeypatch.setattr(sim, "meteorology", _FakeMet())

    sim.run_trajectories(timeout=1, rm_dat=False, write=False)

    assert sim.trajectories is not None
    assert sim.error_trajectories is not None
    assert sim.log_path.read_text() == "ok"


def test_log_property_raises_when_missing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    with pytest.raises(FileNotFoundError):
        _ = sim.log


def test_simulation_log_loads_from_artifact_store_fallback(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    storage_root = tmp_path / "remote"
    log_path = storage_root / "simulations" / "by-id" / sid / "stilt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("cloud log")

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(storage_root),
    )

    assert sim.log == "cloud log"


def test_simulation_log_loads_from_store(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    output_root = tmp_path / "artifacts"
    log_path = output_root / "simulations" / "by-id" / sid / "stilt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("artifact log")

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(output_root),
    )

    assert sim.log == "artifact log"


def test_get_footprint_returns_none_when_not_present(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.get_footprint("slv") is None


def test_get_footprint_loads_from_artifact_store_fallback(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    storage_root = tmp_path / "remote"
    sim_dir = storage_root / "simulations" / "by-id" / sid
    sim_dir.mkdir(parents=True, exist_ok=True)
    foot_path = sim_dir / f"{sid}_slv_foot.nc"

    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    )
    data = xr.DataArray(
        [[[1.0]]],
        coords={
            "time": [pd.Timestamp(point_receptor.time)],
            "lat": [40.77],
            "lon": [-111.85],
        },
        dims=("time", "lat", "lon"),
    )
    Footprint(point_receptor, config, data, name="slv").to_netcdf(foot_path)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(storage_root),
    )

    foot = sim.get_footprint("slv")
    assert foot is not None
    assert foot.name == "slv"


def test_simulation_foot_path(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.footprint_path("slv").name.endswith("_slv_foot.nc")
    assert sim.footprint_path("").name.endswith("_foot.nc")


def test_simulation_time_range_backward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=-24)
    start, stop = sim.time_range
    assert stop == point_receptor.time
    assert start < stop


def test_generate_footprint_applies_configured_particle_transforms(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)
    particles = pd.DataFrame(
        {
            "time": [0.0, -60.0, -120.0],
            "indx": [1, 1, 1],
            "long": [-111.86, -111.86, -111.86],
            "lati": [40.76, 40.76, 40.76],
            "zagl": [50.0, 50.0, 50.0],
            "foot": [1.0, 1.0, 1.0],
        }
    )
    sim._trajectories = Trajectories.from_particles(
        particles=particles,
        receptor=point_receptor,
        params=_params(tmp_path),
        met_files=[tmp_path / "metfile"],
    )
    base_config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1),
        time_integrate=True,
        smooth_factor=0.0,
    )
    transformed_config = FootprintConfig(
        grid=base_config.grid,
        time_integrate=True,
        smooth_factor=0.0,
        transforms=[
            FirstOrderLifetimeTransformSpec(
                kind="first_order_lifetime",
                lifetime_hours=1.0,
                time_column="time",
                time_unit="min",
            )
        ],
    )

    base = sim.generate_footprint("base", base_config)
    transformed = sim.generate_footprint("chem", transformed_config)

    assert base is not None
    assert transformed is not None
    assert float(transformed.data.sum()) < float(base.data.sum())


def test_simulation_error_trajectories_none_when_no_file(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.error_trajectories is None


def test_simulation_trajectories_load_from_storage_backend(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    _write_remote_trajectories(storage_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(storage_root),
    )

    assert sim.trajectories is not None
    assert not sim.trajectories.is_error


def test_simulation_trajectories_load_from_store(point_receptor, tmp_path):
    output_root = tmp_path / "artifacts"
    _write_remote_trajectories(output_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(output_root),
    )

    assert sim.trajectories is not None
    assert not sim.trajectories.is_error


def test_simulation_error_trajectories_load_from_storage_backend(
    point_receptor, tmp_path
):
    storage_root = tmp_path / "remote"
    _write_remote_trajectories(storage_root, point_receptor, is_error=True)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(storage_root),
    )

    assert sim.error_trajectories is not None
    assert sim.error_trajectories.is_error


def test_simulation_trajectories_none_when_no_parquet(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.trajectories is None


def test_simulation_status_uses_storage_backed_artifacts(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    storage_root = tmp_path / "remote"
    _write_remote_trajectories(storage_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=FsspecStore(storage_root),
    )
    assert sim.status == "complete"

    traj_path = storage_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.unlink()
    log_path = storage_root / "simulations" / "by-id" / sid / "stilt.log"
    log_path.write_text("Insufficient number of meteorological files found")

    assert "MISSING_MET_FILES" in str(sim.status)
