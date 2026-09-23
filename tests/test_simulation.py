"""Tests for stilt.simulation (SimID and Simulation behavior)."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    STILTParams,
)
from stilt.errors import HYSPLITTimeoutError
from stilt.footprint import Footprint
from stilt.meteorology import MetStream
from stilt.simulation import ERROR_TRAJECTORY, TRAJECTORY, SimID, Simulation
from stilt.store import LocalStore
from stilt.trajectory import Trajectories
from stilt.transforms import FirstOrderLifetime


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
    met = MetStream(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
        n_min=mc.n_min,
        source_type=mc.source,
        source_kwargs=mc.source_kwargs,
        backend=mc.backend,
        subgrid_enable=mc.subgrid_enable,
        subgrid_bounds=mc.subgrid_bounds,
        subgrid_buffer=mc.subgrid_buffer,
        subgrid_levels=mc.subgrid_levels,
        subgrid_dir=mc.subgrid_dir,
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
    met = MetStream(
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
    assert not sim.directory.exists()
    assert sim.status is None


def test_simulation_without_directory_uses_canonical_temp_sim_id(
    point_receptor, tmp_path
):
    mc = _met_config(tmp_path)
    met = MetStream(
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


def test_meteorology_subgrid_requires_bounds(point_receptor, tmp_path):
    """subgrid_enable=True without bounds raises a ValidationError at config time."""
    with pytest.raises(Exception, match="subgrid_bounds is required"):
        _sim(tmp_path, point_receptor, met_kwargs={"subgrid_enable": True})


def test_meteorology_subgrid_enable_accepts_bool(point_receptor, tmp_path):
    """subgrid_enable=True with bounds is accepted (no longer raises NotImplementedError)."""
    from stilt.config.spatial import Bounds

    # Should not raise — subgrid is now implemented
    sim = _sim(
        tmp_path,
        point_receptor,
        met_kwargs={
            "subgrid_enable": True,
            "subgrid_bounds": Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
        },
    )
    assert sim.meteorology.subgrid_enable is True


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
    sim.meteorology = MetStream(
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
            self.error_particles = {}

    class _FakeRunner:
        def __init__(self, **kwargs):
            seen["runner_met_files"] = kwargs["met_files"]

        def prepare(self):
            return None

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
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

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
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
            self.error_particles = {
                0: pd.DataFrame(
                    {
                        "time": [-60],
                        "indx": [1],
                        "long": [-111.9],
                        "lati": [40.7],
                        "zagl": [10.0],
                        "foot": [2e-5],
                    }
                )
            }

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
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
        store=LocalStore(storage_root),
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
        store=LocalStore(output_root),
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
        store=LocalStore(storage_root),
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


def test_simulation_time_range_forward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=24)
    start, stop = sim.time_range
    assert start == point_receptor.time
    assert stop - start == dt.timedelta(hours=24)


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
        transforms=[FirstOrderLifetime(lifetime_hours=1.0)],
    )

    base = sim.generate_footprint("base", base_config)
    transformed = sim.generate_footprint("chem", transformed_config)

    assert base is not None
    assert transformed is not None
    assert float(transformed.data.sum()) < float(base.data.sum())


class HalvingTransform:
    """Python-side transform that halves ``foot`` and records its context."""

    def __init__(self):
        self.context = None

    def apply(self, particles, context):
        self.context = context
        out = particles.copy()
        out["foot"] = out["foot"] * 0.5
        return out


def _sim_with_particles(point_receptor, tmp_path):
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
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1),
        time_integrate=True,
        smooth_factor=0.0,
    )
    return sim, config


def test_generate_footprint_applies_extra_python_transforms(point_receptor, tmp_path):
    sim, config = _sim_with_particles(point_receptor, tmp_path)
    halve = HalvingTransform()

    base = sim.generate_footprint("base", config)
    halved = sim.generate_footprint("halved", config, transforms=[halve])

    assert float(base.data.sum()) > 0
    assert float(halved.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))
    assert halve.context is not None
    assert halve.context.receptor is sim.receptor
    assert halve.context.footprint_name == "halved"
    assert halve.context.is_error is False


def test_generate_footprint_applies_dotted_path_config_transforms(
    point_receptor, tmp_path
):
    sim, config = _sim_with_particles(point_receptor, tmp_path)
    dotted = FootprintConfig(
        grid=config.grid,
        time_integrate=True,
        smooth_factor=0.0,
        transforms=[{"kind": f"{__name__}.{HalvingTransform.__name__}"}],
    )
    assert isinstance(dotted.transforms[0], HalvingTransform)
    halve = dotted.transforms[0]

    base = sim.generate_footprint("base", config)
    halved = sim.generate_footprint("halved", dotted)

    assert float(base.data.sum()) > 0
    assert float(halved.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))
    assert halve.context is not None
    assert halve.context.receptor is sim.receptor
    assert halve.context.footprint_name == "halved"
    assert halve.context.is_error is False


def test_simulation_error_trajectories_none_when_no_file(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.error_trajectories is None


def test_simulation_trajectories_load_from_storage_backend(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    _write_remote_trajectories(storage_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=LocalStore(storage_root),
    )

    assert sim.trajectories is not None
    assert not sim.trajectories.is_error


def test_simulation_trajectories_load_from_store(point_receptor, tmp_path):
    output_root = tmp_path / "artifacts"
    _write_remote_trajectories(output_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=LocalStore(output_root),
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
        store=LocalStore(storage_root),
    )

    assert sim.error_trajectories is not None
    assert sim.error_trajectories.is_error


def test_simulation_trajectories_none_when_no_parquet(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.trajectories is None


def test_generate_footprint_autoruns_trajectories_when_missing(
    point_receptor, tmp_path, monkeypatch
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
    run_calls: list[bool] = []

    def _fake_run_trajectories(*, write: bool = False, **kwargs) -> None:
        del kwargs
        run_calls.append(write)
        sim._trajectories = Trajectories.from_particles(
            particles=particles,
            receptor=point_receptor,
            params=_params(tmp_path),
            met_files=[tmp_path / "metfile"],
        )

    monkeypatch.setattr(sim, "run_trajectories", _fake_run_trajectories)

    foot = sim.generate_footprint(
        "slv",
        FootprintConfig(
            grid=Grid(
                xmin=-114.0,
                xmax=-111.0,
                ymin=39.0,
                ymax=42.0,
                xres=0.1,
                yres=0.1,
            ),
            time_integrate=True,
            smooth_factor=0.0,
        ),
    )

    assert foot is not None
    assert run_calls == [False]
    assert sim.trajectories is not None


def test_simulation_status_uses_storage_backed_artifacts(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    storage_root = tmp_path / "remote"
    _write_remote_trajectories(storage_root, point_receptor, is_error=False)

    sim = _sim(
        tmp_path / "cache",
        point_receptor,
        store=LocalStore(storage_root),
    )
    assert sim.status == "complete"

    traj_path = storage_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.unlink()
    log_path = storage_root / "simulations" / "by-id" / sid / "stilt.log"
    log_path.write_text("Insufficient number of meteorological files found")

    assert "MISSING_MET_FILES" in str(sim.status)


# ---------------------------------------------------------------------------
# Error-only backfill: reuse an existing main trajectory, run only the error pass
# ---------------------------------------------------------------------------

_ERR = dict(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)


def _particles_df(foot: float = 1e-5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60],
            "indx": [1],
            "long": [-111.9],
            "lati": [40.7],
            "zagl": [10.0],
            "foot": [foot],
        }
    )


def _write_trajectory(sim, params, *, is_error: bool = False, foot: float = 1e-5):
    traj = Trajectories.from_particles(
        _particles_df(foot),
        receptor=sim.receptor,
        params=params,
        met_files=[],
        is_error=is_error,
    )
    path = sim.error_trajectories_path() if is_error else sim.trajectories_path
    path.parent.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(path)


def test_reuse_main_for_error_when_present_and_params_match(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _write_trajectory(sim, sim.params)
    assert sim._can_reuse_main_for_error() is True


def test_reuse_main_for_error_when_only_error_params_were_added(
    tmp_path, point_receptor
):
    # Main was generated WITHOUT error params; now error params are configured —
    # the main is still valid, only the error trajectory needs running.
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _write_trajectory(sim, _params(tmp_path))  # stored params carry no error fields
    assert sim._can_reuse_main_for_error() is True


def test_no_reuse_when_error_not_configured(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor)  # winderrtf == 0
    _write_trajectory(sim, sim.params)
    assert sim._can_reuse_main_for_error() is False


def test_no_reuse_when_main_missing(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    assert sim._can_reuse_main_for_error() is False


def test_no_reuse_when_error_already_present(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _write_trajectory(sim, sim.params)
    _write_trajectory(sim, sim.params, is_error=True, foot=2e-5)
    assert sim._can_reuse_main_for_error() is False


def test_no_reuse_when_non_error_params_differ(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    stored = _params(tmp_path, numpar=sim.params.numpar + 100, **_ERR)
    _write_trajectory(sim, stored)
    assert sim._can_reuse_main_for_error() is False


def test_run_trajectories_error_only_skips_main_run(
    tmp_path, point_receptor, monkeypatch
):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _write_trajectory(sim, sim.params)
    original_main = sim.trajectories_path.read_bytes()

    captured: dict = {}

    class _FakeMet:
        def required_files(self, **kwargs):
            return []

        def stage_files_for_simulation(self, **kwargs):
            return []

    class _Result:
        stdout = "ok"
        particles = None
        error_particles = {0: _particles_df(2e-5)}

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
            captured["error_only"] = error_only
            return _Result()

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    monkeypatch.setattr(sim, "meteorology", _FakeMet())

    sim.run_trajectories(timeout=1, rm_dat=False, write=True)

    assert captured["error_only"] is True
    assert sim.error_trajectories is not None
    # The existing main trajectory is left untouched (not recomputed/overwritten).
    assert sim.trajectories_path.read_bytes() == original_main


# ---------------------------------------------------------------------------
# Construction, keys, and directory creation
# ---------------------------------------------------------------------------


class _FakeMet:
    def required_files(self, **kwargs):
        return []

    def stage_files_for_simulation(self, **kwargs):
        return []


def _fake_runner_returning(particles, error_particles=None):
    class _Result:
        stdout = "ok"

    _Result.particles = particles
    _Result.error_particles = (
        {0: error_particles} if error_particles is not None else {}
    )

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
            return _Result()

    return _FakeRunner


def test_construction_does_not_create_directory(point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    sim_dir = tmp_path / "simulations" / "by-id" / sid
    mc = _met_config(tmp_path)
    met = MetStream(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
    )

    sim = Simulation(
        directory=sim_dir,
        receptor=point_receptor,
        params=_params(tmp_path),
        meteorology=met,
    )

    assert sim.directory == sim_dir
    assert not sim_dir.exists()
    assert not sim.has_trajectory
    assert sim.status is None


def test_run_trajectories_creates_directory(monkeypatch, point_receptor, tmp_path):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    sim_dir = tmp_path / "simulations" / "by-id" / sid
    mc = _met_config(tmp_path)
    met = MetStream(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
    )
    sim = Simulation(
        directory=sim_dir,
        receptor=point_receptor,
        params=_params(tmp_path),
        meteorology=met,
    )
    assert not sim_dir.exists()

    monkeypatch.setattr(
        "stilt.simulation.HYSPLITDriver", _fake_runner_returning(_particles_df())
    )
    monkeypatch.setattr(sim, "meteorology", _FakeMet())

    sim.run_trajectories(timeout=1, rm_dat=False, write=True)

    assert sim_dir.is_dir()
    assert sim.trajectories_path.exists()
    assert sim.has_trajectory
    assert sim.log_path.read_text() == "ok"


def test_key_prefix_and_key(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    sid = str(sim.id)

    assert sim.key_prefix == f"simulations/by-id/{sid}"
    assert sim.key(sim.trajectories_path) == (
        f"simulations/by-id/{sid}/{sid}_traj.parquet"
    )
    assert sim.key(sim.error_trajectories_path()) == (
        f"simulations/by-id/{sid}/{sid}_error.parquet"
    )
    assert sim.key(sim.log_path) == f"simulations/by-id/{sid}/stilt.log"
    assert sim.key(sim.footprint_path("slv")) == (
        f"simulations/by-id/{sid}/{sid}_slv_foot.nc"
    )
    assert sim.key(sim.empty_footprint_path("slv")) == (
        f"simulations/by-id/{sid}/{sid}_slv_foot.empty"
    )
    # Only the basename matters: an out-of-tree path maps onto this sim's prefix.
    assert sim.key("elsewhere/stilt.log") == f"simulations/by-id/{sid}/stilt.log"


def test_resolve_prefers_local_then_store_then_none(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))

    assert sim.resolve(sim.log_path) is None

    remote_log = storage_root / sim.key_prefix / "stilt.log"
    remote_log.parent.mkdir(parents=True)
    remote_log.write_text("remote")
    assert sim.resolve(sim.log_path) == remote_log

    sim.log_path.write_text("local")
    assert sim.resolve(sim.log_path) == sim.log_path


# ---------------------------------------------------------------------------
# Expected outputs and completion (ported from tests/test_completion.py)
# ---------------------------------------------------------------------------


def _touch(sim, kind, name=""):
    path = {
        "traj": sim.trajectories_path,
        "error": sim.error_trajectories_path(),
        "foot": sim.footprint_path(name),
        "empty": sim.empty_footprint_path(name),
    }[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_expected_outputs_without_error(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.expected_outputs(["default"]) == (TRAJECTORY, "default")
    assert sim.expected_outputs() == (TRAJECTORY,)


def test_expected_outputs_with_error_includes_error_trajectory(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    assert sim.params.error_enabled
    assert sim.expected_outputs(["default"]) == (
        TRAJECTORY,
        ERROR_TRAJECTORY,
        "default",
    )


def test_expected_outputs_never_requires_error_footprint(point_receptor, tmp_path):
    """error_enabled gates the error trajectory, never an error footprint."""
    sim = _sim(tmp_path, point_receptor, **_ERR)
    assert "default_error" not in sim.expected_outputs(["default"])


def test_is_complete_when_error_not_expected(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "traj")
    _touch(sim, "foot", "default")
    assert sim.is_complete(["default"]) is True


def test_incomplete_when_error_expected_but_missing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _touch(sim, "traj")
    _touch(sim, "foot", "default")
    assert sim.is_complete(["default"]) is False


def test_complete_when_error_present(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, **_ERR)
    _touch(sim, "traj")
    _touch(sim, "error")
    _touch(sim, "foot", "default")
    assert sim.is_complete(["default"]) is True


def test_empty_footprint_marker_counts_complete(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "traj")
    _touch(sim, "empty", "default")
    assert sim.is_complete(["default"]) is True


def test_incomplete_when_trajectory_missing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "foot", "default")
    assert sim.is_complete(["default"]) is False


def test_incomplete_when_footprint_missing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "traj")
    assert sim.is_complete(["default"]) is False
    assert sim.is_complete() is True


def test_is_complete_checks_trajectory_first(monkeypatch, point_receptor, tmp_path):
    """With no trajectory, footprints are never inspected."""
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "foot", "default")
    checked: list[str] = []

    def _spy_has_footprint(name):
        checked.append(name)
        return True

    monkeypatch.setattr(sim, "has_footprint", _spy_has_footprint)

    assert sim.is_complete(["default"]) is False
    assert checked == []


# ---------------------------------------------------------------------------
# has_footprint / missing_footprints / has_output / markers
# ---------------------------------------------------------------------------


def test_has_footprint_from_netcdf_or_marker(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert not sim.has_footprint("slv")

    _touch(sim, "foot", "slv")
    assert sim.has_footprint("slv")

    _touch(sim, "empty", "other")
    assert sim.has_footprint("other")
    assert not sim.has_footprint("missing")


def test_has_footprint_falls_back_to_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    remote = storage_root / sim.key(sim.footprint_path("slv"))
    remote.parent.mkdir(parents=True)
    remote.write_bytes(b"x")

    assert sim.has_footprint("slv")
    assert not sim.has_footprint("other")


def test_missing_footprints(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _touch(sim, "foot", "a")
    _touch(sim, "empty", "b")

    assert sim.missing_footprints(["a", "b", "c", "d"]) == ["c", "d"]
    assert sim.missing_footprints([]) == []


def test_has_output_dispatches_by_name(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert not sim.has_output(TRAJECTORY)
    assert not sim.has_output(ERROR_TRAJECTORY)
    assert not sim.has_output("slv")

    _touch(sim, "traj")
    _touch(sim, "error")
    _touch(sim, "foot", "slv")

    assert sim.has_output(TRAJECTORY)
    assert sim.has_output(ERROR_TRAJECTORY)
    assert sim.has_output("slv")
    assert sim.has_trajectory
    assert sim.has_error_trajectory


def test_write_and_clear_empty_footprint_marker(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    marker = sim.write_empty_footprint_marker("slv")

    assert marker == sim.empty_footprint_path("slv")
    assert marker.name.endswith("_slv_foot.empty")
    assert marker.exists()
    assert sim.has_footprint("slv")

    sim.clear_empty_footprint_marker("slv")
    assert not marker.exists()
    assert not sim.has_footprint("slv")
    # Clearing twice is harmless.
    sim.clear_empty_footprint_marker("slv")


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _write_all_outputs(sim):
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.log_path.write_text("log")
    sim.trajectories_path.write_bytes(b"traj")
    sim.error_trajectories_path().write_bytes(b"error")
    sim.footprint_path("slv").write_bytes(b"foot")
    sim.write_empty_footprint_marker("empty")


def test_publish_copies_outputs_into_store(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    store = LocalStore(storage_root)
    sim = _sim(tmp_path / "compute", point_receptor, store=store)
    _write_all_outputs(sim)

    sim.publish()

    published = storage_root / sim.key_prefix
    assert (published / "stilt.log").read_text() == "log"
    assert (published / sim.trajectories_path.name).read_bytes() == b"traj"
    assert (published / sim.error_trajectories_path().name).read_bytes() == b"error"
    assert (published / sim.footprint_path("slv").name).read_bytes() == b"foot"
    assert (published / sim.empty_footprint_path("empty").name).exists()
    assert not list(published.glob("*.tmp"))
    # Nothing is written outside the by-id layout.
    assert sorted(p.name for p in storage_root.iterdir()) == ["simulations"]


def test_publish_skips_missing_outputs(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(tmp_path / "compute", point_receptor, store=LocalStore(storage_root))
    sim.trajectories_path.write_bytes(b"traj")

    sim.publish()

    published = storage_root / sim.key_prefix
    assert (published / sim.trajectories_path.name).read_bytes() == b"traj"
    assert not (published / "stilt.log").exists()
    assert not (published / sim.error_trajectories_path().name).exists()


def test_publish_noop_when_store_root_contains_sim_directory(point_receptor, tmp_path):
    """When the store's location for the sim *is* its directory, nothing is copied."""
    sim = _sim(tmp_path, point_receptor, store=LocalStore(tmp_path))
    assert sim.directory == LocalStore(tmp_path).path(sim.key_prefix)
    _write_all_outputs(sim)
    before = {
        p.name: p.stat().st_mtime_ns for p in sim.directory.iterdir() if p.is_file()
    }

    sim.publish()

    after = {
        p.name: p.stat().st_mtime_ns for p in sim.directory.iterdir() if p.is_file()
    }
    assert after == before
    assert not list(sim.directory.glob("*.tmp"))
    assert sim.trajectories_path.read_bytes() == b"traj"


def test_publish_noop_without_store(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _write_all_outputs(sim)
    sim.publish()  # must not raise
    assert sim.trajectories_path.read_bytes() == b"traj"


def test_publish_without_directory_is_noop(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(tmp_path / "compute", point_receptor, store=LocalStore(storage_root))
    sim.directory.rmdir()

    sim.publish()

    assert not (storage_root / sim.key_prefix).exists()


def test_generate_footprint_uses_the_receptor_kernel_from_a_project_table(
    column_receptor, tmp_path
):
    """Two receptors, one config, one table: each footprint gets its own kernel."""
    from stilt.receptors import ColumnReceptor
    from stilt.store import LocalStore
    from stilt.transforms import AveragingKernel, averaging_kernel_table

    store = LocalStore(tmp_path)
    other = ColumnReceptor(
        column_receptor.time,
        column_receptor.longitude + 0.1,
        column_receptor.latitude,
        column_receptor.bottom,
        column_receptor.top,
    )
    averaging_kernel_table(
        [column_receptor, other],
        levels=[0.0, 3000.0],
        values=[[1.0, 1.0], [0.25, 0.25]],
    ).to_parquet(tmp_path / "kernels.parquet")
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1),
        time_integrate=True,
        smooth_factor=0.0,
        transforms=[AveragingKernel(table="kernels.parquet")],
    )

    sums = {}
    for receptor in (column_receptor, other):
        sim = _sim(tmp_path, receptor, store=store)
        particles = pd.DataFrame(
            {
                "time": [0.0, -60.0, 0.0, -60.0],
                "indx": [1, 1, 2, 2],
                "long": [receptor.longitude] * 4,
                "lati": [receptor.latitude] * 4,
                "zagl": [500.0, 500.0, 2500.0, 2500.0],
                "xhgt": [500.0, 500.0, 2500.0, 2500.0],
                "foot": [1.0, 1.0, 1.0, 1.0],
            }
        )
        sim._trajectories = Trajectories.from_particles(
            particles=particles,
            receptor=receptor,
            params=_params(tmp_path),
            met_files=[tmp_path / "metfile"],
        )
        sums[receptor.id] = float(sim.generate_footprint("column", config).data.sum())

    assert sums[column_receptor.id] > 0
    assert sums[other.id] == pytest.approx(0.25 * sums[column_receptor.id])


def test_transform_context_carries_receptor_name_error_flag_and_store(
    point_receptor, tmp_path
):
    from stilt.store import LocalStore

    store = LocalStore(tmp_path)
    sim = _sim(tmp_path, point_receptor, store=store)

    ctx = sim.transform_context("column", error=True)

    assert ctx.receptor is point_receptor
    assert ctx.footprint_name == "column"
    assert ctx.is_error is True
    assert ctx.store is store
    assert _sim(tmp_path, point_receptor).transform_context().store is None


# -- error realizations ------------------------------------------------------------


def _write_error_realization(sim, realization: int, foot: float = 2e-5) -> None:
    traj = Trajectories.from_particles(
        _particles_df(foot),
        receptor=sim.receptor,
        params=sim.params,
        met_files=[],
        is_error=True,
    )
    path = sim.error_trajectories_path(realization)
    path.parent.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(path)


def test_error_trajectories_path_suffixes_realizations_after_the_first(
    tmp_path, point_receptor
):
    sim = _sim(tmp_path, point_receptor, **_ERR, error_realizations=3)
    assert sim.error_trajectories_path().name == f"{sim.id}_error.parquet"
    assert sim.error_trajectories_path(0) == sim.error_trajectories_path()
    assert sim.error_trajectories_path(2).name == f"{sim.id}_error_2.parquet"
    assert sim.error_realizations == (0, 1, 2)


def test_error_realizations_are_empty_without_wind_error(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, error_realizations=3)
    assert sim.error_realizations == ()
    # file-existence semantics survive for the unconfigured case
    assert not sim.has_error_trajectory
    _touch(sim, "error")
    assert sim.has_error_trajectory


def test_completion_requires_every_realization(tmp_path, point_receptor):
    sim = _sim(tmp_path, point_receptor, **_ERR, error_realizations=3)
    _write_trajectory(sim, sim.params)
    _write_error_realization(sim, 0)

    assert sim.missing_error_realizations == [1, 2]
    assert not sim.has_error_trajectory
    assert not sim.is_complete()
    assert sim.error_trajectories is not None
    assert sim.error_trajectory(1) is None
    assert len(sim.all_error_trajectories) == 1

    _write_error_realization(sim, 1)
    _write_error_realization(sim, 2)

    assert sim.missing_error_realizations == []
    assert sim.has_error_trajectory
    assert sim.is_complete()
    ensemble = sim.all_error_trajectories
    assert len(ensemble) == 3
    assert all(t.is_error for t in ensemble)
    assert ensemble[2] is sim.error_trajectory(2)


def test_run_trajectories_requests_only_missing_realizations(
    tmp_path, point_receptor, monkeypatch
):
    sim = _sim(tmp_path, point_receptor, **_ERR, error_realizations=3)
    _write_trajectory(sim, sim.params)
    _write_error_realization(sim, 0)
    main_bytes = sim.trajectories_path.read_bytes()

    asked: dict = {}

    class _FakeMet:
        def required_files(self, **kwargs):
            return []

        def stage_files_for_simulation(self, **kwargs):
            return []

    class _Result:
        stdout = "ok"
        particles = None
        error_particles = {1: _particles_df(3e-5), 2: _particles_df(4e-5)}

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(
            self, timeout, rm_dat, *, error_only=False, error_realizations=(0,)
        ):
            asked["error_only"] = error_only
            asked["realizations"] = list(error_realizations)
            return _Result()

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    sim.meteorology = _FakeMet()

    sim.run_trajectories(write=True)

    assert asked == {"error_only": True, "realizations": [1, 2]}
    assert sim.trajectories_path.read_bytes() == main_bytes  # main untouched
    assert sim.missing_error_realizations == []
    assert sim.is_complete()
    assert float(sim.error_trajectory(2).data["foot"].iloc[0]) == pytest.approx(4e-5)


def test_publish_copies_every_realization(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(
        tmp_path / "compute",
        point_receptor,
        store=LocalStore(storage_root),
        **_ERR,
        error_realizations=2,
    )
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.trajectories_path.write_bytes(b"traj")
    sim.error_trajectories_path(0).write_bytes(b"e0")
    sim.error_trajectories_path(1).write_bytes(b"e1")

    sim.publish()

    published = storage_root / sim.key_prefix
    assert (published / sim.error_trajectories_path(0).name).read_bytes() == b"e0"
    assert (published / sim.error_trajectories_path(1).name).read_bytes() == b"e1"
