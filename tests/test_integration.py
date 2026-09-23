"""
Integration tests for PYSTILT end-to-end execution.

These tests run real HYSPLIT against real met files and are collected by
default.  They are automatically skipped when the stilt-tutorials met data
is not present (the ``met_dir`` fixture handles this).

Run integration tests explicitly:
    pytest tests/ -v -m integration

Skip them:
    pytest tests/ -m "not integration"
"""

import pandas as pd
import xarray as xr

from stilt.config import MetConfig
from stilt.model import Model
from stilt.simulation import SimID

from .conftest import integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_id(receptor, met: str = "hrrr") -> str:
    return str(SimID.from_parts(met, receptor))


# ---------------------------------------------------------------------------
# Point receptor - trajectory only
# ---------------------------------------------------------------------------


@integration
def test_trajectory(tmp_path, wbb_receptor, traj_only_config):
    """HYSPLIT runs and produces a non-empty trajectory parquet."""
    model = Model(
        project=tmp_path / "trajectory",
        config=traj_only_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sid = _sim_id(wbb_receptor)
    sim_dir = model.project.directory / "simulations" / "by-id" / sid

    parquet_files = list(sim_dir.glob("*.parquet"))
    assert parquet_files, f"No parquet found in {sim_dir}"
    assert len(pd.read_parquet(parquet_files[0])) > 0, "Trajectory parquet is empty"
    assert sid in model.simulations
    assert model.simulations[sid].has_trajectory

    log_file = sim_dir / "stilt.log"
    assert log_file.exists(), "stilt.log missing"
    log_text = log_file.read_text()
    for phrase in ("FATAL ERROR", "Segmentation fault", "hycs_std: not found"):
        assert phrase not in log_text, f"Fatal phrase in log: {phrase!r}"


# ---------------------------------------------------------------------------
# Point receptor - trajectory + footprint
# ---------------------------------------------------------------------------


@integration
def test_footprint(tmp_path, wbb_receptor, wbb_config):
    """Full run produces a readable footprint NetCDF with (time, lat, lon) dims."""
    model = Model(
        project=tmp_path / "footprint",
        config=wbb_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sid = _sim_id(wbb_receptor)
    sim_dir = model.project.directory / "simulations" / "by-id" / sid

    foot_files = list(sim_dir.glob("*_foot.nc"))
    assert foot_files, f"No footprint NetCDF found in {sim_dir}"

    ds = xr.open_dataset(foot_files[0])
    assert {"time", "lat", "lon"} <= set(ds.dims), f"Missing dims in {set(ds.dims)}"
    ds.close()

    assert model.simulations[sid].has_footprint("default")


# ---------------------------------------------------------------------------
# Failure path - missing met files
# ---------------------------------------------------------------------------


@integration
def test_failure_missing_met(tmp_path, wbb_receptor, traj_only_config):
    """Simulation fails gracefully when the met directory is empty."""
    empty_met = tmp_path / "empty_met"
    empty_met.mkdir()

    bad_config = traj_only_config.model_copy(
        update={
            "mets": {
                "hrrr": MetConfig(
                    directory=empty_met,
                    file_format="%Y%m%d.%Hz.hrrra",
                    file_tres="6h",
                )
            }
        }
    )
    model = Model(
        project=tmp_path / "fail_missing_met",
        config=bad_config,
        receptors=[wbb_receptor],
    )
    # run() must not raise: a per-simulation failure is captured, not fatal.
    model.run()

    sid = _sim_id(wbb_receptor)
    # The by-key store has no "failed" state, so the trajectory is simply absent
    # (incomplete). Failure is surfaced through the log-derived Simulation.status.
    assert not model.simulations[sid].has_trajectory
    assert model.simulations[sid].status == "failed:MISSING_MET_FILES"


# ---------------------------------------------------------------------------
# Idempotency - skip_existing=True
# ---------------------------------------------------------------------------


@integration
def test_idempotency(tmp_path, wbb_receptor, traj_only_config):
    """Second run with skip_existing=True does not overwrite existing output."""
    model = Model(
        project=tmp_path / "idempotency",
        config=traj_only_config,
        receptors=[wbb_receptor],
    )

    model.run()
    sid = _sim_id(wbb_receptor)
    assert model.simulations[sid].has_trajectory

    sim_dir = model.project.directory / "simulations" / "by-id" / sid
    parquet = next(sim_dir.glob("*.parquet"))
    mtime_before = parquet.stat().st_mtime

    model.run()  # skip_existing=True is the default
    assert parquet.stat().st_mtime == mtime_before, (
        "Parquet was overwritten on second run"
    )


# ---------------------------------------------------------------------------
# Column receptor - same lat/lon, two heights
# ---------------------------------------------------------------------------


@integration
def test_column(tmp_path, wbb_column_receptor, wbb_config):
    """Column receptor produces trajectory and footprint; sim_id ends with _X."""
    model = Model(
        project=tmp_path / "column",
        config=wbb_config,
        receptors=[wbb_column_receptor],
    )
    model.run()

    sid = _sim_id(wbb_column_receptor)
    assert sid.endswith("_X"), f"Expected column sim_id to end '_X', got {sid!r}"

    sim_dir = model.project.directory / "simulations" / "by-id" / sid
    assert list(sim_dir.glob("*.parquet")), "No trajectory parquet"
    assert list(sim_dir.glob("*_foot.nc")), "No footprint NetCDF"
    assert model.simulations[sid].has_trajectory


# ---------------------------------------------------------------------------
# Multipoint receptor - different lat/lon/zagl
# ---------------------------------------------------------------------------


@integration
def test_multipoint(tmp_path, wbb_multipoint_receptor, multipoint_config):
    """Multipoint receptor (3 locations) produces trajectory and footprint."""
    model = Model(
        project=tmp_path / "multipoint",
        config=multipoint_config,
        receptors=[wbb_multipoint_receptor],
    )
    model.run()

    sid = _sim_id(wbb_multipoint_receptor)
    assert "multi_" in sid, (
        f"Expected multipoint sim_id to contain 'multi_', got {sid!r}"
    )

    sim_dir = model.project.directory / "simulations" / "by-id" / sid
    assert list(sim_dir.glob("*.parquet")), "No trajectory parquet"
    assert list(sim_dir.glob("*_foot.nc")), "No footprint NetCDF"
    assert model.simulations[sid].has_trajectory


# ---------------------------------------------------------------------------
# Multiple footprints - one trajectory, two named configs
# ---------------------------------------------------------------------------


@integration
def test_multifoot(tmp_path, wbb_receptor, multifoot_config):
    """Single trajectory generates two named footprints at different resolutions."""
    model = Model(
        project=tmp_path / "multifoot",
        config=multifoot_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sid = _sim_id(wbb_receptor)
    sim_dir = model.project.directory / "simulations" / "by-id" / sid

    assert list(sim_dir.glob("*_fine_foot.nc")), "No 'fine' footprint NetCDF"
    assert list(sim_dir.glob("*_coarse_foot.nc")), "No 'coarse' footprint NetCDF"
    assert model.simulations[sid].has_footprint("fine")
    assert model.simulations[sid].has_footprint("coarse")


# ---------------------------------------------------------------------------
# CLI - stilt run via CliRunner
# ---------------------------------------------------------------------------


@integration
def test_cli_run(tmp_path, wbb_config, wbb_receptor):
    """CLI `stilt run` produces trajectory and footprint artifacts."""
    from typer.testing import CliRunner

    from stilt.cli import app

    project_dir = tmp_path / "cli_project"
    project_dir.mkdir()

    wbb_config.to_yaml(project_dir / "config.yaml")

    (project_dir / "receptors.csv").write_text(
        "time,lati,long,zagl\n2021-01-15 06:00:00,40.5,-112.0,5.0\n"
    )

    result = CliRunner().invoke(app, ["run", str(project_dir)])

    assert result.exit_code == 0, (
        f"stilt run exited {result.exit_code}\nOutput:\n{result.output}"
    )
    assert "completed=1" in result.output

    sid = _sim_id(wbb_receptor)
    sim_dir = project_dir / "simulations" / "by-id" / sid
    assert list(sim_dir.glob("*.parquet")), "CLI run: no trajectory parquet"
    assert list(sim_dir.glob("*_foot.nc")), "CLI run: no footprint NetCDF"


# ---------------------------------------------------------------------------
# Error trajectory - winderrtf > 0
# ---------------------------------------------------------------------------


@integration
def test_error_trajectory(tmp_path, wbb_receptor, traj_only_config):
    """XY wind error params trigger a second HYSPLIT run; error parquet is saved."""
    error_config = traj_only_config.model_copy(
        update={
            "siguverr": 1.0,
            "tluverr": 60.0,
            "zcoruverr": 500.0,
            "horcoruverr": 40.0,
        }
    )
    model = Model(
        project=tmp_path / "error_traj",
        config=error_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sid = _sim_id(wbb_receptor)
    sim_dir = model.project.directory / "simulations" / "by-id" / sid

    main_files = list(sim_dir.glob("*_traj.parquet"))
    error_files = list(sim_dir.glob("*_error.parquet"))

    assert main_files, "No main trajectory parquet"
    assert error_files, "No error trajectory parquet — winderrtf path not triggered"

    main_traj = pd.read_parquet(main_files[0])
    error_traj = pd.read_parquet(error_files[0])

    assert len(main_traj) > 0, "Main trajectory is empty"
    assert len(error_traj) > 0, "Error trajectory is empty"
    assert set(main_traj.columns) == set(error_traj.columns), (
        "Error trajectory has different columns than main trajectory"
    )
    assert (
        not main_traj["long"]
        .reset_index(drop=True)
        .equals(error_traj["long"].reset_index(drop=True))
    ), "Error trajectory identical to main — wind perturbation had no effect"


# ---------------------------------------------------------------------------
# Geometry-derived footprint config
# ---------------------------------------------------------------------------


@integration
def test_geometry_footprint(tmp_path, wbb_receptor, met_dir):
    """A footprint named by geometry derives its raster, runs, and aggregates."""
    from stilt.config import ModelConfig
    from stilt.footprint import Footprint

    from .fixtures.r_stilt_reference import (
        REFERENCE_KRAND,
        REFERENCE_MET_FILE_FORMAT,
        REFERENCE_SEED,
    )

    # Half-degree windows so particles stay in the derived raster for hours.
    spec = {
        "kind": "windows",
        "coords": [(-112.0, 40.5), (-112.6, 40.9)],
        "size": 0.5,
        "ids": ["wbb", "nw"],
    }
    config = ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": REFERENCE_MET_FILE_FORMAT,
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        footprints={"sources": {"geometry": spec, "cells_per_target": 10}},
    )
    fc = config.footprints["sources"]
    assert fc.grid.xres == fc.grid.yres == 0.05  # 0.5 / 10
    assert fc.geometry_hash

    model = Model(project=tmp_path / "geom", config=config, receptors=[wbb_receptor])
    model.run()

    sid = _sim_id(wbb_receptor)
    sim_dir = model.project.directory / "simulations" / "by-id" / sid
    foot_files = list(sim_dir.glob("*_foot.nc"))
    assert len(foot_files) == 1, foot_files
    foot = Footprint.from_netcdf(foot_files[0])
    assert foot.config.grid == fc.grid
    assert foot.config.geometry == fc.geometry
    assert foot.config.geometry_hash == fc.geometry_hash
    assert model.simulations[sid].has_footprint("sources")

    mesh = fc.geometry.build()
    r_time = pd.Timestamp(wbb_receptor.time)
    bins = pd.interval_range(
        start=r_time - pd.Timedelta(hours=6), end=r_time, freq="1h", closed="left"
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        agg = foot.aggregate(mesh, bins)
    assert agg.index.tolist() == ["wbb", "nw"]
    assert agg.loc["wbb"].sum() > 0  # the receptor sits inside its own window
    assert agg.to_numpy().sum() <= float(foot.data.sum()) + 1e-12


# ---------------------------------------------------------------------------
# Forward run (n_hours > 0)
# ---------------------------------------------------------------------------


@integration
def test_forward_run(tmp_path, met_dir, wbb_grid):
    """A forward simulation runs end to end and carries a forward time axis."""
    from stilt.config import FootprintConfig, ModelConfig
    from stilt.footprint import Footprint
    from stilt.receptors import PointReceptor

    from .fixtures.r_stilt_reference import (
        REFERENCE_ALTITUDE,
        REFERENCE_KRAND,
        REFERENCE_LATITUDE,
        REFERENCE_LONGITUDE,
        REFERENCE_MET_FILE_FORMAT,
        REFERENCE_SEED,
        REFERENCE_SUMMER_TIME,
    )

    # The 2021-07-15 06:00-11:59 block holds the whole +3 h window.
    receptor = PointReceptor(
        REFERENCE_SUMMER_TIME,
        REFERENCE_LONGITUDE,
        REFERENCE_LATITUDE,
        REFERENCE_ALTITUDE,
    )
    config = ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": REFERENCE_MET_FILE_FORMAT,
                "file_tres": "6h",
            }
        },
        n_hours=3,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        hnf_plume=True,  # exercises calc_plume_dilution on a forward track
        footprints={"default": FootprintConfig(grid=wbb_grid)},
    )

    model = Model(project=tmp_path / "forward", config=config, receptors=[receptor])
    model.run()

    sid = _sim_id(receptor)
    sim = model.simulations[sid]
    assert sim.has_trajectory, f"no trajectory for {sid}"

    particles = sim.trajectories.data
    assert len(particles) > 0
    # HYSPLIT reports elapsed minutes signed by run direction
    assert (particles["time"] >= 0).all()
    assert particles["time"].max() > 0
    assert "foot_no_hnf_dilution" in particles.columns

    start, stop = sim.time_range
    assert start == receptor.time
    assert stop == receptor.time + pd.Timedelta(hours=3)

    foot_files = list(sim.directory.glob("*_foot.nc"))
    assert foot_files, f"no footprint NetCDF in {sim.directory}"
    foot = Footprint.from_netcdf(foot_files[0])
    times = pd.DatetimeIndex(foot.data["time"].values)
    # hourly layers running forward, inside the window time_range reports
    assert times.is_monotonic_increasing
    assert times.min() >= pd.Timestamp(start)
    assert times.max() <= pd.Timestamp(stop)
    assert set(times.to_series().diff().dropna()) == {pd.Timedelta(hours=1)}
    assert float(foot.data.sum()) > 0
