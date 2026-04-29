"""
Integration tests for PYSTILT end-to-end execution.

These tests run real HYSPLIT against real met files and are collected by
default.  They are automatically skipped when the stilt-tutorials met data
is not present (the ``met_dir`` fixture handles this).

Run all integration tests explicitly:
    pytest tests/integration/ -v

Skip them when you only want unit tests:
    pytest tests/ --ignore=tests/integration
    pytest tests/ -m "not integration"
"""

import pandas as pd
import xarray as xr

from stilt.config import MetConfig
from stilt.index import OutputSummary
from stilt.model import Model
from stilt.simulation import SimID

from .conftest import integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_id(receptor, met: str = "hrrr") -> str:
    return str(SimID.from_parts(met, receptor))


def _state_call(model: Model, method: str, *args, **kwargs):
    if method == "output_summaries":
        return model.index.summaries(*args, **kwargs)
    if method == "footprint_complete":
        sim_id, name = args
        summaries = model.index.summaries([sim_id])
        return summaries.get(sim_id, OutputSummary()).footprint_complete(name)
    if method == "trajectory_status":
        [sim_id] = args
        summaries = model.index.summaries([sim_id])
        summary = summaries.get(sim_id, OutputSummary())
        if summary.traj_present:
            return "complete"
        if summary.error_traj_present or summary.log_present:
            return "failed"
        return "pending"
    return getattr(model.index, method)(*args, **kwargs)


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
    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid

    parquet_files = list(sim_dir.glob("*.parquet"))
    assert parquet_files, f"No parquet found in {sim_dir}"
    assert len(pd.read_parquet(parquet_files[0])) > 0, "Trajectory parquet is empty"
    assert sid in _state_call(model, "output_summaries", [sid])
    assert _state_call(model, "trajectory_status", sid) == "complete"

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
    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid

    foot_files = list(sim_dir.glob("*_foot.nc"))
    assert foot_files, f"No footprint NetCDF found in {sim_dir}"

    ds = xr.open_dataset(foot_files[0])
    assert {"time", "lat", "lon"} <= set(ds.dims), f"Missing dims in {set(ds.dims)}"
    ds.close()

    assert _state_call(model, "footprint_complete", sid, "default")


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
    model.run()

    sid = _sim_id(wbb_receptor)
    assert _state_call(model, "trajectory_status", sid) == "failed"


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
    assert _state_call(model, "trajectory_status", sid) == "complete"

    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid
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
def test_column(tmp_path, column_receptor, wbb_config):
    """Column receptor produces trajectory and footprint; sim_id ends with _X."""
    model = Model(
        project=tmp_path / "column",
        config=wbb_config,
        receptors=[column_receptor],
    )
    model.run()

    sid = _sim_id(column_receptor)
    assert sid.endswith("_X"), f"Expected column sim_id to end '_X', got {sid!r}"

    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid
    assert list(sim_dir.glob("*.parquet")), "No trajectory parquet"
    assert list(sim_dir.glob("*_foot.nc")), "No footprint NetCDF"
    assert _state_call(model, "trajectory_status", sid) == "complete"


# ---------------------------------------------------------------------------
# Multipoint receptor - different lat/lon/zagl
# ---------------------------------------------------------------------------


@integration
def test_multipoint(tmp_path, multipoint_receptor, multipoint_config):
    """Multipoint receptor (3 locations) produces trajectory and footprint."""
    model = Model(
        project=tmp_path / "multipoint",
        config=multipoint_config,
        receptors=[multipoint_receptor],
    )
    model.run()

    sid = _sim_id(multipoint_receptor)
    assert "multi_" in sid, (
        f"Expected multipoint sim_id to contain 'multi_', got {sid!r}"
    )

    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid
    assert list(sim_dir.glob("*.parquet")), "No trajectory parquet"
    assert list(sim_dir.glob("*_foot.nc")), "No footprint NetCDF"
    assert _state_call(model, "trajectory_status", sid) == "complete"


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
    sim_dir = model.layout.project_dir / "simulations" / "by-id" / sid

    assert list(sim_dir.glob("*_fine_foot.nc")), "No 'fine' footprint NetCDF"
    assert list(sim_dir.glob("*_coarse_foot.nc")), "No 'coarse' footprint NetCDF"
    assert _state_call(model, "footprint_complete", sid, "fine")
    assert _state_call(model, "footprint_complete", sid, "coarse")


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
        "time,lati,long,zagl\n2015-12-10 00:00:00,40.5,-112.0,5.0\n"
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
