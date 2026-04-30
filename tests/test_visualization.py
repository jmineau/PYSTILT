"""Visualization smoke tests — all rendered against the non-interactive Agg backend."""

from __future__ import annotations

import datetime as dt
import uuid

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must come after use("Agg")

from stilt.config import FootprintConfig, Grid, STILTParams
from stilt.footprint import Footprint
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.trajectory import Trajectories
from stilt.visualization import (
    ModelPlotAccessor,
    SimulationPlotAccessor,
    _draw_bounds_box,
    _log10_safe,
    _make_ax,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


@pytest.fixture
def receptor():
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def col_receptor():
    return ColumnReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        bottom=5.0,
        top=50.0,
    )


@pytest.fixture
def multi_receptor():
    return MultiPointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitudes=[-111.85, -111.86, -111.84],
        latitudes=[40.77, 40.78, 40.76],
        altitudes=[5.0, 5.0, 5.0],
    )


@pytest.fixture
def grid():
    return Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=1.0, yres=1.0)


@pytest.fixture
def minimal_trajectories(receptor):
    data = pd.DataFrame(
        {
            "time": [-60.0, -30.0, 0.0],
            "long": [-111.85, -111.90, -111.95],
            "lati": [40.77, 40.75, 40.73],
            "zagl": [100.0, 200.0, 300.0],
            "foot": [0.1, 0.2, 0.3],
        }
    )
    return Trajectories(
        receptor=receptor,
        params=STILTParams(n_hours=-24),
        met_files=[],
        data=data,
    )


@pytest.fixture
def minimal_footprint(receptor, grid):
    times = [dt.datetime(2023, 1, 1, 11), dt.datetime(2023, 1, 1, 12)]
    lons = np.array([-113.5, -112.5, -111.5])
    lats = np.array([39.5, 40.5, 41.5])
    data = xr.DataArray(
        np.random.rand(2, len(lats), len(lons)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
    )
    config = FootprintConfig(grid=grid)
    return Footprint(receptor=receptor, config=config, data=data)


# ---------------------------------------------------------------------------
# _make_ax
# ---------------------------------------------------------------------------


def test_make_ax_no_args():
    fig, ax = _make_ax()
    assert fig is not None
    assert ax is not None


def test_make_ax_with_extent():
    fig, ax = _make_ax(extent=(-115.0, -110.0, 38.0, 43.0))
    assert fig is not None
    assert ax is not None


def test_make_ax_with_existing_ax():
    _, existing = plt.subplots()
    fig, ax = _make_ax(ax=existing)
    assert ax is existing


# ---------------------------------------------------------------------------
# _log10_safe
# ---------------------------------------------------------------------------


def test_log10_safe_positive():
    result = _log10_safe(np.array([1.0, 10.0, 100.0]))
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0])


def test_log10_safe_zero_and_negative_become_nan():
    result = _log10_safe(np.array([0.0, -1.0, 5.0]))
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert not np.isnan(result[2])


# ---------------------------------------------------------------------------
# _draw_bounds_box
# ---------------------------------------------------------------------------


def test_draw_bounds_box_adds_patch(grid):
    _, ax = plt.subplots()
    _draw_bounds_box(ax, grid, label="Domain")
    assert len(ax.patches) == 1


# ---------------------------------------------------------------------------
# ReceptorPlotAccessor
# ---------------------------------------------------------------------------


def test_receptor_map_point_returns_axes(receptor):
    ax = receptor.plot.map()
    assert ax is not None


def test_receptor_map_column_returns_axes(col_receptor):
    ax = col_receptor.plot.map()
    assert ax is not None


def test_receptor_map_multipoint_returns_axes(multi_receptor):
    ax = multi_receptor.plot.map()
    assert ax is not None


def test_receptor_map_with_domain(receptor, grid):
    ax = receptor.plot.map(domain=grid)
    assert ax is not None


def test_receptor_map_met_bounds(receptor, grid):
    ax = receptor.plot.map(met_bounds=grid)
    assert ax is not None


def test_receptor_map_reuses_ax(receptor):
    _, existing = plt.subplots()
    ax = receptor.plot.map(ax=existing)
    assert ax is existing


# ---------------------------------------------------------------------------
# TrajectoriesPlotAccessor
# ---------------------------------------------------------------------------


def test_trajectories_map_default(minimal_trajectories):
    ax = minimal_trajectories.plot.map()
    assert ax is not None


def test_trajectories_map_color_by_zagl(minimal_trajectories):
    ax = minimal_trajectories.plot.map(color_by="zagl")
    assert ax is not None


def test_trajectories_map_color_by_foot(minimal_trajectories):
    ax = minimal_trajectories.plot.map(color_by="foot")
    assert ax is not None


def test_trajectories_map_invalid_color_by_raises(minimal_trajectories):
    with pytest.raises(ValueError, match="color_by"):
        minimal_trajectories.plot.map(color_by="bad_col")


def test_trajectories_map_reuses_ax(minimal_trajectories):
    _, existing = plt.subplots()
    ax = minimal_trajectories.plot.map(ax=existing)
    assert ax is existing


# ---------------------------------------------------------------------------
# FootprintPlotAccessor
# ---------------------------------------------------------------------------


def test_footprint_map_default(minimal_footprint):
    ax = minimal_footprint.plot.map()
    assert ax is not None


def test_footprint_map_no_log(minimal_footprint):
    ax = minimal_footprint.plot.map(log=False)
    assert ax is not None


def test_footprint_map_specific_time(minimal_footprint):
    ax = minimal_footprint.plot.map(time=dt.datetime(2023, 1, 1, 12))
    assert ax is not None


def test_footprint_map_show_grid(minimal_footprint):
    ax = minimal_footprint.plot.map(show_grid=True)
    assert ax is not None


def test_footprint_map_met_bounds(minimal_footprint, grid):
    ax = minimal_footprint.plot.map(met_bounds=grid)
    assert ax is not None


def test_footprint_facet_returns_fig_axes(minimal_footprint):
    fig, axes = minimal_footprint.plot.facet()
    assert fig is not None
    assert axes is not None


def test_footprint_facet_no_log(minimal_footprint):
    fig, axes = minimal_footprint.plot.facet(log=False)
    assert fig is not None


def test_footprint_facet_single_time(receptor, grid):
    """Facet with one time step — nrows=1, unused panels hidden."""
    times = [dt.datetime(2023, 1, 1, 12)]
    lons = np.array([-113.5, -112.5])
    lats = np.array([39.5, 40.5])
    data = xr.DataArray(
        np.random.rand(1, len(lats), len(lons)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
    )
    config = FootprintConfig(grid=grid)
    foot = Footprint(receptor=receptor, config=config, data=data)
    fig, axes = foot.plot.facet(ncols=3)
    assert fig is not None


# ---------------------------------------------------------------------------
# SimulationPlotAccessor
# ---------------------------------------------------------------------------


def test_simulation_map_no_data(receptor):
    """Simulation with no footprint, no trajectories — falls back to receptor extent."""
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = None
    sim.trajectories = None
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_trajectories(receptor, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = None
    sim.trajectories = minimal_trajectories
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_footprint(receptor, minimal_footprint):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = minimal_footprint
    sim.trajectories = None
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_met_bounds(receptor, grid):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = None
    sim.trajectories = None
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map(met_bounds=grid)
    assert ax is not None


def test_simulation_map_show_traj_false(receptor, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = None
    sim.trajectories = minimal_trajectories
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map(show_traj=False)
    assert ax is not None


def test_simulation_map_show_receptor_false(receptor):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.get_footprint.return_value = None
    sim.trajectories = None
    sim.id = "hrrr_202301011200_test"

    ax = SimulationPlotAccessor(sim).map(show_receptor=False)
    assert ax is not None


# ---------------------------------------------------------------------------
# ModelPlotAccessor
# ---------------------------------------------------------------------------


def test_model_availability_empty():
    from unittest.mock import MagicMock

    class _FakeState:
        def sim_ids(self):
            return []

    model = MagicMock()
    model.index = _FakeState()

    ax = ModelPlotAccessor(model).availability()
    assert ax is not None


def test_model_availability_with_sims(tmp_path):
    from stilt.index.sqlite import SqliteIndex
    from stilt.model import Model
    from stilt.simulation import SimID

    repo = SqliteIndex(
        tmp_path,
        db_path=f"file:{uuid.uuid4().hex}?mode=memory&cache=shared",
        uri=True,
    )
    r = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    repo.register([(str(SimID.from_parts("hrrr", r)), r)])

    model = Model(project=tmp_path)
    model._index = repo
    ax = model.plot.availability()
    assert ax is not None


def test_model_availability_reuses_ax():
    from unittest.mock import MagicMock

    class _FakeState:
        def sim_ids(self):
            return []

    _, existing = plt.subplots()
    model = MagicMock()
    model.index = _FakeState()

    ax = ModelPlotAccessor(model).availability(ax=existing)
    assert ax is existing
