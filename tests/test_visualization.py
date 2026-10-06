"""Visualization smoke tests — all rendered against the non-interactive Agg backend."""

from __future__ import annotations

import datetime as dt

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must come after use("Agg")

from stilt.footprint.config import FootprintConfig
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.spatial import Grid
from stilt.visualization import (
    ProjectPlotAccessor,
    SimulationPlotAccessor,
    _draw_bounds_box,
    _log10_safe,
    _make_ax,
)

from .fixtures.footprints import as_footprint

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
    return data


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
    return as_footprint(data, receptor, config, "")


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
# ParticlesPlotAccessor
# ---------------------------------------------------------------------------


def test_trajectories_map_default(minimal_trajectories):
    ax = minimal_trajectories.stilt.plot.map()
    assert ax is not None


def test_trajectories_map_color_by_zagl(minimal_trajectories):
    ax = minimal_trajectories.stilt.plot.map(color_by="zagl")
    assert ax is not None


def test_trajectories_map_color_by_foot(minimal_trajectories):
    ax = minimal_trajectories.stilt.plot.map(color_by="foot")
    assert ax is not None


def test_trajectories_map_invalid_color_by_raises(minimal_trajectories):
    with pytest.raises(ValueError, match="color_by"):
        minimal_trajectories.stilt.plot.map(color_by="bad_col")


def test_trajectories_map_reuses_ax(minimal_trajectories):
    _, existing = plt.subplots()
    ax = minimal_trajectories.stilt.plot.map(ax=existing)
    assert ax is existing


# ---------------------------------------------------------------------------
# FootprintPlotAccessor
# ---------------------------------------------------------------------------


def test_footprint_map_default(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map()
    assert ax is not None


def test_footprint_map_no_log(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map(log=False)
    assert ax is not None


def test_footprint_map_specific_time(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map(time=dt.datetime(2023, 1, 1, 12))
    assert ax is not None


def test_footprint_map_show_grid(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map(show_grid=True)
    assert ax is not None


def test_footprint_map_met_bounds(minimal_footprint, grid):
    ax = minimal_footprint.stilt.plot.map(met_bounds=grid)
    assert ax is not None


def test_footprint_facet_returns_fig_axes(minimal_footprint):
    fig, axes = minimal_footprint.stilt.plot.facet()
    assert fig is not None
    assert axes is not None


def test_footprint_facet_no_log(minimal_footprint):
    fig, axes = minimal_footprint.stilt.plot.facet(log=False)
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
    foot = as_footprint(data, receptor, config, "")
    fig, axes = foot.stilt.plot.facet(ncols=3)
    assert fig is not None


@pytest.fixture
def projected_footprint(receptor):
    """A two-step footprint on a 10 km UTM grid (x and y, not lon and lat)."""
    grid = Grid(
        xmin=-112.3,
        xmax=-111.4,
        ymin=40.4,
        ymax=41.1,
        xres=10000.0,
        yres=10000.0,
        crs="EPSG:32612",
    )
    x, y = grid.axes
    data = xr.DataArray(
        np.random.rand(2, len(y), len(x)),
        dims=("time", "y", "x"),
        coords={
            "time": [dt.datetime(2023, 1, 1, 11), dt.datetime(2023, 1, 1, 12)],
            "y": y,
            "x": x,
        },
    )
    return as_footprint(data, receptor, FootprintConfig(grid=grid), "")


def test_footprint_map_on_a_projected_grid(projected_footprint):
    ax = projected_footprint.stilt.plot.map()
    assert ax is not None
    plt.close("all")


def test_footprint_facet_on_a_projected_grid(projected_footprint):
    fig, axes = projected_footprint.stilt.plot.facet()
    assert fig is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# SimulationPlotAccessor
# ---------------------------------------------------------------------------


def test_simulation_map_no_data(receptor):
    """Simulation with no footprint, no trajectories — falls back to receptor extent."""
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_trajectories(receptor, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = None
    sim.particles = minimal_trajectories
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_footprint(receptor, minimal_footprint):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = minimal_footprint
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_met_bounds(receptor, grid):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(met_bounds=grid)
    assert ax is not None


def test_simulation_map_show_traj_false(receptor, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = None
    sim.particles = minimal_trajectories
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(show_particles=False)
    assert ax is not None


def test_simulation_map_show_receptor_false(receptor):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.receptor = receptor
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(show_receptor=False)
    assert ax is not None


# ---------------------------------------------------------------------------
# ProjectPlotAccessor
# ---------------------------------------------------------------------------


def test_project_availability_empty():
    from unittest.mock import MagicMock

    project = MagicMock()
    project.receptors = pd.DataFrame(columns=["location", "time"])

    ax = ProjectPlotAccessor(project).availability()
    assert ax is not None


def test_project_availability_with_sims(tmp_path, receptor):
    from stilt.config import ProjectConfig
    from stilt.project import Project

    config = ProjectConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
        variants={"hrrr": {}},
    )
    project = Project.init(tmp_path, config=config, receptors=[receptor])
    assert len(project.simulations) == 1

    ax = project.plot.availability()
    assert ax is not None
    assert len(ax.patches) == 1


def test_project_availability_reuses_ax():
    from unittest.mock import MagicMock

    _, existing = plt.subplots()
    project = MagicMock()
    project.receptors = pd.DataFrame(columns=["location", "time"])

    ax = ProjectPlotAccessor(project).availability(ax=existing)
    assert ax is existing


def test_project_availability_formats_the_figure_of_the_given_ax(receptor):
    """The dates are formatted on the ax's figure, not the current figure."""
    from unittest.mock import MagicMock

    fig, existing = plt.subplots()
    other = plt.figure()  # now the current figure
    default_bottom = other.subplotpars.bottom
    project = MagicMock()
    project.receptors = pd.DataFrame(
        {"location": [receptor.location_id], "time": [receptor.time]}
    )

    ProjectPlotAccessor(project).availability(ax=existing)

    assert fig.subplotpars.bottom == pytest.approx(0.2)  # set by autofmt_xdate
    assert other.subplotpars.bottom == default_bottom
