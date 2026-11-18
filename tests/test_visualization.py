"""Visualization smoke tests — all rendered against the non-interactive Agg backend."""

from __future__ import annotations

import datetime as dt

import matplotlib
import matplotlib.legend
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # must come after use("Agg")
from matplotlib.layout_engine import ConstrainedLayoutEngine

from stilt.footprint.config import FootprintConfig
from stilt.releases import ColumnRelease, MultiPointRelease, PointRelease
from stilt.spatial import Grid
from stilt.stilt_visualization import SimulationPlotAccessor, _log10_safe
from stilt.visualization import ProjectPlotAccessor, _draw_bounds_box, _make_ax

from .fixtures.factories import make_project_config
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
def release():
    return PointRelease(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def col_release():
    return ColumnRelease(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        bottom=5.0,
        top=50.0,
    )


@pytest.fixture
def multi_release():
    return MultiPointRelease(
        time=dt.datetime(2023, 1, 1, 12),
        longitudes=[-111.85, -111.86, -111.84],
        latitudes=[40.77, 40.78, 40.76],
        altitudes=[5.0, 5.0, 5.0],
    )


@pytest.fixture
def grid():
    return Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=1.0, yres=1.0)


@pytest.fixture
def minimal_trajectories(release):
    data = pd.DataFrame(
        {
            "age": [-60.0, -30.0, 0.0],
            "lon": [-111.85, -111.90, -111.95],
            "lat": [40.77, 40.75, 40.73],
            "zagl": [100.0, 200.0, 300.0],
            "foot": [0.1, 0.2, 0.3],
        }
    )
    return data


@pytest.fixture
def minimal_footprint(release, grid):
    times = [dt.datetime(2023, 1, 1, 11), dt.datetime(2023, 1, 1, 12)]
    lons = np.array([-113.5, -112.5, -111.5])
    lats = np.array([39.5, 40.5, 41.5])
    data = xr.DataArray(
        np.random.default_rng(0).random((2, len(lats), len(lons))),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
    )
    config = FootprintConfig(grid=grid)
    return as_footprint(data, release, config, "")


def _release_marks(ax):
    """Return the artists the release's map draws: its star, or a multipoint release's scatter."""
    return [c for c in ax.collections if c.get_label() in ("Release", "Releases")]


def _legend_labels(ax):
    """Return the labels in *ax*'s legend, in order (empty when it has none)."""
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def _simulation(release, footprint=None, particles=None):
    """Return a stand-in for a simulation with the given release, footprint and particles."""
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = footprint
    sim.particles = particles
    sim.id = "202301011200_test/hrrr"
    return sim


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


def test_make_ax_has_a_constrained_layout():
    """Each colorbar a map adds gets room, with its label."""
    fig, _ = _make_ax()
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)


def test_make_ax_leaves_the_layout_of_a_given_ax():
    fig, existing = plt.subplots()
    _make_ax(ax=existing)
    assert fig.get_layout_engine() is None


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
# ReleasePlotAccessor
# ---------------------------------------------------------------------------


def test_release_map_point_returns_axes(release):
    ax = release.plot.map()
    assert ax is not None


def test_release_map_column_returns_axes(col_release):
    ax = col_release.plot.map()
    assert ax is not None


def test_release_map_multipoint_returns_axes(multi_release):
    ax = multi_release.plot.map()
    assert ax is not None


def test_release_map_legend_has_a_fixed_place(release, monkeypatch):
    """
    The legend sits in a corner instead of searching for the emptiest spot.

    With loc="best", matplotlib tries every candidate against every point on the
    axes, and warns when that takes over a second, as it does with the particles
    of a simulation.
    """

    def search(self, *args, **kwargs):
        raise AssertionError("the legend searched for its best position")

    monkeypatch.setattr(matplotlib.legend.Legend, "_find_best_position", search)
    fig, ax = plt.subplots()
    release.plot.map(ax=ax)
    fig.canvas.draw()  # the legend is placed at draw time


def test_release_map_column_names_its_height_reference(col_release):
    """A column's heights are labeled with the reference they are given in."""
    for ref in ("agl", "msl"):
        ax = col_release.model_copy(update={"altitude_ref": ref}).plot.map()
        assert any(f"m {ref.upper()}" in t.get_text() for t in ax.texts)
        plt.close("all")


def test_release_map_multipoint_names_its_height_reference(multi_release):
    """The height colorbar names the reference the release's heights are in."""
    for ref in ("agl", "msl"):
        ax = multi_release.model_copy(update={"altitude_ref": ref}).plot.map()
        labels = [a.get_ylabel() for a in ax.figure.axes]
        assert f"Height {ref.upper()} (m)" in labels
        plt.close("all")


def test_release_map_with_domain(release, grid):
    ax = release.plot.map(domain=grid)
    assert ax is not None


def test_release_map_met_bounds(release, grid):
    ax = release.plot.map(met_bounds=grid)
    assert ax is not None


def test_release_map_reuses_ax(release):
    _, existing = plt.subplots()
    ax = release.plot.map(ax=existing)
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


def test_trajectories_map_color_by_foot_without_foot_says_how_to_get_one(
    minimal_trajectories,
):
    particles = minimal_trajectories.drop(columns="foot")
    with pytest.raises(ValueError, match="no 'foot' column.*footprint_particles"):
        particles.stilt.plot.map(color_by="foot")


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


def test_footprint_map_marks_the_release_once(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map()
    assert len(_release_marks(ax)) == 1
    assert _legend_labels(ax) == ["Release"]


def test_footprint_map_show_release_false(minimal_footprint):
    ax = minimal_footprint.stilt.plot.map(show_release=False)
    assert _release_marks(ax) == []
    assert _legend_labels(ax) == []


def test_footprint_map_show_release_false_keeps_the_legend_of_the_outlines(
    minimal_footprint, grid
):
    ax = minimal_footprint.stilt.plot.map(
        show_release=False, show_grid=True, met_bounds=grid
    )
    assert _release_marks(ax) == []
    assert _legend_labels(ax) == ["Domain", "Met domain"]


def test_footprint_facet_returns_fig_axes(minimal_footprint):
    fig, axes = minimal_footprint.stilt.plot.facet()
    assert fig is not None
    assert axes is not None


def test_footprint_facet_no_log(minimal_footprint):
    fig, axes = minimal_footprint.stilt.plot.facet(log=False)
    assert fig is not None


def test_footprint_facet_single_time(release, grid):
    """Facet with one time step — nrows=1, unused panels hidden."""
    times = [dt.datetime(2023, 1, 1, 12)]
    lons = np.array([-113.5, -112.5])
    lats = np.array([39.5, 40.5])
    data = xr.DataArray(
        np.random.default_rng(0).random((1, len(lats), len(lons))),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
    )
    config = FootprintConfig(grid=grid)
    foot = as_footprint(data, release, config, "")
    fig, axes = foot.stilt.plot.facet(ncols=3)
    assert fig is not None


@pytest.fixture
def projected_footprint(release):
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
        np.random.default_rng(0).random((2, len(y), len(x))),
        dims=("time", "y", "x"),
        coords={
            "time": [dt.datetime(2023, 1, 1, 11), dt.datetime(2023, 1, 1, 12)],
            "y": y,
            "x": x,
        },
    )
    return as_footprint(data, release, FootprintConfig(grid=grid), "")


def test_footprint_map_on_a_projected_grid(projected_footprint):
    ax = projected_footprint.stilt.plot.map()
    assert ax is not None
    plt.close("all")


def test_footprint_facet_on_a_projected_grid(projected_footprint):
    fig, axes = projected_footprint.stilt.plot.facet()
    assert fig is not None
    plt.close("all")


@pytest.fixture
def hourly_footprint(release, grid):
    """Thirteen hourly steps from 19:00 to 07:00, so the panels cross midnight."""
    times = pd.date_range("2023-01-01 19:00", periods=13, freq="h")
    lons = np.array([-113.5, -112.5, -111.5])
    lats = np.array([39.5, 40.5, 41.5])
    data = xr.DataArray(
        np.random.default_rng(0).random((len(times), len(lats), len(lons))),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return as_footprint(data, release, FootprintConfig(grid=grid), "")


def _colorbar_axes(fig, axes):
    [cbar_ax] = [ax for ax in fig.axes if ax not in axes.flat]
    return cbar_ax


def test_footprint_facet_titles_each_panel_with_its_clock_time(hourly_footprint):
    fig, axes = hourly_footprint.stilt.plot.facet()
    titles = [ax.get_title() for ax in axes.flat][:13]
    assert titles[0] == "2023-01-01 19:00"
    assert titles[5] == "2023-01-02 00:00"  # xarray titles midnight with no hour
    assert titles[-1] == "2023-01-02 07:00"


def test_footprint_facet_has_no_title_over_the_panels(hourly_footprint):
    fig, _ = hourly_footprint.stilt.plot.facet()
    assert [text.get_text() for text in fig.texts] == []


def test_footprint_facet_is_sized_by_its_panels(hourly_footprint):
    """Each row of panels is 2.2 inches high, whatever the number of panels."""
    fig, _ = hourly_footprint.stilt.plot.facet(ncols=4)  # 13 panels: 4 rows
    assert fig.get_figheight() == pytest.approx(4 * 2.2)


def test_footprint_facet_figsize_wins(hourly_footprint):
    fig, _ = hourly_footprint.stilt.plot.facet(figsize=(6, 5))
    assert tuple(fig.get_size_inches()) == (6, 5)


def test_footprint_facet_colorbar_is_slim(hourly_footprint):
    """The colorbar was as tall as the figure and 0.7 inch wide for thirteen panels."""
    fig, axes = hourly_footprint.stilt.plot.facet()
    fig.canvas.draw()  # the colorbar's aspect is applied when it is drawn
    box = _colorbar_axes(fig, axes).get_position()
    assert box.width * fig.get_figwidth() < 0.4
    assert box.height < 0.7


def test_footprint_facet_merges_cbar_kwargs(hourly_footprint):
    fig, axes = hourly_footprint.stilt.plot.facet(cbar_kwargs={"label": "ppm"})
    assert _colorbar_axes(fig, axes).get_ylabel() == "ppm"


# ---------------------------------------------------------------------------
# SimulationPlotAccessor
# ---------------------------------------------------------------------------


def test_simulation_map_no_data(release):
    """Simulation with no footprint, no trajectories — falls back to release extent."""
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_trajectories(release, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = minimal_trajectories
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_with_footprint(release, minimal_footprint):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = minimal_footprint
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()
    assert ax is not None


def test_simulation_map_draws_the_tiler_under_a_new_map(release):
    pytest.importorskip("cartopy")
    from unittest.mock import MagicMock

    class Tiler:
        def image_for_domain(self, domain, zoom):
            raise AssertionError("the map is not drawn in this test")

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"
    tiler = Tiler()

    ax = SimulationPlotAccessor(sim).map(tiler=tiler, tiler_zoom=5)

    assert ax.img_factories == [[tiler, (5,), {}]]  # cartopy draws it with the map


def test_simulation_map_met_bounds(release, grid):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(met_bounds=grid)
    assert ax is not None


def test_simulation_map_show_traj_false(release, minimal_trajectories):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = minimal_trajectories
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(show_particles=False)
    assert ax is not None


def test_simulation_map_colorbar_labels_do_not_overlap(
    release, minimal_footprint, minimal_trajectories
):
    """The particles' colorbar label was drawn over the footprint's colorbar."""
    pytest.importorskip("cartopy")
    from unittest.mock import MagicMock

    from cartopy.mpl.feature_artist import FeatureArtist

    sim = MagicMock()
    sim.release = release
    sim.footprint = minimal_footprint
    sim.particles = minimal_trajectories
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()

    fig = ax.figure
    for artist in ax.get_children():  # drawing the coastlines would download them
        if isinstance(artist, FeatureArtist):
            artist.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bars = [other for other in fig.axes if other is not ax]
    assert len(bars) == 2
    for bar, other in (bars, bars[::-1]):
        label = bar.yaxis.label.get_window_extent(renderer)
        assert not label.overlaps(other.get_window_extent(renderer))


def test_simulation_map_show_release_false(release):
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = release
    sim.footprint = None
    sim.particles = None
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map(show_release=False)
    assert ax is not None


def test_simulation_map_marks_the_release_once(
    release, minimal_footprint, minimal_trajectories
):
    """The footprint's map and the simulation's both mark the release; only one may."""
    sim = _simulation(release, minimal_footprint, minimal_trajectories)
    ax = SimulationPlotAccessor(sim).map()
    assert len(_release_marks(ax)) == 1
    assert _legend_labels(ax) == ["Domain", "Release"]


def test_simulation_map_show_release_false_leaves_the_release_off(
    release, minimal_footprint, minimal_trajectories
):
    sim = _simulation(release, minimal_footprint, minimal_trajectories)
    ax = SimulationPlotAccessor(sim).map(show_release=False)
    assert _release_marks(ax) == []
    assert _legend_labels(ax) == ["Domain"]


def test_simulation_map_without_a_footprint_marks_the_release_once(
    release, minimal_trajectories
):
    sim = _simulation(release, particles=minimal_trajectories)
    ax = SimulationPlotAccessor(sim).map()
    assert len(_release_marks(ax)) == 1
    assert _legend_labels(ax) == ["Release"]


def test_simulation_map_multipoint_release_has_one_height_colorbar(multi_release, grid):
    times = [dt.datetime(2023, 1, 1, 12)]
    data = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [39.5, 40.5], "lon": [-113.5, -112.5]},
    )
    foot = as_footprint(data, multi_release, FootprintConfig(grid=grid), "")
    ax = SimulationPlotAccessor(_simulation(multi_release, foot)).map()
    label = f"Height {multi_release.altitude_ref.upper()} (m)"
    assert [a.get_ylabel() for a in ax.figure.axes].count(label) == 1
    assert len(_release_marks(ax)) == 1


# ---------------------------------------------------------------------------
# ProjectPlotAccessor
# ---------------------------------------------------------------------------


def test_project_availability_empty():
    from unittest.mock import MagicMock

    project = MagicMock()
    project.releases = pd.DataFrame(columns=["location", "time"])

    ax = ProjectPlotAccessor(project).availability()
    assert ax is not None


def test_project_availability_with_sims(tmp_path, release):
    from stilt import Project

    config = make_project_config(tmp_path)
    project = Project.init(tmp_path, config=config, releases=[release])
    assert len(project.simulations) == 1

    ax = project.plot.availability()
    assert ax is not None
    [bars] = ax.collections
    assert len(bars.get_paths()) == 1


def test_project_availability_bins_releases_by_day_and_location(release):
    from unittest.mock import MagicMock

    t = pd.Timestamp(release.time)
    project = MagicMock()
    project.releases = pd.DataFrame(
        {
            "location": ["a", "a", "a", "b"],
            "time": [t, t + pd.Timedelta("2h"), t + pd.Timedelta("1D"), t],
        }
    )

    ax = ProjectPlotAccessor(project).availability()

    a, b = ax.collections[:2]
    assert [len(a.get_paths()), len(b.get_paths())] == [2, 1]  # days with releases
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b"]


def test_project_availability_reuses_ax():
    from unittest.mock import MagicMock

    _, existing = plt.subplots()
    project = MagicMock()
    project.releases = pd.DataFrame(columns=["location", "time"])

    ax = ProjectPlotAccessor(project).availability(ax=existing)
    assert ax is existing


def test_project_availability_formats_the_figure_of_the_given_ax(release):
    """The dates are formatted on the ax's figure, not the current figure."""
    from unittest.mock import MagicMock

    fig, existing = plt.subplots()
    other = plt.figure()  # now the current figure
    default_bottom = other.subplotpars.bottom
    project = MagicMock()
    project.releases = pd.DataFrame(
        {"location": [release.location_id], "time": [release.time]}
    )

    ProjectPlotAccessor(project).availability(ax=existing)

    assert fig.subplotpars.bottom == pytest.approx(0.2)  # set by autofmt_xdate
    assert other.subplotpars.bottom == default_bottom


def test_project_availability_figure_keeps_long_location_labels(release):
    """A label of any length stays inside the figure, so a saved figure has it whole."""
    from unittest.mock import MagicMock

    project = MagicMock()
    project.releases = pd.DataFrame(
        {"location": ["-111.848_40.766_10-with-a-long-name"], "time": [release.time]}
    )

    ax = ProjectPlotAccessor(project).availability()

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*ax.get_yticklabels(), ax.yaxis.label]:
        assert text.get_window_extent(renderer).x0 >= 0


def test_project_availability_rotates_the_dates_of_its_own_figure(release):
    from unittest.mock import MagicMock

    project = MagicMock()
    project.releases = pd.DataFrame(
        {"location": [release.location_id], "time": [release.time]}
    )

    ax = ProjectPlotAccessor(project).availability()

    assert {label.get_rotation() for label in ax.get_xticklabels()} == {30.0}


# ---------------------------------------------------------------------------
# Maps across 180 degrees
# ---------------------------------------------------------------------------


@pytest.fixture
def dateline_footprint():
    """A footprint on a grid from 170 to 190 degrees east, written with the grid's own longitudes."""
    release = PointRelease(
        time=dt.datetime(2023, 1, 1, 12), longitude=-179.5, latitude=45.0, altitude=5.0
    )
    grid = Grid(xmin=170.0, xmax=190.0, ymin=40.0, ymax=50.0, xres=1.0, yres=1.0)
    lons = np.arange(170.5, 190.0, 1.0)
    lats = np.arange(40.5, 50.0, 1.0)
    data = xr.DataArray(
        np.random.default_rng(0).random((1, len(lats), len(lons))),
        dims=("time", "lat", "lon"),
        coords={"time": [dt.datetime(2023, 1, 1, 12)], "lat": lats, "lon": lons},
    )
    return as_footprint(data, release, FootprintConfig(grid=grid), "")


def _width_of_map(ax):
    """Return the width in degrees of the longitudes a map shows, in its own coordinates (cartopy widens it to fill the axes)."""
    west, east = ax.get_xlim()
    return east - west


def test_central_longitude_is_zero_unless_the_extent_runs_past_180():
    from stilt.visualization import _central_longitude

    assert _central_longitude(None) == 0.0
    assert _central_longitude((-114.0, -111.0, 39.0, 42.0)) == 0.0
    assert _central_longitude((-180.0, 180.0, -90.0, 90.0)) == 0.0
    assert _central_longitude((169.95, 190.05, 40.0, 50.0)) == pytest.approx(180.0)
    assert _central_longitude((-190.05, -169.95, 40.0, 50.0)) == pytest.approx(-180.0)


def test_footprint_map_across_180_shows_only_the_grid(dateline_footprint):
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs

    ax = dateline_footprint.stilt.plot.map(show_grid=True)

    assert isinstance(ax.projection, ccrs.PlateCarree)
    assert ax.projection.proj4_params["pm"] == pytest.approx(180.0)
    assert 19.0 <= _width_of_map(ax) < 25.0  # the grid, not the globe's 360


def test_footprint_map_across_180_without_cartopy_shows_only_the_grid(
    dateline_footprint, monkeypatch
):
    monkeypatch.setitem(__import__("sys").modules, "cartopy", None)
    monkeypatch.setitem(__import__("sys").modules, "cartopy.crs", None)

    ax = dateline_footprint.stilt.plot.map()

    assert ax.get_xlim() == pytest.approx((170.45, 189.55))


def test_footprint_map_keeps_the_prime_meridian_center_elsewhere(minimal_footprint):
    pytest.importorskip("cartopy")

    ax = minimal_footprint.stilt.plot.map()

    assert ax.projection.proj4_params["pm"] == 0
    assert _width_of_map(ax) < 10.0


def test_release_map_across_180_fits_the_domain(dateline_footprint):
    pytest.importorskip("cartopy")
    release = dateline_footprint.stilt.release

    ax = release.plot.map(domain=dateline_footprint.stilt.grid)

    assert ax.projection.proj4_params["pm"] == pytest.approx(180.0)
    assert 20.0 <= _width_of_map(ax) < 25.0
    # the release at -179.5 is drawn 0.5 degrees past 180: inside the map
    (star,) = [c for c in ax.collections if c.get_label() == "Release"]
    assert star.get_transform() is not ax.transData


def test_simulation_map_across_180_fits_the_footprint(dateline_footprint):
    pytest.importorskip("cartopy")
    from unittest.mock import MagicMock

    sim = MagicMock()
    sim.release = dateline_footprint.stilt.release
    sim.footprint = dateline_footprint
    sim.particles = pd.DataFrame(
        {
            "age": [-60.0, -30.0, 0.0],
            "lon": [-179.9, 179.9, 175.0],
            "lat": [45.0, 45.1, 45.2],
            "zagl": [10.0, 20.0, 30.0],
            "foot": [0.1, 0.2, 0.3],
        }
    )
    sim.id = "202301011200_test/hrrr"

    ax = SimulationPlotAccessor(sim).map()

    assert ax.projection.proj4_params["pm"] == pytest.approx(180.0)
    assert 20.0 <= _width_of_map(ax) < 25.0
