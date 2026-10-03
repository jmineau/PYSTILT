"""Tests for stilt.spatial: the footprint grid and the CRS helpers."""

import numpy as np

from stilt import Grid

# ---------------------------------------------------------------------------
# Grid as a geometry
# ---------------------------------------------------------------------------


def test_grid_axes_cells_index_are_consistent():
    grid = Grid(xmin=-114.0, xmax=-113.8, ymin=39.0, ymax=39.35, xres=0.1, yres=0.1)
    x, y = grid.axes
    np.testing.assert_allclose(x, [-113.95, -113.85])
    np.testing.assert_allclose(y, [39.05, 39.15, 39.25])

    cx, cy = grid.cells
    assert cx.shape == cy.shape == (6,)
    np.testing.assert_allclose(cx[:3], [-113.95] * 3)  # x outer, y inner
    np.testing.assert_allclose(cy[:3], [39.05, 39.15, 39.25])

    idx = grid.index
    assert idx.names == ["lon", "lat"]
    np.testing.assert_allclose(idx.get_level_values("lon"), cx)
    assert grid.is_longlat


def test_grid_axes_keep_last_cell_for_inexact_bounds():
    # 40.93 - 40.45 is 0.4799999999999969 in float64; the top row must survive.
    grid = Grid(
        xmin=-112.25, xmax=-111.75, ymin=40.45, ymax=40.93, xres=0.01, yres=0.01
    )
    x, y = grid.axes
    assert (len(x), len(y)) == (50, 48)
    assert y[-1] == 40.925


def test_grid_axes_rounding_keeps_coarse_centres():
    """Rounding strips float noise but never corrupts centres such as 0.5, 1.5."""
    coarse = Grid(xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0, xres=1.0, yres=1.0)
    np.testing.assert_array_equal(coarse.axes[0], [0.5, 1.5])
    fine = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.25, yres=0.25)
    np.testing.assert_array_equal(fine.axes[1], [0.125, 0.375, 0.625, 0.875])


def test_one_longlat_test_for_grids_and_meshes():
    """EPSG:4326 is longitude/latitude for a grid as it is for a mesh."""
    from stilt.footprint.targets import Mesh
    from stilt.spatial import Grid, is_longlat

    grid = Grid(
        xmin=-112.0,
        xmax=-111.0,
        ymin=40.0,
        ymax=41.0,
        xres=0.5,
        yres=0.5,
        crs="EPSG:4326",
    )
    assert is_longlat("EPSG:4326") and is_longlat("+proj=longlat")
    assert not is_longlat("EPSG:32612")
    assert grid.is_longlat
    assert list(grid.index.names) == ["lon", "lat"]
    assert Mesh.from_grid(grid).is_longlat


def test_grid_reads_stilt_r_projection_as_crs():
    """``projection``, STILT-R's name, is read as ``crs``; PYSTILT writes ``crs``."""
    from stilt.spatial import Grid

    kw = {
        "xmin": -112.0,
        "xmax": -111.0,
        "ymin": 40.0,
        "ymax": 41.0,
        "xres": 0.5,
        "yres": 0.5,
    }
    old = Grid(**kw, projection="EPSG:32612")
    assert old == Grid(**kw, crs="EPSG:32612")
    assert "crs" in old.model_dump() and "projection" not in old.model_dump()
