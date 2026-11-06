"""Tests for stilt.spatial: the footprint grid and the CRS helpers."""

import numpy as np
import pandas as pd
import pytest

from stilt import Bounds, Grid
from stilt.spatial import _grid_cell_starts

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


def test_grid_spans_at_most_360_degrees_of_longitude():
    """A grid wider than a turn would hold some longitudes twice, so it is refused (#338)."""
    kw = {"ymin": 40.0, "ymax": 41.0, "xres": 0.5, "yres": 0.5}
    for xmin, xmax in [(-200.0, 200.0), (350.0, -350.0), (10.0, -350.0)]:
        with pytest.raises(ValueError, match="at most 360 degrees"):
            Grid(xmin=xmin, xmax=xmax, **kw)
    assert Grid(xmin=-180.0, xmax=180.0, **kw).xmax == 180.0
    assert Grid(xmin=-10.0, xmax=190.0, **kw).xmin == -10.0
    # Bounds, such as a met's crop box, are not grids and may be wider.
    assert Bounds(xmin=-200.0, xmax=200.0, ymin=40.0, ymax=41.0).xmax == 200.0


def test_grid_axes_run_east_from_xmin():
    """A grid written with xmax west of xmin has the cells of the same box written past 180 (#352)."""
    kw = {"ymin": 40.0, "ymax": 41.0, "xres": 0.5, "yres": 0.5}
    x, _ = Grid(xmin=170.0, xmax=-170.0, **kw).axes
    np.testing.assert_array_equal(x, Grid(xmin=170.0, xmax=190.0, **kw).axes[0])
    assert (x[0], x[-1]) == (170.25, 189.75)


def test_grid_cell_starts_use_complete_half_open_cells():
    starts = _grid_cell_starts(0.0, 1.0, 0.3)

    np.testing.assert_allclose(starts, [0.0, 0.3, 0.6])


def test_grid_cell_starts_keep_decimal_boundary_cell():
    starts = _grid_cell_starts(-113.0, -111.0, 0.01)

    assert len(starts) == 200
    assert starts[0] == pytest.approx(-113.0)
    assert starts[-1] == pytest.approx(-111.01)


@pytest.mark.parametrize("resolution", [0.1, 0.05, 0.01, 0.002])
@pytest.mark.parametrize("base", [40.0, -112.0, -180.0])
def test_grid_cell_starts_keep_last_cell_for_inexact_bounds(base, resolution):
    # Bounds on the 0.01 grid are not exact in binary; the rounding error in
    # ``maximum - minimum`` must not cost the final cell (or the only cell).
    for i in range(100):
        minimum = round(base + i * 0.01, 10)
        for k in range(1, 60):
            maximum = round(minimum + k * resolution, 10)
            assert len(_grid_cell_starts(minimum, maximum, resolution)) == k, (
                minimum,
                maximum,
            )


def test_grid_cell_starts_inexact_bound_examples():
    assert len(_grid_cell_starts(40.45, 40.93, 0.01)) == 48
    np.testing.assert_allclose(_grid_cell_starts(40.0, 40.01, 0.01), [40.0])


def test_grid_cell_starts_still_drop_partial_cell():
    # The tolerance covers float roundoff only, not a real shortfall.
    assert len(_grid_cell_starts(40.45, 40.93 - 1e-8, 0.01)) == 47
    with pytest.raises(ValueError, match="at least one complete cell"):
        _grid_cell_starts(40.0, 40.01 - 1e-8, 0.01)


def test_grid_to_xarray_longlat_centers():
    """Grid.to_xarray() yields CF lon/lat cell centers matching the footprint grid."""
    grid = Grid(xmin=-114.0, xmax=-113.8, ymin=39.0, ymax=39.2, xres=0.1, yres=0.1)
    ds = grid.to_xarray()

    np.testing.assert_allclose(ds["lon"].values, [-113.95, -113.85])
    np.testing.assert_allclose(ds["lat"].values, [39.05, 39.15])
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "crs" in ds
    assert ds["lon"].attrs["standard_name"] == "longitude"
    assert ds["lat"].attrs["units"] == "degrees_north"


# ---------------------------------------------------------------------------
# Bounds.contains: longitudes count modulo 360
# ---------------------------------------------------------------------------


def _inside(xmin, xmax, longitudes, latitude=40.0):
    """Return the longitudes that a box from *xmin* to *xmax* and latitudes 30 to 50 holds."""
    box = Bounds(xmin=xmin, xmax=xmax, ymin=30, ymax=50)
    held = box.contains(longitudes, latitude)
    return [lon for lon, h in zip(longitudes, held, strict=True) if h]


@pytest.mark.parametrize(
    ("xmin", "xmax", "longitudes", "inside"),
    [
        # A box across the dateline or past 180.
        (-125, -100, [-111.9, -90, -130], [-111.9]),
        (170, -170, [175, -175, 0], [175, -175]),
        (170, 190, [175, -175, 185, 0], [175, -175, 185]),
        (-190, -170, [-175, 175, 185], [-175, 175, 185]),
        # A point on an edge is inside, written either way.
        (-125, -100, [-125, -100], [-125, -100]),
        (170, -170, [170, -170, 190, -190], [170, -170, 190, -190]),
        (170, 190, [170, 190, -170, -190], [170, 190, -170, -190]),
        (-190, -170, [-190, -170, 170, 190], [-190, -170, 170, 190]),
        # Just past an edge is outside.
        (170, -170, [169.99, -169.99], []),
        (-125, -100, [-125.01, -99.99], []),
        # A box a turn wide or more holds every longitude, whatever its edges.
        (-180, 180, [-180, 0, 180, 90], [-180, 0, 180, 90]),
        (0, 360, [-180, 0, 180, 359, -1], [-180, 0, 180, 359, -1]),
        (-200, 200, [-180, 0, 180, 195], [-180, 0, 180, 195]),
        (-10, 400, [-180, 0, 180, 195], [-180, 0, 180, 195]),
        # A box that reaches past 360 wraps.
        (350, 370, [355, -355, 5, -5, 0, 10, 11, 349], [355, -355, 5, -5, 0, 10]),
        # A box with no width holds its meridian only.
        (-111, -111, [-111, 249, -110.9], [-111, 249]),
        # West of east by more than half a turn: the box is the long way round.
        (-170, 170, [175, -175, 0, 170, -170], [0, 170, -170]),
    ],
)
def test_bounds_contains_counts_longitudes_modulo_360(xmin, xmax, longitudes, inside):
    assert _inside(xmin, xmax, longitudes) == inside


def test_bounds_contains_allows_rounding_at_an_edge_written_in_another_turn():
    box = Bounds(xmin=-175.3, xmax=-170.1, ymin=30, ymax=50)

    assert box.contains([184.7, -175.3, 189.9], 40.0).all()


def test_bounds_contains_checks_latitude_across_the_dateline():
    box = Bounds(xmin=170, xmax=-170, ymin=30, ymax=50)

    held = box.contains([175, -175, 175, -175], [40, 40, 60, 29.99])

    assert held.tolist() == [True, True, False, False]


def test_bounds_contains_takes_a_scalar_and_a_series():
    box = Bounds(xmin=-114, xmax=-109, ymin=37, ymax=42)

    assert bool(box.contains(-111.9, 40.8))
    assert not box.contains(pd.Series([-111.9, -90.0]), 40.8).all()
    assert box.contains(pd.Series([-111.9, -110.0]), 40.8).all()


def test_bounds_contains_does_not_hold_a_point_with_a_missing_coordinate():
    box = Bounds(xmin=-180, xmax=180, ymin=-90, ymax=90)

    assert box.contains([np.nan, 0.0], [0.0, np.nan]).tolist() == [False, False]


def test_a_grid_holds_points_as_its_bounds_do():
    grid = Grid(xmin=170, xmax=190, ymin=40, ymax=50, xres=1, yres=1)

    assert grid.contains([-175, 175, 0], 45.0).tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("key", "nearest"), [("csr", "crs"), ("xrse", "xres"), ("xmn", "xmin")]
)
def test_grid_refuses_a_key_that_is_no_setting_and_names_the_nearest(key, nearest):
    kw = {
        "xmin": -113,
        "xmax": -110,
        "ymin": 40,
        "ymax": 42,
        "xres": 0.01,
        "yres": 0.01,
    }
    with pytest.raises(
        ValueError, match=rf"'{key}' is not a grid setting.*{nearest!r}"
    ):
        Grid(**kw, **{key: "EPSG:32612"})


def test_grid_refuses_an_unknown_key_in_the_config(tmp_path):
    import stilt

    met = {"directory": str(tmp_path), "file_format": "%Y%m%d", "file_tres": "6h"}
    grid = {
        "xmin": -113,
        "xmax": -110,
        "ymin": 40,
        "ymax": 42,
        "xres": 1000,
        "yres": 1000,
    }
    with pytest.raises(ValueError, match="'csr' is not a grid setting"):
        stilt.ProjectConfig(
            mets={"a": met}, variants={"a": {}}, grid={**grid, "csr": "EPSG:32612"}
        )
    with pytest.raises(ValueError, match="'xresolution' is not a grid setting"):
        stilt.ProjectConfig(
            mets={"a": met}, variants={"a": {"grid": {**grid, "xresolution": 3}}}
        )
