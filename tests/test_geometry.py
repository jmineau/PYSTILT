"""Tests for spatial geometries: Grid derivation, Mesh, Zones, overlap weights."""

import numpy as np
import pytest
import shapely

from stilt import Grid, Mesh, Zones
from stilt.geometry import overlap_weights, same_crs

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
    assert grid.min_cell_width == 0.1


def test_grid_axes_keep_last_cell_for_inexact_bounds():
    # 40.93 - 40.45 is 0.4799999999999969 in float64; the top row must survive.
    grid = Grid(
        xmin=-112.25, xmax=-111.75, ymin=40.45, ymax=40.93, xres=0.01, yres=0.01
    )
    x, y = grid.axes
    assert (len(x), len(y)) == (50, 48)
    assert y[-1] == 40.925


def test_grid_from_geometry_derives_resolution_and_snapped_bounds():
    # Smallest cell 0.0095 wide / 4 = 0.002375 -> rounded down to 0.002
    mesh = Mesh.from_windows([(-111.9, 40.7), (-112.05, 40.6)], (0.0095, 0.03))
    grid = Grid.from_geometry(mesh)
    assert grid.xres == grid.yres == 0.002
    assert grid.projection == "+proj=longlat"
    xmin, ymin, xmax, ymax = mesh.bounds
    assert grid.xmin <= xmin and grid.xmax >= xmax
    assert grid.ymin <= ymin and grid.ymax >= ymax
    # snapped to whole cells
    assert grid.xmin == pytest.approx(round(grid.xmin / 0.002) * 0.002)
    # and the derived raster resolves the smallest cell by >= 4 cells
    assert mesh.min_cell_width / grid.xres >= 4


def test_grid_from_geometry_cells_per_target_knob():
    mesh = Mesh.from_windows([(0.5, 0.5)], 0.1)
    assert Grid.from_geometry(mesh, cells_per_target=10).xres == pytest.approx(0.01)
    assert Grid.from_geometry(mesh, cells_per_target=2).xres == pytest.approx(0.05)


def test_grid_from_geometries_takes_union_and_finest():
    a = Mesh.from_windows([(0.5, 0.5)], 0.1)
    b = Mesh.from_windows([(2.0, 2.0)], 0.4)
    grid = Grid.from_geometries([a, b])
    assert grid.xres == pytest.approx(0.02)
    assert grid.xmin <= 0.45 and grid.xmax >= 2.2


def test_grid_from_geometry_warns_when_huge():
    mesh = Mesh.from_windows([(0.0, 0.0), (50.0, 50.0)], 0.001)
    with pytest.warns(UserWarning, match="cells"):
        Grid.from_geometry(mesh, max_cells=1000)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


def test_mesh_from_windows_ids_index_bounds():
    mesh = Mesh.from_windows(
        [(-111.97, 40.515), (-112.015, 40.779)], 0.01, ids=["landfill", "wwtp"]
    )
    assert len(mesh) == 2
    assert mesh.index.tolist() == ["landfill", "wwtp"]
    assert mesh.index.name == "cell"
    assert mesh.is_longlat
    xmin, ymin, xmax, ymax = mesh.bounds
    assert xmin == pytest.approx(-112.02) and ymax == pytest.approx(40.784)
    assert mesh.min_cell_width == pytest.approx(0.01)
    assert len(mesh.hash) == 10


def test_mesh_default_ids_and_hash_changes_with_geometry():
    a = Mesh.from_windows([(1.0, 2.0)], 1.0)
    b = Mesh.from_windows([(1.0, 2.5)], 1.0)
    assert a.ids == ("1,2",)
    assert a.hash != b.hash


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(ids=(), geometries=(), crs="+proj=longlat"), "at least one"),
        (dict(ids=("a", "b"), geometries=(shapely.box(0, 0, 1, 1),)), "same length"),
        (
            dict(ids=("a", "a"), geometries=(shapely.box(0, 0, 1, 1),) * 2),
            "unique",
        ),
        (dict(ids=("a",), geometries=(shapely.Point(0, 0),)), "Polygon"),
    ],
)
def test_mesh_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Mesh(**kwargs)


def test_mesh_from_geodataframe_uses_column_and_crs():
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    mesh = Mesh.from_geodataframe(gdf, ids="name")
    assert mesh.ids == ("a", "b")
    assert mesh.is_longlat
    assert same_crs(mesh.crs, "+proj=longlat")


def test_mesh_from_grid_matches_grid_index_order():
    grid = Grid(xmin=0.0, xmax=0.2, ymin=0.0, ymax=0.3, xres=0.1, yres=0.1)
    mesh = Mesh.from_grid(grid)
    assert len(mesh) == 6
    cx, cy = grid.cells
    centroids = shapely.centroid(np.asarray(mesh.geometries, dtype=object))
    np.testing.assert_allclose(shapely.get_x(centroids), cx)
    np.testing.assert_allclose(shapely.get_y(centroids), cy)


def test_mesh_to_crs_roundtrip():
    pytest.importorskip("pyproj")
    mesh = Mesh.from_windows([(-111.9, 40.7)], 0.01)
    utm = mesh.to_crs("EPSG:32612")
    assert not utm.is_longlat
    back = utm.to_crs("EPSG:4326")
    np.testing.assert_allclose(back.bounds, mesh.bounds, atol=1e-6)


def test_mesh_from_h3_requires_h3():
    h3 = pytest.importorskip("h3")
    mesh = Mesh.from_h3(7, (-112.1, 40.5, -111.8, 40.8))
    assert len(mesh) > 10
    assert all(h3.is_valid_cell(c) for c in mesh.ids)
    assert mesh.is_longlat


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


def test_zones_index_and_membership():
    grid = Grid(xmin=0.0, xmax=0.2, ymin=0.0, ymax=0.2, xres=0.1, yres=0.1)
    part = Zones.from_labels(grid, ["A", "A", "B", "A"])
    assert part.index.tolist() == ["A", "B"]
    m = part.membership.toarray()
    np.testing.assert_array_equal(m, [[1, 1, 0, 1], [0, 0, 1, 0]])
    assert part.min_cell_width == 0.1
    assert part.is_longlat


def test_zones_reject_wrong_label_count():
    grid = Grid(xmin=0.0, xmax=0.2, ymin=0.0, ymax=0.2, xres=0.1, yres=0.1)
    with pytest.raises(ValueError, match="one entry per base cell"):
        Zones.from_labels(grid, ["A"])


# ---------------------------------------------------------------------------
# overlap_weights
# ---------------------------------------------------------------------------


def _raster(n: int, res: float):
    centers = (np.arange(n) + 0.5) * res
    return centers, centers, res, res


def test_grid_weights_block_sum_partition_of_unity():
    x, y, xres, yres = _raster(6, 0.01)
    coarse = Grid(xmin=0.0, xmax=0.06, ymin=0.0, ymax=0.06, xres=0.03, yres=0.03)
    w = overlap_weights(coarse, x, y, xres, yres, "+proj=longlat")
    assert w.shape == (4, 36)
    # every native cell fully inside exactly one coarse cell
    np.testing.assert_allclose(np.asarray(w.sum(axis=0)).ravel(), 1.0)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 9.0)


def test_grid_weights_row_order_matches_grid_index():
    x, y, xres, yres = _raster(4, 0.5)  # native 4x4 of 0.5 over [0, 2]
    coarse = Grid(xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0, xres=1.0, yres=1.0)
    w = overlap_weights(coarse, x, y, xres, yres, "+proj=longlat").toarray()
    cx, cy = coarse.cells
    # native flatten is y-outer/x-inner: idx = iy*4 + ix
    for row, (tx, ty) in enumerate(zip(cx, cy, strict=True)):
        ix = np.where(np.abs(x - tx) < 0.5)[0]
        iy = np.where(np.abs(y - ty) < 0.5)[0]
        expected = np.zeros(16)
        for j in iy:
            for i in ix:
                expected[j * 4 + i] = 1.0
        np.testing.assert_allclose(w[row], expected)


def test_mesh_weights_equal_grid_weights_for_same_boxes():
    x, y, xres, yres = _raster(6, 0.01)
    coarse = Grid(xmin=0.0, xmax=0.06, ymin=0.0, ymax=0.06, xres=0.03, yres=0.03)
    w_grid = overlap_weights(coarse, x, y, xres, yres, "+proj=longlat").toarray()
    w_mesh = overlap_weights(
        Mesh.from_grid(coarse), x, y, xres, yres, "+proj=longlat"
    ).toarray()
    np.testing.assert_allclose(w_mesh, w_grid, atol=1e-12)


def test_mesh_weights_triangle_area_fractions():
    x, y, xres, yres = _raster(2, 1.0)  # 2x2 unit cells over [0, 2]
    tri = Mesh(ids=("t",), geometries=(shapely.Polygon([(0, 0), (2, 0), (0, 2)]),))
    w = overlap_weights(tri, x, y, xres, yres, "+proj=longlat").toarray()
    # cells: (0,0) full, (1,0) half, (0,1) half, (1,1) none; flatten iy*2+ix
    np.testing.assert_allclose(w[0], [1.0, 0.5, 0.5, 0.0])


def test_zones_weights_merge_rows():
    x, y, xres, yres = _raster(6, 0.01)
    coarse = Grid(xmin=0.0, xmax=0.06, ymin=0.0, ymax=0.06, xres=0.03, yres=0.03)
    part = Zones.from_labels(coarse, ["west", "west", "east", "east"])
    w = overlap_weights(part, x, y, xres, yres, "+proj=longlat")
    assert w.shape == (2, 36)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), [18.0, 18.0])


def test_overlap_weights_are_cached():
    x, y, xres, yres = _raster(3, 1.0)
    mesh = Mesh.from_windows([(1.5, 1.5)], 1.0)
    a = overlap_weights(mesh, x, y, xres, yres, "+proj=longlat")
    b = overlap_weights(mesh, x, y, xres, yres, "+proj=longlat")
    assert a is b


def test_overlap_weights_reproject_mesh_to_raster_crs():
    pytest.importorskip("pyproj")
    # 1 km UTM raster; a lon/lat window 0.01 deg wide (~840 m x 1110 m at 40.7N)
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    cx, cy = tr.transform(-111.9, 40.7)
    x = cx + (np.arange(-3, 4)) * 1000.0
    y = cy + (np.arange(-3, 4)) * 1000.0
    mesh = Mesh.from_windows([(-111.9, 40.7)], 0.01)
    w = overlap_weights(mesh, x, y, 1000.0, 1000.0, "EPSG:32612")
    total_frac = w.sum()  # native-cell fractions -> window area in km^2
    assert 0.7 < total_frac < 1.1


# ---------------------------------------------------------------------------
# Remaining paths: Zones over a Mesh, projected from_geometry, from_file
# ---------------------------------------------------------------------------


def test_zones_over_mesh_merge_polygon_rows():
    x, y, xres, yres = _raster(4, 1.0)  # 4x4 unit cells over [0, 4]
    mesh = Mesh(
        ids=("a", "b", "c"),
        geometries=(
            shapely.box(0, 0, 2, 4),
            shapely.box(2, 0, 4, 2),
            shapely.box(2, 2, 4, 4),
        ),
    )
    zones = Zones.from_labels(mesh, ["west", "east", "east"])
    assert zones.index.tolist() == ["west", "east"]
    assert zones.crs == mesh.crs and zones.min_cell_width == 2.0
    w = overlap_weights(zones, x, y, xres, yres, "+proj=longlat")
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), [8.0, 8.0])
    np.testing.assert_allclose(np.asarray(w.sum(axis=0)).ravel(), 1.0)


def test_grid_from_geometry_projected_mesh_keeps_lonlat_bounds():
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    cx, cy = tr.transform(-111.9, 40.7)
    mesh = Mesh.from_windows([(cx, cy)], 2500.0, crs="EPSG:32612")  # 2.5 km box
    grid = Grid.from_geometry(mesh)
    assert grid.projection == "EPSG:32612"
    assert grid.xres == grid.yres == pytest.approx(600.0)  # 2500/4 = 625 -> 600
    # bounds are lon/lat and enclose the window
    assert grid.xmin < -111.9 < grid.xmax and grid.ymin < 40.7 < grid.ymax
    assert grid.xmin > -113 and grid.xmax < -111 and 40 < grid.ymin < grid.ymax < 41.5
    # the projected axes cover the window with >= 4 cells
    ax, ay = grid.axes
    assert (np.abs(ax - cx) <= 1250).sum() >= 4 and (np.abs(ay - cy) <= 1250).sum() >= 4


def test_mesh_from_file_reads_geojson(tmp_path):
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.GeoDataFrame(
        {"NAME": ["one", "two"]},
        geometry=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    path = tmp_path / "cells.geojson"
    gdf.to_file(path, driver="GeoJSON")
    mesh = Mesh.from_file(path, ids="NAME")
    assert mesh.ids == ("one", "two")
    assert mesh.is_longlat
    assert mesh.bounds == pytest.approx((0.0, 0.0, 2.0, 1.0))


def test_grid_axes_rounding_keeps_coarse_centres():
    """Rounding strips float noise but never corrupts centres such as 0.5, 1.5."""
    coarse = Grid(xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0, xres=1.0, yres=1.0)
    np.testing.assert_array_equal(coarse.axes[0], [0.5, 1.5])
    fine = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.25, yres=0.25)
    np.testing.assert_array_equal(fine.axes[1], [0.125, 0.375, 0.625, 0.875])


# ---------------------------------------------------------------------------
# exactextract backend (optional): identical weights to shapely
# ---------------------------------------------------------------------------


def test_backend_names_are_validated():
    x, y, xres, yres = _raster(2, 1.0)
    mesh = Mesh.from_windows([(1.0, 1.0)], 1.0)
    with pytest.raises(ValueError, match="backend"):
        overlap_weights(mesh, x, y, xres, yres, "+proj=longlat", backend="nope")  # type: ignore[arg-type]


def test_exactextract_backend_matches_shapely_on_triangle():
    pytest.importorskip("exactextract")
    x, y, xres, yres = _raster(2, 1.0)
    tri = Mesh(ids=("t",), geometries=(shapely.Polygon([(0, 0), (2, 0), (0, 2)]),))
    w = overlap_weights(tri, x, y, xres, yres, "+proj=longlat", backend="exactextract")
    np.testing.assert_allclose(w.toarray()[0], [1.0, 0.5, 0.5, 0.0])


def test_exactextract_backend_matches_shapely_on_random_polygons():
    pytest.importorskip("exactextract")
    rng = np.random.default_rng(1)
    x = np.round((np.arange(30) + 0.5) * 0.1, 10)
    y = np.round((np.arange(20) + 0.5) * 0.1, 10)
    polys = [
        shapely.Point(rng.uniform(0, 3), rng.uniform(0, 2)).buffer(
            rng.uniform(0.05, 0.5)
        )
        for _ in range(12)
    ]
    mesh = Mesh(ids=tuple(str(i) for i in range(12)), geometries=tuple(polys))
    a = overlap_weights(mesh, x, y, 0.1, 0.1, "+proj=longlat", backend="shapely")
    b = overlap_weights(mesh, x, y, 0.1, 0.1, "+proj=longlat", backend="exactextract")
    assert a.shape == b.shape
    np.testing.assert_allclose(a.toarray(), b.toarray(), atol=1e-9)
    assert a is not b  # cached separately per backend
