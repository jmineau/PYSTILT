"""Tests for runtraj.spatial: the footprint grid and the CRS helpers."""

import numpy as np
import pytest

from stilt import Grid

# ---------------------------------------------------------------------------
# Grid as a geometry
# ---------------------------------------------------------------------------


def test_one_longlat_test_for_grids_and_meshes():
    """EPSG:4326 is longitude/latitude for a grid as it is for a mesh."""
    from runtraj.spatial import is_longlat

    from stilt.footprint.grid import Grid
    from stilt.footprint.targets import Mesh

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


_LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m"
_NORTH_POLAR = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45"
_ALASKA = "+proj=lcc +lat_1=50 +lat_2=60 +lat_0=55 +lon_0=180 +units=m"


def _outline(xmin, xmax, ymin, ymax, n=2001):
    """Points along a longitude/latitude box's four edges, east of ``xmin`` up to its east edge."""
    east = xmax + 360.0 if xmax < xmin else xmax
    lon = np.linspace(xmin, east, n)
    lat = np.linspace(ymin, ymax, n)
    lons = np.concatenate([lon, lon, np.full(n, xmin), np.full(n, east)])
    lats = np.concatenate([np.full(n, ymin), np.full(n, ymax), lat, lat])
    return lons, lats


@pytest.mark.parametrize(
    ("bounds", "res", "crs"),
    [
        # The contiguous US on a Lambert conformal grid: the two projected
        # corners left out 18 % of it.
        ((-125.0, -66.0, 24.0, 50.0), 10_000.0, _LCC),
        # The cap north of 60 N: the two corners gave one quadrant.
        ((-180.0, 180.0, 60.0, 90.0), 25_000.0, _NORTH_POLAR),
        # Across 180 degrees, written both ways.
        ((170.0, -170.0, 50.0, 60.0), 10_000.0, _ALASKA),
        ((170.0, 190.0, 50.0, 60.0), 10_000.0, _ALASKA),
        # The docs' UTM example.
        ((-113.0, -110.5, 40.0, 42.0), 1_000.0, "EPSG:32612"),
    ],
)
def test_projected_grid_covers_the_box(bounds, res, crs):
    """A projected grid's cells hold every point of its longitude/latitude box's outline (#504)."""
    from pyproj import Transformer

    xmin, xmax, ymin, ymax = bounds
    grid = Grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xres=res, yres=res, crs=crs)
    x, y = grid.axes
    lons, lats = _outline(*bounds)
    px, py = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        lons, lats
    )
    assert px.min() >= x[0] - res / 2 and px.max() <= x[-1] + res / 2
    assert py.min() >= y[0] - res / 2 and py.max() <= y[-1] + res / 2
    # and starts at the outline's west and south extremes, not a cell before
    assert px.min() - (x[0] - res / 2) < res / 100
    assert py.min() - (y[0] - res / 2) < res / 100


# ---------------------------------------------------------------------------
# Bounds.contains: longitudes count modulo 360
# ---------------------------------------------------------------------------


def test_grid_refuses_an_unknown_key_in_the_config(tmp_path):
    import stilt

    met = {"directory": str(tmp_path), "file_format": "%Y%m%d", "file_tres": "6h"}
    stage = "stilt.footprint.FootprintStage"
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
            mets={"a": met},
            stages={
                "footprint": {"class": stage, "grid": {**grid, "csr": "EPSG:32612"}}
            },
            variants={"a": {}},
        )
    with pytest.raises(ValueError, match="'xresolution' is not a grid setting"):
        stilt.ProjectConfig(
            mets={"a": met},
            stages={"footprint": {"class": stage}},
            variants={
                "a": {"stages": {"footprint": {"grid": {**grid, "xresolution": 3}}}}
            },
        )


def test_a_grid_refuses_a_crs_pyproj_cannot_read():
    """A misspelled crs fails when the grid is made, not when a footprint is (#439)."""
    with pytest.raises(ValueError, match="not a coordinate reference system"):
        Grid(xmin=-113, xmax=-110, ymin=40, ymax=42, xres=1000, yres=1000, crs="nope")
