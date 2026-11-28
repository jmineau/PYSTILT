"""Tests for the footprint grid and its CRS helpers."""

import numpy as np
import pytest

from stilt import Grid
from stilt.footprint.grid import _is_longlat, _same_crs

# ---------------------------------------------------------------------------
# Grid as a geometry
# ---------------------------------------------------------------------------


def test_one_longlat_test_for_grids_and_meshes():
    """EPSG:4326 is longitude/latitude for a grid as it is for a mesh."""
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
    assert _is_longlat("EPSG:4326") and _is_longlat("+proj=longlat")
    assert not _is_longlat("EPSG:32612")
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


@pytest.mark.parametrize(
    ("bounds", "res", "crs"),
    [
        # The contiguous US on a Lambert conformal grid
        ((-2840e3, 3240e3, -1630e3, 1680e3), 10_000.0, _LCC),
        # The cap north of 60 N, across the pole
        ((-3325e3, 3325e3, -3325e3, 3325e3), 25_000.0, _NORTH_POLAR),
        # Across 180 degrees
        ((-400e3, 400e3, -500e3, 600e3), 10_000.0, _ALASKA),
        # UTM, with a partial cell at the east and north edges
        ((329_000.0, 543_500.0, 4_427_000.0, 4_652_700.0), 1_000.0, "EPSG:32612"),
    ],
)
def test_projected_grid_has_the_cells_of_its_bounds(bounds, res, crs):
    """A projected grid's bounds are x and y in its crs, and its cells fill them (#543)."""
    xmin, xmax, ymin, ymax = bounds
    grid = Grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xres=res, yres=res, crs=crs)
    x, y = grid.axes
    assert x[0] == xmin + res / 2 and y[0] == ymin + res / 2
    # whole cells only, as on a longitude/latitude grid
    assert len(x) == int((xmax - xmin) // res) and len(y) == int((ymax - ymin) // res)


@pytest.mark.parametrize(
    ("bounds", "res", "crs"),
    [
        ((-125.0, -66.0, 24.0, 50.0), 10_000.0, _LCC),
        ((-180.0, 180.0, 60.0, 90.0), 25_000.0, _NORTH_POLAR),
        ((170.0, -170.0, 50.0, 60.0), 10_000.0, _ALASKA),
        ((-113.0, -110.5, 40.0, 42.0), 1_000.0, "EPSG:32612"),
        ((-112.2, -111.7, 40.4, 40.9), 0.001, "EPSG:32612"),
    ],
)
def test_projected_grid_refuses_longitude_latitude_bounds(bounds, res, crs):
    """A longitude/latitude box given with a projected crs is refused, not read as x and y (#543)."""
    xmin, xmax, ymin, ymax = bounds
    with pytest.raises(ValueError, match="look like longitude and latitude"):
        Grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xres=res, yres=res, crs=crs)


def test_projected_grid_refuses_swapped_bounds():
    with pytest.raises(ValueError, match="xmin < xmax and ymin < ymax"):
        Grid(
            xmin=5e5,
            xmax=4e5,
            ymin=4.4e6,
            ymax=4.5e6,
            xres=1e3,
            yres=1e3,
            crs="EPSG:32612",
        )


def test_projected_grid_contains_projects_the_points():
    grid = Grid(
        xmin=4e5,
        xmax=4.5e5,
        ymin=4.48e6,
        ymax=4.53e6,
        xres=1e3,
        yres=1e3,
        crs="EPSG:32612",
    )
    # Salt Lake City is inside; the same numbers read as x and y would not be.
    inside = grid.contains([-111.9, -100.0, np.nan], [40.7, 40.7, 40.7])
    assert inside.tolist() == [True, False, False]


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
        Grid(xmin=4e5, xmax=5e5, ymin=4.4e6, ymax=4.5e6, xres=1e3, yres=1e3, crs="nope")


# ---------------------------------------------------------------------------
# _is_longlat and _same_crs: the spellings of a geographic CRS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crs",
    [
        "+proj=longlat +datum=WGS84",
        "+proj=latlong +datum=WGS84",
        "+proj=lonlat +datum=WGS84",
        "EPSG:4326",
        "WGS84",
    ],
)
def test_is_longlat_holds_each_spelling_of_a_geographic_crs(crs):
    assert _is_longlat(crs)


@pytest.mark.parametrize(
    "crs",
    ["EPSG:32612", "+proj=utm +zone=12 +datum=WGS84", "+proj=lcc +lat_1=33 +lat_2=45"],
)
def test_is_longlat_refuses_a_projected_crs(crs):
    assert not _is_longlat(crs)


def test_is_longlat_refuses_a_string_pyproj_cannot_read():
    assert not _is_longlat("+proj=bogus")


def test_same_crs_matches_geographic_spellings_and_refuses_others():
    assert _same_crs("+proj=latlong +datum=WGS84", "+proj=lonlat +datum=WGS84")
    assert _same_crs("+proj=longlat +datum=WGS84", "EPSG:4326")
    assert _same_crs("EPSG:32612", "EPSG:32612")
    assert not _same_crs("EPSG:32612", "EPSG:32613")
    assert not _same_crs("EPSG:32612", "+proj=latlong +datum=WGS84")
