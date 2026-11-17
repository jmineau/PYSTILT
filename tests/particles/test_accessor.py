"""Tests for the particle table's ``.stilt`` accessor: where particles first leave a grid."""

import numpy as np
import pandas as pd
import pytest

import stilt

#: Two degrees by two, in half-degree cells: -112 to -110 east, 40 to 42 north.
GRID = stilt.Grid(xmin=-112, xmax=-110, ymin=40, ymax=42, xres=0.5, yres=0.5)


def _particles(*rows: tuple, time: bool = False) -> pd.DataFrame:
    """Return a table from ``(particle, age, lon, lat, zagl)`` rows."""
    p = pd.DataFrame(rows, columns=["particle", "age", "lon", "lat", "zagl"])
    if time:
        p["time"] = pd.Timestamp("2023-01-01 12:00") + pd.to_timedelta(
            p["age"], unit="min"
        )
    return p


def _exit(rows: list[tuple], grid: stilt.Grid = GRID) -> pd.Series:
    """Return the one exit of a table made of one particle's rows."""
    exits = _particles(*rows).stilt.grid_exits(grid)
    assert len(exits) == 1
    return exits.iloc[0]


@pytest.mark.parametrize(
    ("edge", "last"),
    [
        ("n", (-111, 42.5)),
        ("e", (-109.5, 41)),
        ("s", (-111, 39.5)),
        ("w", (-112.5, 41)),
    ],
)
def test_the_edge_is_the_side_the_particle_left_by(edge, last):
    rows = [(1, 0, -111, 41, 10), (1, -60, last[0], last[1], 30)]
    assert _exit(rows)["edge"] == edge


def test_position_height_and_age_are_interpolated_onto_the_edge():
    """The step from -110.5 to -109.5 is crossed halfway, at -110."""
    exit_ = _exit(
        [
            (1, 0, -111, 41.0, 10),
            (1, -60, -110.5, 41.0, 20),
            (1, -120, -109.5, 42.0, 30),
        ]
    )
    assert exit_["edge"] == "e"
    assert exit_["lon"] == pytest.approx(-110.0)
    assert exit_["lat"] == pytest.approx(41.5)
    assert exit_["zagl"] == pytest.approx(25.0)
    assert exit_["age"] == pytest.approx(-90.0)


def test_time_is_interpolated_when_the_table_has_it():
    p = _particles(
        (1, 0, -111, 41, 10),
        (1, -60, -110.5, 41, 20),
        (1, -120, -109.5, 41, 30),
        time=True,
    )
    exits = p.stilt.grid_exits(GRID)
    assert exits["time"].iloc[0] == pd.Timestamp("2023-01-01 10:30")
    assert "time" not in p.drop(columns="time").stilt.grid_exits(GRID).columns


def test_a_particle_that_stays_inside_is_not_in_the_result():
    p = _particles(
        (1, 0, -111, 41, 10),
        (1, -60, -110.5, 41, 10),
        (2, 0, -111, 41, 10),
        (2, -60, -109, 41, 10),
    )
    exits = p.stilt.grid_exits(GRID)
    assert exits["particle"].tolist() == [2]
    assert list(exits.columns) == ["particle", "edge", "lon", "lat", "zagl", "age"]


def test_the_first_exit_counts_when_a_particle_comes_back():
    """Out the north edge, then back in, then out the east: the exit is the north one."""
    exit_ = _exit(
        [
            (1, 0, -111, 41.5, 10),
            (1, -60, -111, 42.5, 10),
            (1, -120, -111, 41, 10),
            (1, -180, -109, 41, 10),
        ]
    )
    assert exit_["edge"] == "n"
    assert exit_["lat"] == pytest.approx(42.0)
    assert exit_["age"] == pytest.approx(-30.0)


def test_a_corner_exit_is_the_edge_crossed_first():
    """A step that crosses north and east: the exit is on the one it reaches first."""
    exit_ = _exit([(1, 0, -110.5, 41.5, 10), (1, -60, -109.5, 43.5, 10)])
    assert exit_["edge"] == "n"
    exit_ = _exit([(1, 0, -110.5, 41.5, 10), (1, -60, -108.5, 42.5, 10)])
    assert exit_["edge"] == "e"
    assert exit_["lon"] == pytest.approx(-110.0)
    assert exit_["lat"] == pytest.approx(41.75)


def test_rows_are_read_in_time_order_for_a_backward_and_a_forward_run():
    rows = [(1, 0, -111, 41, 10), (1, -60, -110.5, 41, 20), (1, -120, -109.5, 41, 30)]
    shuffled = _particles(*rows[::-1]).stilt.grid_exits(GRID)
    assert shuffled["age"].tolist() == [-90.0]
    forward = [(1, 0, -111, 41, 10), (1, 60, -110.5, 41, 20), (1, 120, -109.5, 41, 30)]
    assert _particles(*forward[::-1]).stilt.grid_exits(GRID)["age"].tolist() == [90.0]


def test_a_particle_whose_first_row_is_outside_is_reported_at_that_row():
    """No row inside to interpolate from: the row itself, on the edge it is farthest beyond."""
    exit_ = _exit([(1, -1, -109.0, 42.2, 10), (1, -61, -108, 42, 20)])
    assert exit_["edge"] == "e"
    assert (exit_["lon"], exit_["lat"], exit_["age"]) == (-109.0, 42.2, -1.0)


def test_a_point_on_the_edge_is_inside():
    p = _particles((1, 0, -111, 41, 10), (1, -60, -110.0, 42.0, 10))
    assert p.stilt.grid_exits(GRID).empty


def test_the_edge_is_that_of_the_cells_when_the_bounds_do_not_fit_whole_cells():
    """Cells of 0.75 degrees reach -110.5 from -112, not -110."""
    grid = stilt.Grid(xmin=-112, xmax=-110, ymin=40, ymax=42, xres=0.75, yres=0.75)
    exit_ = _exit([(1, 0, -111, 41, 10), (1, -60, -110, 41, 10)], grid)
    assert exit_["lon"] == pytest.approx(-110.5)


def test_a_grid_across_the_dateline_is_read_modulo_360():
    for grid in (
        stilt.Grid(xmin=170, xmax=190, ymin=-10, ymax=10, xres=1, yres=1),
        stilt.Grid(xmin=170, xmax=-170, ymin=-10, ymax=10, xres=1, yres=1),
    ):
        inside = _particles(
            (1, 0, 175, 0, 1), (1, -60, -175, 0, 1), (1, -120, 179, 5, 1)
        )
        assert inside.stilt.grid_exits(grid).empty
        east = _exit([(1, 0, -175, 0, 1), (1, -60, -169, 0, 1)], grid)
        assert east["edge"] == "e"
        assert east["lon"] == pytest.approx(-170.0)
        west = _exit([(1, 0, 172, 0, 1), (1, -60, 168, 0, 1)], grid)
        assert west["edge"] == "w"
        assert west["lon"] == pytest.approx(170.0)


def test_an_exit_across_180_has_its_longitude_wrapped():
    grid = stilt.Grid(xmin=170, xmax=180.5, ymin=-10, ymax=10, xres=0.5, yres=0.5)
    exit_ = _exit([(1, 0, 179, 0, 1), (1, -60, -179, 0, 1)], grid)
    assert exit_["edge"] == "e"
    assert exit_["lon"] == pytest.approx(-179.5)


def test_a_grid_around_the_earth_has_only_a_north_and_a_south_edge():
    grid = stilt.Grid(xmin=-180, xmax=180, ymin=-60, ymax=60, xres=1, yres=1)
    p = _particles(
        (1, 0, 179, 0, 1), (1, -60, -179, 0, 1), (2, 0, 0, 59, 1), (2, -60, 0, 61, 1)
    )
    exits = p.stilt.grid_exits(grid)
    assert exits["particle"].tolist() == [2]
    assert exits["edge"].tolist() == ["n"]


def test_a_projected_grid_is_a_rectangle_in_its_projection():
    """A UTM grid 20 km wide: the exit is at the grid's east edge in meters, not in degrees."""
    from pyproj import Transformer

    crs = "EPSG:32612"
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_deg = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = to_utm.transform(-111.85, 40.77)
    xmin, ymin = (
        np.floor(x0 / 1000) * 1000 - 10_000,
        np.floor(y0 / 1000) * 1000 - 10_000,
    )
    (west, south), (east, north) = (
        to_deg.transform(xmin, ymin),
        to_deg.transform(xmin + 20_000, ymin + 20_000),
    )
    grid = stilt.Grid(
        xmin=west, xmax=east, ymin=south, ymax=north, xres=1000, yres=1000, crs=crs
    )
    x_edge = grid.axes[0][-1] + 500
    lon, lat = to_deg.transform([x0, x_edge + 5_000], [y0, y0])
    exit_ = _exit([(1, 0, lon[0], lat[0], 5), (1, -60, lon[1], lat[1], 5)], grid)
    assert exit_["edge"] == "e"
    back = to_utm.transform(exit_["lon"], exit_["lat"])[0]
    assert back == pytest.approx(x_edge, abs=1.0)


def test_an_empty_table_gives_an_empty_result_with_the_columns():
    exits = _particles().stilt.grid_exits(GRID)
    assert exits.empty
    assert list(exits.columns) == ["particle", "edge", "lon", "lat", "zagl", "age"]
