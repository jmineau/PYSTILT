"""Tests for Project.check: what would stop a project's runs, found before any run."""

import datetime as dt
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from pydantic import BaseModel
from runtraj.exceptions import MeteorologyError
from runtraj.meteorology import run_window
from runtraj.models.hysplit import HysplitConfig, Met
from runtraj.releases import (
    ColumnRelease,
    MultiPointRelease,
    PointRelease,
    Release,
)
from runtraj.spatial import Grid

from stilt import Project
from stilt.observations.placement import slant_points
from stilt.transforms import averaging_kernel_table
from tests.fixtures.factories import blocks

from .fixtures.factories import (
    make_met_config,
    make_met_files,
    make_release,
    write_arl_file,
)

NOON = dt.datetime(2023, 1, 1, 12)
GRID = Grid(xmin=-112.0, xmax=-111.0, ymin=40.0, ymax=41.0, xres=0.1, yres=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _release(
    hour: float = 12, longitude: float = -111.85, latitude: float = 40.77
) -> PointRelease:
    time = dt.datetime(2023, 1, 1) + dt.timedelta(hours=hour)
    return make_release(time, longitude, latitude)


def _project(
    tmp_path: Path,
    releases=None,
    *,
    met: dict[str, Any] | None = None,
    files: bool = True,
    n_hours: int = -3,
    **settings,
) -> Project:
    """
    Make a project whose hourly met files cover a run from noon, unless *met* or ``files=False``.

    *met* replaces the local met's settings, and *settings* the project's.
    """
    met_dir = tmp_path / "met"
    config = make_met_config(met_dir)
    if files:
        make_met_files(met_dir, NOON, n_hours)
    if met is not None:
        config = met
    settings.setdefault("grid", GRID)
    settings.setdefault("variants", {"hrrr": {}})
    return Project.init(
        tmp_path / "proj",
        releases=[_release()] if releases is None else releases,
        **blocks(mets={"hrrr": config}, n_hours=n_hours, **settings),
    )


def _found(table: pd.DataFrame, check: str) -> pd.DataFrame:
    return table[table["check"] == check]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_check_returns_one_row_for_each_thing_found(tmp_path):
    inside, outside = _release(12), _release(12, longitude=-110.0)
    project = _project(tmp_path, [inside, outside])

    table = project.check()

    assert list(table["check"]) == ["grid"]
    assert list(table["variant"]) == ["hrrr"]
    assert list(table["release"]) == [outside.id]
    assert "outside the footprint grid" in table["message"].iloc[0]


# ---------------------------------------------------------------------------
# The met's directory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The met's files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_hours", [-3, -6, 4])
def test_releases_are_listed_when_the_run_would_stop_for_a_missing_file(
    tmp_path, n_hours
):
    """The check and a run's own file search agree, on every hour of the files' 6 h span."""
    met_dir = tmp_path / "met"
    config = make_met_config(met_dir)
    for hour in ("00", "06", "12", "18"):
        write_arl_file(
            met_dir / f"20230101_{hour}",
            times=pd.date_range(f"2023-01-01 {hour}:00", periods=6, freq="h"),
        )
    releases = [_release(hour) for hour in (0, 1, 5, 6, 11, 12, 17, 18, 23, 24, 30, 35)]
    project = Project.init(
        tmp_path / "proj",
        releases=releases,
        **blocks(
            mets={"hrrr": config}, variants={"hrrr": {}}, n_hours=n_hours, grid=GRID
        ),
    )
    met = Met("hrrr", project.variants["hrrr"].met_config)

    def fails(release) -> bool:
        window = run_window(release.time, n_hours)
        try:
            met.files_for(window)
        except MeteorologyError:
            return True
        return False

    table = project.check()

    listed = set(_found(table, "met_files")["release"])
    assert listed == {r.id for r in releases if fails(r)}
    assert listed and listed != {r.id for r in releases}


# ---------------------------------------------------------------------------
# The met's crop
# ---------------------------------------------------------------------------

_CROP = {"xmin": -111.9, "xmax": -111.5, "ymin": 40.5, "ymax": 41.0}


def _cropped(tmp_path, **met) -> dict[str, Any]:
    return {
        "directory": str(tmp_path / "met"),
        "subgrid_enable": True,
        "subgrid_bounds": _CROP,
        "subgrid_dir": str(tmp_path / "crops"),
        **met,
    }


def test_a_release_outside_the_crop_box_is_listed(tmp_path):
    west, inside = _release(12, longitude=-111.95), _release(12, longitude=-111.7)
    project = _project(tmp_path, [west, inside], met=_cropped(tmp_path), grid=None)

    table = project.check()

    assert list(table["check"]) == ["met_crop"]
    assert list(table["release"]) == [west.id]
    assert "outside subgrid_bounds" in table["message"].iloc[0]


def test_the_crop_box_is_not_checked_when_the_crop_is_off(tmp_path):
    met = _cropped(tmp_path, subgrid_enable=False)
    project = _project(tmp_path, [_release(12, longitude=-111.95)], met=met, grid=None)

    assert project.check().empty


def test_a_downloaded_met_is_checked_for_its_crop_only(tmp_path):
    met = {
        "directory": str(tmp_path / "later"),
        "download": "hrrr",
        "subgrid_enable": True,
        "subgrid_bounds": _CROP,
    }
    west = _release(12, longitude=-111.95)
    project = _project(tmp_path, [west], met=met, files=False, grid=None)

    table = project.check()

    assert list(table["check"]) == ["met_crop"]
    assert list(table["release"]) == [west.id]


# ---------------------------------------------------------------------------
# The met's domain
# ---------------------------------------------------------------------------


def test_a_release_outside_the_met_files_grid_is_listed(tmp_path):
    """HYSPLIT stopped such a run with "source point off grid", and check said nothing (#440)."""
    # The test met files' grid has points from 125 W to 106 W and 30 N to 49 N.
    east = _release(12, longitude=-105.5)
    north = _release(12, latitude=49.5)
    edge = _release(12, longitude=-106.0, latitude=49.0)
    inside = _release(12)
    project = _project(tmp_path, [east, north, edge, inside], grid=None)

    table = project.check()

    assert list(table["check"]) == ["met_domain", "met_domain"]
    assert list(table["release"]) == [east.id, north.id]
    assert "outside the met's domain" in table["message"].iloc[0]


def test_a_cropped_met_is_checked_for_its_crop_box_not_its_domain(tmp_path):
    east = _release(12, longitude=-105.5)
    project = _project(tmp_path, [east], met=_cropped(tmp_path), grid=None)

    assert list(project.check()["check"]) == ["met_crop"]


# ---------------------------------------------------------------------------
# The footprint grid
# ---------------------------------------------------------------------------


def test_a_release_outside_the_grid_is_listed(tmp_path):
    east = _release(12, longitude=-110.9)
    north = _release(12, latitude=42.0)
    inside = _release(12, longitude=-111.5, latitude=40.5)
    project = _project(tmp_path, [east, inside, north])

    table = project.check()

    assert list(table["check"]) == ["grid", "grid"]
    assert list(table["release"]) == [east.id, north.id]
    assert "outside the footprint grid" in table["message"].iloc[0]


def _grid_box(xmin, xmax, ymin=40.0, ymax=41.0):
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


@pytest.mark.parametrize(
    ("xmin", "xmax", "longitudes", "flagged"),
    [
        # A grid across the dateline or past 180 holds the longitudes in it.
        (-125, -100, [-111.9, -90, -130], [-90, -130]),
        (170, -170, [175, -175, 0], [0]),
        (170, 190, [175, -175, 0], [0]),
        (-190, -170, [-175, 175, 0], [0]),
        # A release on an edge is inside, however the edge is written.
        (170, -170, [170, -170], []),
        (170, 190, [170, -170], []),
        (-190, -170, [-170, 170], []),
        # Just past an edge is outside.
        (170, -170, [169.99, -169.99], [169.99, -169.99]),
        # A grid around the earth holds every longitude.
        (-180, 180, [-180, 0, 180, 90], []),
        (0, 360, [-180, 0, 180, 179, -1], []),
    ],
)
def test_the_grid_counts_longitudes_modulo_360(
    tmp_path, xmin, xmax, longitudes, flagged
):
    """A release is in the grid when its longitude, east of the grid's west edge, is within its width."""
    grid = {**_grid_box(xmin, xmax), "xres": 0.5, "yres": 0.5}
    releases = [_release(12, longitude=lon, latitude=40.5) for lon in longitudes]
    project = _project(tmp_path, releases, grid=grid)

    table = _found(project.check(), "grid")

    ids = {r.id: lon for r, lon in zip(releases, longitudes, strict=True)}
    assert [ids[r] for r in table["release"]] == flagged


def test_a_release_outside_a_grid_across_the_dateline_in_latitude_is_listed(tmp_path):
    grid = {**_grid_box(170, -170), "xres": 0.5, "yres": 0.5}
    north = _release(12, longitude=175, latitude=42.0)
    project = _project(tmp_path, [north], grid=grid)

    table = _found(project.check(), "grid")

    assert list(table["release"]) == [north.id]
    assert "170 to -170 longitude, 40 to 41 latitude" in table["message"].iloc[0]


def test_a_column_or_multipoint_release_with_a_point_outside_is_listed_once(tmp_path):
    column = ColumnRelease(
        time=NOON, longitude=-110.0, latitude=40.5, bottom=0.0, top=3000.0
    )
    multi = MultiPointRelease(
        time=NOON,
        longitudes=(-111.5, -110.0, -109.0),
        latitudes=(40.5, 40.6, 40.7),
        altitudes=(10.0, 20.0, 30.0),
    )
    project = _project(tmp_path, [column, multi])

    table = project.check()

    assert sorted(table["release"]) == sorted([column.id, multi.id])


def test_a_variant_that_makes_no_footprint_has_no_grid_to_check(tmp_path):
    project = _project(
        tmp_path,
        [_release(12, longitude=-110.0)],
        variants={"hrrr": {"grid": None}},
    )

    assert project.check().empty


def test_variants_with_their_own_grid_are_checked_against_it(tmp_path):
    wide = {"xmin": -113.0, "xmax": -109.0, "ymin": 39.0, "ymax": 42.0}
    project = _project(
        tmp_path,
        [_release(12, longitude=-110.0)],
        variants={"hrrr": {}, "wide": {"grid": wide}},
    )

    table = project.check()

    assert list(table["check"]) == ["grid"]
    assert list(table["variant"]) == ["hrrr"]


# ---------------------------------------------------------------------------
# The averaging-kernel table
# ---------------------------------------------------------------------------

_KERNELS = {
    "hrrr-ak": {"transforms": [{"kind": "averaging_kernel", "table": "kernels"}]}
}


def _kernels(project: Project, releases) -> None:
    table = averaging_kernel_table(
        releases, levels=[0.0, 3000.0], values=[[1.0, 1.0]] * len(releases)
    )
    project.add_table("kernels", table)


def test_a_kernel_table_that_matches_the_releases_has_nothing_to_check(tmp_path):
    releases = [_release(11), _release(12)]
    project = _project(tmp_path, releases, variants=_KERNELS)
    _kernels(project, releases)

    assert project.check().empty


def test_a_release_with_no_kernel_is_listed(tmp_path):
    a, b = _release(11), _release(12)
    project = _project(tmp_path, [a, b], variants=_KERNELS)
    _kernels(project, [a])

    table = project.check()

    assert list(table["check"]) == ["kernel_missing"]
    assert list(table["release"]) == [b.id]
    assert list(table["variant"]) == ["hrrr-ak"]


def test_a_kernel_with_no_release_is_listed(tmp_path):
    a, b = _release(11), _release(12)
    project = _project(tmp_path, [a], variants=_KERNELS)
    _kernels(project, [a, b])

    table = project.check()

    assert list(table["check"]) == ["kernel_unused"]
    assert list(table["release"]) == [b.id]


def test_a_kernel_table_that_does_not_exist_is_found_once(tmp_path):
    project = _project(tmp_path, [_release(11), _release(12)], variants=_KERNELS)

    table = project.check()

    assert list(table["check"]) == ["kernel_table"]
    assert table["release"].isna().all()
    assert "'kernels' does not exist" in table["message"].iloc[0]


def test_a_kernel_table_without_its_columns_is_found(tmp_path):
    project = _project(tmp_path, [_release(11)], variants=_KERNELS)
    path = project.table_path("kernels")
    path.parent.mkdir()
    pd.DataFrame({"release": [_release(11).id]}).to_parquet(path)

    table = project.check()

    assert list(table["check"]) == ["kernel_table"]
    assert "lacks columns" in table["message"].iloc[0]


def test_a_kernel_table_given_as_a_file_is_read_from_the_project(tmp_path):
    a, b = _release(11), _release(12)
    variants = {
        "hrrr-ak": {"transforms": [{"kind": "averaging_kernel", "table": "k.csv"}]}
    }
    project = _project(tmp_path, [a, b], variants=variants)
    pd.DataFrame(
        {"release": [a.id] * 2, "level": [0.0, 1.0], "value": [1.0, 1.0]}
    ).to_csv(project.directory / "k.csv", index=False)

    table = project.check()

    assert list(table["check"]) == ["kernel_missing"]
    assert list(table["release"]) == [b.id]


def test_an_inline_kernel_has_no_table_to_check(tmp_path):
    variants = {
        "hrrr-ak": {
            "transforms": [
                {"kind": "averaging_kernel", "levels": [0, 3000], "values": [1, 1]}
            ]
        }
    }

    assert _project(tmp_path, variants=variants).check().empty


# ---------------------------------------------------------------------------
# zsfc in varsiwant
# ---------------------------------------------------------------------------

_WEIGHTING = {"transforms": [{"kind": "pressure_weighting"}]}
_WITH_ZSFC = [*HysplitConfig(n_hours=-24).varsiwant, "zsfc"]


def _above_sea_level_point() -> PointRelease:
    return PointRelease(
        time=NOON,
        longitude=-111.85,
        latitude=40.77,
        altitude=1500.0,
        altitude_ref="msl",
    )


def _multipoint(
    altitude_ref: str,
    altitudes: tuple[float, ...] = (1500.0, 2200.0, 3000.0),
    step: float = 0.005,
) -> MultiPointRelease:
    """Return a multipoint release whose points are *step* degrees (about 425 m for 0.005) apart."""
    return MultiPointRelease(
        time=NOON,
        longitudes=tuple(-111.50 - i * step for i in range(len(altitudes))),
        latitudes=(40.5,) * len(altitudes),
        altitudes=altitudes,
        altitude_ref=altitude_ref,
    )


def _column(altitude_ref: str) -> ColumnRelease:
    return ColumnRelease(
        time=NOON,
        longitude=-111.5,
        latitude=40.5,
        bottom=1500.0,
        top=3000.0,
        altitude_ref=altitude_ref,
    )


def test_a_release_above_sea_level_needs_zsfc_for_the_near_field_correction(tmp_path):
    project = _project(tmp_path, [_above_sea_level_point()])

    table = _found(project.check(), "zsfc")

    assert len(table) == 1
    assert pd.isna(table["release"].iloc[0])
    message = table["message"].iloc[0]
    assert "1 release is above sea level" in message
    assert "near-field correction" in message
    assert "pressure_weighting" not in message
    assert "multipoint" not in message


def test_the_message_gives_the_varsiwant_line_that_adds_zsfc(tmp_path):
    project = _project(
        tmp_path, [_multipoint("msl")], varsiwant=["pres", "zagl"], hnf_plume=False
    )

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "varsiwant: [pres, zagl, zsfc]" in message


def test_the_default_varsiwant_line_keeps_the_defaults(tmp_path):
    project = _project(tmp_path, [_above_sea_level_point()])

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, " in message
    assert message.count("zsfc") == 2


def test_a_multipoint_release_above_sea_level_needs_zsfc_to_match_its_heights(
    tmp_path,
):
    project = _project(tmp_path, [_multipoint("msl")], hnf_plume=False)

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "1 of them is a multipoint release" in message
    assert "less than 1 km apart and at different altitudes" in message
    assert "near-field" not in message


def test_pressure_weighting_of_a_release_above_sea_level_needs_zsfc(tmp_path):
    project = _project(
        tmp_path,
        [_column("msl")],
        hnf_plume=False,
        variants={"hrrr": _WEIGHTING},
    )

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "pressure_weighting" in message
    assert "multipoint" not in message


def test_a_slant_release_from_soundings_is_one_the_check_finds(tmp_path):
    points = slant_points(
        -111.5, 40.5, [1500.0, 2200.0, 3000.0], zenith=40.0, azimuth=120.0
    )
    slant = Release.from_points(NOON, points, altitude_ref="msl")
    project = _project(tmp_path, [slant], variants={"hrrr": _WEIGHTING})

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert all(
        phrase in message
        for phrase in ("near-field correction", "multipoint", "pressure_weighting")
    )


def test_a_slant_release_alone_is_found_by_its_release_heights(tmp_path):
    points = slant_points(
        -111.5, 40.5, [1500.0, 2200.0, 3000.0], zenith=40.0, azimuth=120.0
    )
    slant = Release.from_points(NOON, points, altitude_ref="msl")
    project = _project(tmp_path, [slant], hnf_plume=False)

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "1 of them is a multipoint release" in message
    assert "near-field" not in message
    assert "pressure_weighting" not in message


def test_a_multipoint_release_with_one_altitude_does_not_need_zsfc(tmp_path):
    level = _multipoint("msl", altitudes=(2000.0, 2000.0, 2000.0))
    project = _project(tmp_path, [level], hnf_plume=False)

    assert _found(project.check(), "zsfc").empty


def test_a_multipoint_release_with_two_points_at_one_altitude_does_not_need_zsfc(
    tmp_path,
):
    shared = _multipoint("msl", altitudes=(1500.0, 1500.0, 3000.0))
    project = _project(tmp_path, [shared], hnf_plume=False)

    assert _found(project.check(), "zsfc").empty


def test_a_multipoint_release_with_points_far_apart_does_not_need_zsfc(tmp_path):
    spread = _multipoint("msl", step=0.05)  # about 4 km
    project = _project(tmp_path, [spread], hnf_plume=False)

    assert _found(project.check(), "zsfc").empty


def test_a_multipoint_release_with_one_altitude_still_needs_zsfc_for_the_rest(
    tmp_path,
):
    level = _multipoint("msl", altitudes=(2000.0, 2000.0, 2000.0))
    project = _project(tmp_path, [level], variants={"hrrr": _WEIGHTING})

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "multipoint" not in message
    assert "near-field correction" in message
    assert "pressure_weighting" in message


def test_only_the_releases_above_sea_level_are_counted(tmp_path):
    project = _project(
        tmp_path,
        [_release(), _above_sea_level_point()],
        variants={"hrrr": _WEIGHTING},
    )

    message = _found(project.check(), "zsfc")["message"].iloc[0]

    assert "1 release is above sea level" in message
    assert "fails their footprints" in message


def test_every_reason_is_named_once_for_all_the_releases(tmp_path):
    project = _project(
        tmp_path,
        [_above_sea_level_point(), _multipoint("msl")],
        variants={"hrrr": _WEIGHTING},
    )

    table = _found(project.check(), "zsfc")

    assert len(table) == 1
    assert "2 releases are above sea level" in table["message"].iloc[0]


def test_no_zsfc_is_needed_when_it_is_in_varsiwant(tmp_path):
    project = _project(
        tmp_path,
        [_above_sea_level_point(), _multipoint("msl")],
        variants={"hrrr": _WEIGHTING},
        varsiwant=_WITH_ZSFC,
    )

    assert _found(project.check(), "zsfc").empty


def test_no_zsfc_is_needed_for_releases_above_ground(tmp_path):
    project = _project(
        tmp_path,
        [_release(), _multipoint("agl"), _column("agl")],
        variants={"hrrr": _WEIGHTING},
    )

    assert _found(project.check(), "zsfc").empty


def test_no_zsfc_is_needed_when_nothing_reads_it(tmp_path):
    project = _project(tmp_path, [_above_sea_level_point()], hnf_plume=False)

    assert _found(project.check(), "zsfc").empty


def test_variants_that_need_zsfc_for_the_same_reason_share_the_row(tmp_path):
    project = _project(
        tmp_path,
        [_above_sea_level_point()],
        variants={"hrrr": {}, "hrrr-b": {"veght": 0.3}},
    )

    table = _found(project.check(), "zsfc")

    assert list(table["variant"]) == ["hrrr, hrrr-b"]


def test_a_variant_with_zsfc_is_not_listed_with_one_without(tmp_path):
    project = _project(
        tmp_path,
        [_above_sea_level_point()],
        variants={"hrrr": {}, "hrrr-z": {"varsiwant": _WITH_ZSFC}},
    )

    table = _found(project.check(), "zsfc")

    assert list(table["variant"]) == ["hrrr"]


# ---------------------------------------------------------------------------
# More than one thing, and another transport model
# ---------------------------------------------------------------------------


def test_every_check_runs_for_every_variant(tmp_path):
    a, b = _release(11), _release(20, longitude=-110.0)
    project = _project(tmp_path, [a, b], variants=_KERNELS)
    _kernels(project, [a])

    table = project.check()

    assert set(table["check"]) == {"met_files", "grid", "kernel_missing"}
    assert set(table["release"]) == {b.id}


class _SelfChecking(BaseModel):
    """A user's transform that checks its own inputs and reads zsfc, named by import path."""

    reads_zsfc: ClassVar[bool] = True

    def apply(self, particles, release=None, directory=None):
        return particles.copy()

    def check(self, releases, directory):
        return [
            ("my_input", rid, f"no input for this release in {directory}")
            for rid in releases
        ]


def test_a_transform_checks_its_own_inputs_and_says_it_reads_zsfc(tmp_path):
    project = _project(
        tmp_path,
        [_column("msl")],
        hnf_plume=False,
        variants={"hrrr": {"transforms": [{"kind": "tests.test_check._SelfChecking"}]}},
    )

    found = project.check()

    mine = _found(found, "my_input")
    assert list(mine["release"]) == [str(project.releases["release"].iloc[0])]
    assert str(project.directory) in mine["message"].iloc[0]
    zsfc = _found(found, "zsfc")["message"].iloc[0]
    assert "the tests.test_check._SelfChecking transform fails their footprints" in zsfc
