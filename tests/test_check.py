"""Tests for Project.check: what would stop a project's runs, found before any run."""

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

from stilt import transport
from stilt.exceptions import MeteorologyError
from stilt.meteorology import run_window
from stilt.project import Project
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.spatial import Grid
from stilt.transforms import averaging_kernel_table
from stilt.transport.hysplit import Met

from .fixtures.factories import (
    make_met_config,
    make_met_files,
    make_receptor,
    write_arl_file,
)

NOON = dt.datetime(2023, 1, 1, 12)
GRID = Grid(xmin=-112.0, xmax=-111.0, ymin=40.0, ymax=41.0, xres=0.1, yres=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _receptor(
    hour: float = 12, longitude: float = -111.85, latitude: float = 40.77
) -> PointReceptor:
    time = dt.datetime(2023, 1, 1) + dt.timedelta(hours=hour)
    return make_receptor(time, longitude, latitude)


def _project(
    tmp_path: Path,
    receptors=None,
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
    config = make_met_config(met_dir, file_tres="1h")
    if files:
        make_met_files(met_dir, NOON, n_hours)
    if met is not None:
        config = {"file_format": "%Y%m%d_%H", "file_tres": "1h", **met}
    settings.setdefault("grid", GRID)
    settings.setdefault("variants", {"hrrr": {}})
    return Project.init(
        tmp_path / "proj",
        receptors=[_receptor()] if receptors is None else receptors,
        mets={"hrrr": config},
        n_hours=n_hours,
        **settings,
    )


def _found(table: pd.DataFrame, check: str) -> pd.DataFrame:
    return table[table["check"] == check]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_a_project_with_everything_in_place_has_nothing_to_check(tmp_path):
    table = _project(tmp_path).check()

    assert table.empty
    assert list(table.columns) == ["check", "variant", "receptor", "message"]


def test_check_returns_one_row_for_each_thing_found(tmp_path):
    inside, outside = _receptor(12), _receptor(12, longitude=-110.0)
    project = _project(tmp_path, [inside, outside])

    table = project.check()

    assert list(table["check"]) == ["grid"]
    assert list(table["variant"]) == ["hrrr"]
    assert list(table["receptor"]) == [outside.id]
    assert "outside the footprint grid" in table["message"].iloc[0]


# ---------------------------------------------------------------------------
# The met's directory
# ---------------------------------------------------------------------------


def test_a_met_directory_that_does_not_exist_is_found_once_for_the_met(tmp_path):
    receptors = [_receptor(12), _receptor(13)]
    project = _project(
        tmp_path,
        receptors,
        met={"directory": str(tmp_path / "typo")},
        files=False,
    )

    table = project.check()

    assert list(table["check"]) == ["met_directory"]
    assert table["receptor"].isna().all()
    assert f"{tmp_path / 'typo'} does not exist" in table["message"].iloc[0]


def test_a_met_directory_with_no_arl_file_is_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "README").write_text("not an ARL file")
    project = _project(tmp_path, met={"directory": str(empty)}, files=False)

    table = project.check()

    assert list(table["check"]) == ["met_directory"]
    assert "No ARL file" in table["message"].iloc[0]


def test_the_directory_is_only_a_download_destination_with_download(tmp_path):
    """A met with download makes its directory, and its files, when a run needs them."""
    project = _project(
        tmp_path,
        met={"directory": str(tmp_path / "later"), "download": "hrrr"},
        files=False,
    )

    assert project.check().empty


# ---------------------------------------------------------------------------
# The met's files
# ---------------------------------------------------------------------------


def test_a_receptor_missing_a_met_file_is_listed_with_its_hours(tmp_path):
    whole, short = _receptor(12), _receptor(20)
    project = _project(tmp_path, [whole, short])
    # The run of `short` covers 17:00 to 20:00; the files stop after 13:00.

    table = project.check()

    assert list(table["check"]) == ["met_files"]
    assert list(table["receptor"]) == [short.id]
    message = table["message"].iloc[0]
    assert message == (
        "no met file for 2023-01-01 17:00, 2023-01-01 18:00, "
        "2023-01-01 19:00, 2023-01-01 20:00"
    )


def _one_period_files(tmp_path: Path) -> list[Path]:
    """Rewrite the project's met files to hold one time period each, their own hour."""
    files = sorted((tmp_path / "met").glob("2023*"))
    for path in files:
        hour = pd.Timestamp(dt.datetime.strptime(path.name, "%Y%m%d_%H"))
        write_arl_file(path, times=[hour])
    return files


@pytest.mark.parametrize("n_hours", [-3, 4])
def test_met_files_that_hold_one_time_period_are_found_once_for_the_met(
    tmp_path, n_hours
):
    """HYSPLIT stops on a file with one time period, so every run of the met would fail (#286)."""
    project = _project(tmp_path, [_receptor(12), _receptor(13)], n_hours=n_hours)
    _one_period_files(tmp_path)

    table = project.check()

    assert list(table["check"]) == ["met_periods"]
    assert table["receptor"].isna().all()
    message = table["message"].iloc[0]
    assert "holds one time period" in message
    assert "Only one time period of meteo data" in message


def test_the_file_with_one_time_period_is_named(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    _one_period_files(tmp_path)

    table = project.check()

    # a backward run of 3 hours from 12:00 starts in the file of 09:00
    assert str(tmp_path / "met" / "20230101_09") in table["message"].iloc[0]


def test_a_missing_first_file_is_not_also_a_file_with_one_time_period(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    _one_period_files(tmp_path)
    (tmp_path / "met" / "20230101_09").unlink()

    assert list(project.check()["check"]) == ["met_files"]


def test_a_damaged_first_file_is_left_to_the_run(tmp_path):
    """The check counts time periods only; a file it cannot read as ARL fails at the run, which names the damage."""
    project = _project(tmp_path, [_receptor(12)])
    first = tmp_path / "met" / "20230101_09"
    first.write_bytes(first.read_bytes()[:-10])  # cut inside a record

    assert project.check().empty


def test_checking_opens_one_met_file(tmp_path, monkeypatch):
    """The check counts the time periods of one file, not of every file the receptors need."""
    import arlmet

    project = _project(tmp_path, [_receptor(12), _receptor(13)])
    opened: list[str] = []
    real_file = arlmet.File

    def counting_file(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(arlmet, "File", counting_file)

    assert project.check().empty
    assert opened == ["20230101_09"]


def test_a_met_file_the_hour_after_a_backward_run_needs_is_found_missing(tmp_path):
    """A backward run also reads the hour after the release, which HYSPLIT interpolates."""
    receptor = _receptor(12.5)
    project = _project(tmp_path, [receptor])
    (tmp_path / "met" / "20230101_13").unlink()

    table = project.check()

    assert list(table["message"]) == ["no met file for 2023-01-01 13:00"]


def test_a_met_file_the_hour_after_a_forward_run_needs_is_found_missing(tmp_path):
    """A forward run reads the hour after its end too, so `stilt check` lists it (#275)."""
    project = _project(tmp_path, [_receptor(12.5)], n_hours=3)
    (tmp_path / "met" / "20230101_16").unlink()

    table = project.check()

    assert list(table["message"]) == ["no met file for 2023-01-01 16:00"]


@pytest.mark.parametrize("n_hours", [-3, -6, 4])
def test_receptors_are_listed_when_the_run_would_stop_for_a_missing_file(
    tmp_path, n_hours
):
    """The check and a run's own file search agree, on every hour of the files' 6 h span."""
    met_dir = tmp_path / "met"
    config = make_met_config(met_dir, file_tres="6h")
    for stamp in ("20230101_00", "20230101_06", "20230101_12", "20230101_18"):
        write_arl_file(met_dir / stamp)
    receptors = [
        _receptor(hour) for hour in (0, 1, 5, 6, 11, 12, 17, 18, 23, 24, 30, 35)
    ]
    project = Project.init(
        tmp_path / "proj",
        receptors=receptors,
        mets={"hrrr": config},
        variants={"hrrr": {}},
        n_hours=n_hours,
        grid=GRID,
    )
    met = Met("hrrr", project.variants["hrrr"].met_config)

    def fails(receptor) -> bool:
        window = run_window(receptor.time, n_hours)
        try:
            met.files_for(window)
        except MeteorologyError:
            return True
        return False

    table = project.check()

    listed = set(_found(table, "met_files")["receptor"])
    assert listed == {r.id for r in receptors if fails(r)}
    assert listed and listed != {r.id for r in receptors}


def test_a_variant_with_a_longer_run_is_listed_for_its_own_files(tmp_path):
    project = _project(
        tmp_path,
        [_receptor(12)],
        variants={"hrrr": {}, "hrrr-long": {"n_hours": -8}},
    )

    table = project.check()

    assert list(table["check"]) == ["met_files"]
    assert list(table["variant"]) == ["hrrr-long"]


def test_variants_that_share_a_met_and_a_run_length_share_their_rows(tmp_path):
    project = _project(
        tmp_path,
        [_receptor(20)],
        variants={"hrrr": {}, "hrrr-zi": {"ziscale": 0.8}},
    )

    table = project.check()

    assert list(table["check"]) == ["met_files"]
    assert list(table["variant"]) == ["hrrr, hrrr-zi"]


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


def test_a_receptor_outside_the_crop_box_is_listed(tmp_path):
    west, inside = _receptor(12, longitude=-111.95), _receptor(12, longitude=-111.7)
    project = _project(tmp_path, [west, inside], met=_cropped(tmp_path), grid=None)

    table = project.check()

    assert list(table["check"]) == ["met_crop"]
    assert list(table["receptor"]) == [west.id]
    assert "outside subgrid_bounds" in table["message"].iloc[0]


def test_the_crop_box_is_not_checked_when_the_crop_is_off(tmp_path):
    met = _cropped(tmp_path, subgrid_enable=False)
    project = _project(tmp_path, [_receptor(12, longitude=-111.95)], met=met, grid=None)

    assert project.check().empty


def test_a_downloaded_met_is_checked_for_its_crop_only(tmp_path):
    met = {
        "directory": str(tmp_path / "later"),
        "download": "hrrr",
        "subgrid_enable": True,
        "subgrid_bounds": _CROP,
    }
    west = _receptor(12, longitude=-111.95)
    project = _project(tmp_path, [west], met=met, files=False, grid=None)

    table = project.check()

    assert list(table["check"]) == ["met_crop"]
    assert list(table["receptor"]) == [west.id]


# ---------------------------------------------------------------------------
# The footprint grid
# ---------------------------------------------------------------------------


def test_a_receptor_outside_the_grid_is_listed(tmp_path):
    east = _receptor(12, longitude=-110.9)
    north = _receptor(12, latitude=42.0)
    inside = _receptor(12, longitude=-111.5, latitude=40.5)
    project = _project(tmp_path, [east, inside, north])

    table = project.check()

    assert list(table["check"]) == ["grid", "grid"]
    assert list(table["receptor"]) == [east.id, north.id]
    assert "outside the footprint grid" in table["message"].iloc[0]


def test_a_receptor_on_the_edge_of_the_grid_is_inside_it(tmp_path):
    edge = _receptor(12, longitude=GRID.xmin, latitude=GRID.ymax)

    assert _project(tmp_path, [edge]).check().empty


def test_a_column_or_multipoint_receptor_with_a_point_outside_is_listed_once(tmp_path):
    column = ColumnReceptor(
        time=NOON, longitude=-110.0, latitude=40.5, bottom=0.0, top=3000.0
    )
    multi = MultiPointReceptor(
        time=NOON,
        longitudes=(-111.5, -110.0, -109.0),
        latitudes=(40.5, 40.6, 40.7),
        altitudes=(10.0, 20.0, 30.0),
    )
    project = _project(tmp_path, [column, multi])

    table = project.check()

    assert sorted(table["receptor"]) == sorted([column.id, multi.id])


def test_a_variant_that_makes_no_footprint_has_no_grid_to_check(tmp_path):
    project = _project(
        tmp_path,
        [_receptor(12, longitude=-110.0)],
        variants={"hrrr": {"grid": None}},
    )

    assert project.check().empty


def test_variants_with_their_own_grid_are_checked_against_it(tmp_path):
    wide = {"xmin": -113.0, "xmax": -109.0, "ymin": 39.0, "ymax": 42.0}
    project = _project(
        tmp_path,
        [_receptor(12, longitude=-110.0)],
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


def _kernels(project: Project, receptors) -> None:
    table = averaging_kernel_table(
        receptors, levels=[0.0, 3000.0], values=[[1.0, 1.0]] * len(receptors)
    )
    project.add_table("kernels", table)


def test_a_kernel_table_that_matches_the_receptors_has_nothing_to_check(tmp_path):
    receptors = [_receptor(11), _receptor(12)]
    project = _project(tmp_path, receptors, variants=_KERNELS)
    _kernels(project, receptors)

    assert project.check().empty


def test_a_receptor_with_no_kernel_is_listed(tmp_path):
    a, b = _receptor(11), _receptor(12)
    project = _project(tmp_path, [a, b], variants=_KERNELS)
    _kernels(project, [a])

    table = project.check()

    assert list(table["check"]) == ["kernel_missing"]
    assert list(table["receptor"]) == [b.id]
    assert list(table["variant"]) == ["hrrr-ak"]


def test_a_kernel_with_no_receptor_is_listed(tmp_path):
    a, b = _receptor(11), _receptor(12)
    project = _project(tmp_path, [a], variants=_KERNELS)
    _kernels(project, [a, b])

    table = project.check()

    assert list(table["check"]) == ["kernel_unused"]
    assert list(table["receptor"]) == [b.id]


def test_a_kernel_table_that_does_not_exist_is_found_once(tmp_path):
    project = _project(tmp_path, [_receptor(11), _receptor(12)], variants=_KERNELS)

    table = project.check()

    assert list(table["check"]) == ["kernel_table"]
    assert table["receptor"].isna().all()
    assert "'kernels' does not exist" in table["message"].iloc[0]


def test_a_kernel_table_without_its_columns_is_found(tmp_path):
    project = _project(tmp_path, [_receptor(11)], variants=_KERNELS)
    path = project.table_path("kernels")
    path.parent.mkdir()
    pd.DataFrame({"receptor": [_receptor(11).id]}).to_parquet(path)

    table = project.check()

    assert list(table["check"]) == ["kernel_table"]
    assert "lacks columns" in table["message"].iloc[0]


def test_a_kernel_table_given_as_a_file_is_read_from_the_project(tmp_path):
    a, b = _receptor(11), _receptor(12)
    variants = {
        "hrrr-ak": {"transforms": [{"kind": "averaging_kernel", "table": "k.csv"}]}
    }
    project = _project(tmp_path, [a, b], variants=variants)
    pd.DataFrame(
        {"receptor": [a.id] * 2, "level": [0.0, 1.0], "value": [1.0, 1.0]}
    ).to_csv(project.directory / "k.csv", index=False)

    table = project.check()

    assert list(table["check"]) == ["kernel_missing"]
    assert list(table["receptor"]) == [b.id]


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
# More than one thing, and another transport model
# ---------------------------------------------------------------------------


def test_every_check_runs_for_every_variant(tmp_path):
    a, b = _receptor(11), _receptor(20, longitude=-110.0)
    project = _project(tmp_path, [a, b], variants=_KERNELS)
    _kernels(project, [a])

    table = project.check()

    assert set(table["check"]) == {"met_files", "grid", "kernel_missing"}
    assert set(table["receptor"]) == {b.id}


class _NoFilesMet(BaseModel):
    """A met for a model that reads no files."""

    model_config = ConfigDict(extra="allow")

    def settings(self) -> dict[str, Any]:
        return {"source": "toy"}


class _NoFilesModel:
    """A transport model with no ``check_met``."""

    name = "toy"
    config_class = transport.TransportConfig
    met_config_class = _NoFilesMet

    def version(self, config) -> str:
        return "1.0"

    def data_files(self, config) -> None:
        return None

    def run(self, *args, **kwargs):
        raise NotImplementedError


def test_a_model_with_no_files_to_check_has_no_met_findings(tmp_path, monkeypatch):
    monkeypatch.setitem(transport.MODELS, "toy", _NoFilesModel)
    project = _project(
        tmp_path,
        [_receptor(20)],
        variants={"toy": {"model": "toy", "met": "hrrr"}},
    )

    assert project.check().empty


def test_a_project_whose_files_are_invalid_cannot_be_checked(tmp_path):
    project = _project(tmp_path)
    project.config_path.write_text(project.config_path.read_text() + "nonsense: 1\n")

    with pytest.raises(ValueError, match="nonsense"):
        Project(project.directory).check()
