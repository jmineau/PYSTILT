"""Tests for what a folder's ARL files hold (stilt.transport.hysplit.catalog)."""

from pathlib import Path

import pandas as pd
import pytest

from stilt.transport.hysplit import Met, MetConfig
from stilt.transport.hysplit import catalog as catalog_module
from stilt.transport.hysplit.catalog import Catalog, catalog_path

from ...fixtures.factories import write_arl_file

WINDOW = (pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-01-01 02:00"))


def _block(directory: Path, name: str = "block") -> Path:
    """Write a file of the 6 hours from 2023-01-01 00:00."""
    times = pd.date_range("2023-01-01 00:00", periods=6, freq="h")
    return write_arl_file(directory / name, times=times)


def _catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "met", tmp_path / "kept" / "catalog.parquet")


def _no_reads(monkeypatch) -> None:
    """Fail a test that reads a header."""
    import arlmet

    def refuse(path):
        raise AssertionError(f"read {path}")

    monkeypatch.setattr(arlmet, "probe", refuse)


# ---------------------------------------------------------------------------
# Where the table is kept
# ---------------------------------------------------------------------------


def test_the_table_is_kept_with_the_crops(tmp_path):
    path = catalog_path(Path("/met/hrrr"), tmp_path / "crops")
    assert path.parent == tmp_path / "crops" / "catalogs"


def test_without_crops_the_table_is_kept_in_the_users_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = catalog_path(Path("/met/hrrr"), None)
    assert path.parent == tmp_path / "pystilt" / "met_catalogs"


def test_without_a_cache_variable_the_table_is_kept_in_the_home_cache(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME")
    path = catalog_path(Path("/met/hrrr"), None)
    assert path.parent == Path("~/.cache/pystilt/met_catalogs").expanduser()


def test_each_folder_has_its_own_table(tmp_path):
    assert catalog_path(Path("/a"), tmp_path) != catalog_path(Path("/b"), tmp_path)


def test_a_met_keeps_its_table_where_a_project_says(tmp_path):
    """A relative ``subgrid_dir`` was made absolute by the project, as ``directory`` was."""
    _block(tmp_path / "met")
    config = MetConfig(directory=tmp_path / "met", subgrid_dir=tmp_path / "crops")
    Met("hrrr", config).files_for(WINDOW)
    assert list((tmp_path / "crops" / "catalogs").glob("*.parquet"))


# ---------------------------------------------------------------------------
# Reading the table and refreshing it
# ---------------------------------------------------------------------------


def test_a_new_catalog_reads_the_kept_table(tmp_path, monkeypatch):
    block = _block(tmp_path / "met")
    assert _catalog(tmp_path).cover(*WINDOW).files == [block]
    _no_reads(monkeypatch)

    assert _catalog(tmp_path).cover(*WINDOW).files == [block]


def test_a_refresh_reads_only_the_new_files(tmp_path, monkeypatch):
    import arlmet

    _block(tmp_path / "met")
    _catalog(tmp_path).refresh()
    new = tmp_path / "met" / "new"
    times = pd.date_range("2023-01-01 06:00", periods=6, freq="h")
    write_arl_file(new, times=times)
    read: list[str] = []
    real = arlmet.probe
    monkeypatch.setattr(arlmet, "probe", lambda p: read.append(p) or real(p))

    table = _catalog(tmp_path).refresh()

    assert read == [str(new)]
    assert len(table) == 2


def test_a_file_that_is_not_arl_is_read_once(tmp_path, monkeypatch):
    (tmp_path / "met").mkdir()
    (tmp_path / "met" / "README").write_text("not a met file")
    assert _catalog(tmp_path).refresh().empty
    _no_reads(monkeypatch)

    assert _catalog(tmp_path).refresh().empty


def test_a_gap_looks_in_the_folder_once(tmp_path, monkeypatch):
    """A run missing its files lists the folder, and later runs in the process do not."""
    _block(tmp_path / "met")
    catalog = _catalog(tmp_path)
    late = (pd.Timestamp("2023-01-02 01:00"), pd.Timestamp("2023-01-02 02:00"))
    walks: list[Path] = []
    real = catalog_module.walk
    monkeypatch.setattr(
        catalog_module, "walk", lambda top: walks.append(top) or real(top)
    )

    assert catalog.cover(*late).gaps
    assert catalog.cover(*late).gaps
    assert len(walks) == 1


def test_a_file_added_after_the_table_was_kept_is_found(tmp_path):
    _block(tmp_path / "met")
    _catalog(tmp_path).refresh()
    times = pd.date_range("2023-01-02 00:00", periods=6, freq="h")
    new = write_arl_file(tmp_path / "met" / "new", times=times)
    late = (pd.Timestamp("2023-01-02 01:00"), pd.Timestamp("2023-01-02 02:00"))

    assert _catalog(tmp_path).cover(*late).files == [new]


def test_a_file_removed_after_the_table_was_kept_is_missing(tmp_path):
    block = _block(tmp_path / "met")
    _catalog(tmp_path).refresh()
    block.unlink()

    assert _catalog(tmp_path).cover(*WINDOW).gaps == [WINDOW]


def test_a_file_replaced_after_the_table_was_kept_is_read_again(tmp_path):
    """A crop made again with more hours is seen without being told."""
    path = tmp_path / "met" / "block"
    write_arl_file(path, times=pd.date_range("2023-01-01 00:00", periods=2, freq="h"))
    _catalog(tmp_path).refresh()
    _block(tmp_path / "met")

    assert not _catalog(tmp_path).cover(*WINDOW).gaps


def test_a_catalog_of_another_version_is_read_again(tmp_path):
    block = _block(tmp_path / "met")
    catalog = _catalog(tmp_path)
    catalog.path.parent.mkdir(parents=True)
    pd.DataFrame({"path": ["x"]}).to_parquet(catalog.path)

    assert catalog.cover(*WINDOW).files == [block]


def test_a_catalog_that_cannot_be_read_is_read_again(tmp_path):
    block = _block(tmp_path / "met")
    catalog = _catalog(tmp_path)
    catalog.path.parent.mkdir(parents=True)
    catalog.path.write_text("not parquet")

    assert catalog.cover(*WINDOW).files == [block]


def test_nowhere_to_keep_the_table_still_serves_the_process(tmp_path, caplog):
    block = _block(tmp_path / "met")
    (tmp_path / "kept").write_text("a file where the folder would go")

    assert _catalog(tmp_path).cover(*WINDOW).files == [block]


# ---------------------------------------------------------------------------
# The first table, built once
# ---------------------------------------------------------------------------


def _lock(catalog: Catalog) -> Path:
    lock = catalog.path.with_name(f"{catalog.path.name}.lock")
    lock.mkdir(parents=True)
    return lock


def test_the_lock_is_gone_after_the_first_build(tmp_path):
    _block(tmp_path / "met")
    catalog = _catalog(tmp_path)
    catalog.cover(*WINDOW)
    assert [p.suffix for p in catalog.path.parent.iterdir()] == [".parquet"]


def test_a_process_waits_for_the_one_building_the_first_table(tmp_path, monkeypatch):
    """It reads the other's table when the lock goes, and opens no file itself."""
    block = _block(tmp_path / "met")
    _catalog(tmp_path).refresh()
    catalog = _catalog(tmp_path)
    kept = catalog.path.read_bytes()
    catalog.path.unlink()
    lock = _lock(catalog)

    def built(_seconds):
        catalog.path.write_bytes(kept)
        lock.rmdir()

    monkeypatch.setattr(catalog_module.time, "sleep", built)
    _no_reads(monkeypatch)

    assert catalog.cover(*WINDOW).files == [block]


def test_a_lock_left_by_a_killed_job_only_delays_the_build(tmp_path, monkeypatch):
    block = _block(tmp_path / "met")
    catalog = _catalog(tmp_path)
    _lock(catalog)
    monkeypatch.setattr(catalog_module, "_BUILD_WAIT", 0.0)

    assert catalog.cover(*WINDOW).files == [block]


# ---------------------------------------------------------------------------
# What covers a window
# ---------------------------------------------------------------------------


def test_a_gap_older_than_a_step_does_not_reach_back_to_older_files(tmp_path):
    """An old file is no use across a gap: only what a run could read is chosen."""
    old = pd.date_range("2000-01-01 00:00", periods=6, freq="h")
    write_arl_file(tmp_path / "met" / "old", times=old)
    late = (pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-01-01 02:00"))

    cover = _catalog(tmp_path).cover(*late)

    assert cover.files == []
    assert cover.gaps == [late]


@pytest.mark.parametrize(
    ("start", "end", "gaps"),
    [
        # Before the first file, and past the last
        (
            "2023-01-01 00:00",
            "2023-01-01 03:00",
            [("2023-01-01 00:00", "2023-01-01 02:00")],
        ),
        (
            "2023-01-01 06:00",
            "2023-01-01 09:00",
            [("2023-01-01 07:00", "2023-01-01 09:00")],
        ),
        # A whole file missing between two, reported between the steps either side
        (
            "2023-01-01 03:00",
            "2023-01-01 16:00",
            [("2023-01-01 07:00", "2023-01-01 14:00")],
        ),
    ],
)
def test_gaps_are_the_stretches_with_no_met_step(tmp_path, start, end, gaps):
    for first in ("2023-01-01 02:00", "2023-01-01 14:00"):
        name = first.replace(" ", "_").replace(":", "")
        write_arl_file(
            tmp_path / "met" / name,
            times=pd.date_range(first, periods=6, freq="h"),
        )

    cover = _catalog(tmp_path).cover(pd.Timestamp(start), pd.Timestamp(end))

    assert cover.gaps == [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in gaps]


def test_a_file_cut_partway_through_a_step_is_named(tmp_path):
    block = _block(tmp_path / "met")
    block.write_bytes(block.read_bytes()[:-10])

    cover = _catalog(tmp_path).cover(*WINDOW)

    assert cover.files == [block]
    assert cover.cut == [block]


def test_a_file_of_one_period_is_named(tmp_path):
    single = write_arl_file(tmp_path / "met" / "one", times=["2023-01-01 01:00"])
    other = pd.date_range("2023-01-01 02:00", periods=6, freq="h")
    write_arl_file(tmp_path / "met" / "other", times=other)

    assert _catalog(tmp_path).cover(*WINDOW).single == [single]
