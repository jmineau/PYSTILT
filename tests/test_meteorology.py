"""Tests for stilt.meteorology."""

import datetime as dt
import logging
from pathlib import Path

import pytest

from stilt.errors import MeteorologyError
from stilt.meteorology import MetArchive, MetStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_met(tmp_path: Path, file_format: str, tres: str, n_min: int = 1) -> MetStream:
    return MetStream(
        met_id="hrrr",
        directory=tmp_path,
        file_format=file_format,
        file_tres=tres,
        n_min=n_min,
    )


def _touch_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for name in names:
        f = tmp_path / name
        f.touch()
        files.append(f)
    return files


def test_met_archive_resolves_relative_stream_directories(tmp_path):
    archive = MetArchive(tmp_path / "archive")
    resolved = archive.resolve_directory(Path("hrrr"))
    assert resolved == (tmp_path / "archive" / "hrrr").resolve()


def test_met_archive_keeps_absolute_stream_directories(tmp_path):
    absolute = (tmp_path / "hrrr").resolve()
    archive = MetArchive(tmp_path / "archive")
    resolved = archive.resolve_directory(absolute)
    assert resolved == absolute


def test_met_archive_stage_files_symlinks_or_copies(tmp_path):
    source_dir = tmp_path / "archive"
    staged_dir = tmp_path / "compute" / "met"
    source_dir.mkdir(parents=True)
    src = source_dir / "20230101_12"
    src.write_text("met")

    archive = MetArchive()
    staged = archive.stage_files([src], staged_dir)

    assert staged == [staged_dir / src.name]
    assert staged[0].exists()
    assert staged[0].read_text() == "met"


def test_met_archive_stage_files_deduplicates_duplicate_basenames(tmp_path, caplog):
    source_dir = tmp_path / "archive"
    staged_dir = tmp_path / "compute" / "met"
    first_dir = source_dir / "a"
    second_dir = source_dir / "b"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "20230101_12"
    second = second_dir / "20230101_12"
    first.write_text("first")
    second.write_text("second")

    archive = MetArchive()
    with caplog.at_level(logging.WARNING):
        staged = archive.stage_files([first, second], staged_dir)

    assert staged == [staged_dir / first.name]
    assert staged[0].read_text() == "first"
    assert "duplicate basename" in caplog.text
    assert str(first.resolve()) in caplog.text
    assert str(second.resolve()) in caplog.text


# ---------------------------------------------------------------------------
# Backward run - standard case
# ---------------------------------------------------------------------------


def test_required_files_backward_single_file(tmp_path):
    """Backward 1-h run starting exactly on a 1-h boundary."""
    # Format: %Y%m%d_%H → e.g. "20230101_12"
    _touch_files(tmp_path, ["20230101_11", "20230101_12", "20230101_13"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)
    names = [f.name for f in files]
    assert "20230101_11" in names
    assert "20230101_12" in names


def test_required_files_backward_24h(tmp_path):
    """24-h backward run should span from previous day."""
    names_to_touch = [f"20230101_{h:02d}" for h in range(24)] + [
        f"20221231_{h:02d}" for h in range(24)
    ]
    _touch_files(tmp_path, names_to_touch)
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-24)
    names = [f.name for f in files]
    assert "20230101_12" in names
    assert "20221231_12" in names


def test_required_files_backward_deduplicates(tmp_path):
    """Files should not repeat in the returned list."""
    _touch_files(tmp_path, ["20230101_12"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)
    assert len(files) == len(set(f.name for f in files))


def test_required_files_forward_run(tmp_path):
    """Forward run should include files after the receptor time."""
    _touch_files(tmp_path, ["20230101_12", "20230101_13"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=1)
    names = [f.name for f in files]
    assert "20230101_12" in names
    assert "20230101_13" in names


# ---------------------------------------------------------------------------
# Insufficient files
# ---------------------------------------------------------------------------


def test_required_files_raises_when_no_files(tmp_path):
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    with pytest.raises(MeteorologyError, match="Insufficient"):
        met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)


def test_required_files_raises_when_below_n_min(tmp_path):
    _touch_files(tmp_path, ["20230101_12"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h", n_min=5)
    with pytest.raises(MeteorologyError, match="Insufficient"):
        met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)


# ---------------------------------------------------------------------------
# Lock files are excluded
# ---------------------------------------------------------------------------


def test_required_files_ignores_lock_files(tmp_path):
    _touch_files(tmp_path, ["20230101_12", "20230101_12.lock"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)
    assert all(".lock" not in f.name for f in files)


# ---------------------------------------------------------------------------
# Coarser time resolution (6-hourly)
# ---------------------------------------------------------------------------


def test_required_files_6h_resolution(tmp_path):
    """6-h met files: 12-h backward from 2023-01-01 12Z."""
    _touch_files(tmp_path, ["2023010100", "2023010106", "2023010112"])
    met = _make_met(tmp_path, "%Y%m%d%H", "6h")
    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-12)
    names = [f.name for f in files]
    assert "2023010100" in names
    assert "2023010106" in names
    assert "2023010112" in names


def test_required_files_match_multi_hour_filename_prefixes(tmp_path):
    _touch_files(
        tmp_path,
        [
            "20240531_00-05_hrrr",
            "20240531_06-11_hrrr",
            "20240531_12-17_hrrr",
            "20240531_18-23_hrrr",
            "20240601_00-05_hrrr",
        ],
    )
    met = _make_met(tmp_path, "%Y%m%d_%H", "6 hours")

    files = met.required_files(r_time=dt.datetime(2024, 6, 1, 0), n_hours=-24)

    assert [f.name for f in files] == [
        "20240531_00-05_hrrr",
        "20240531_06-11_hrrr",
        "20240531_12-17_hrrr",
        "20240531_18-23_hrrr",
        "20240601_00-05_hrrr",
    ]


def test_required_files_searches_recursively(tmp_path):
    nested = tmp_path / "2024" / "06"
    nested.mkdir(parents=True)
    _touch_files(nested, ["20240601_00-05_hrrr"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "6 hours")

    files = met.required_files(r_time=dt.datetime(2024, 6, 1, 0), n_hours=-1)

    assert [f.name for f in files] == ["20240601_00-05_hrrr"]


def test_required_files_deduplicates_root_symlink_and_nested_file(tmp_path):
    nested = tmp_path / "2021" / "06"
    nested.mkdir(parents=True)
    target = nested / "20210601_00-05_hrrr"
    target.touch()
    (tmp_path / "20210601_00-05_hrrr").symlink_to(target)
    met = _make_met(tmp_path, "%Y%m%d_%H", "6 hours")

    files = met.required_files(r_time=dt.datetime(2021, 6, 1, 0), n_hours=-1)

    assert len(files) == 1
    assert files[0].name == "20210601_00-05_hrrr"


def test_available_files_is_cached(monkeypatch, tmp_path):
    met = _make_met(tmp_path, "*", "1h")
    calls = {"n": 0}

    def fake_rglob(self, _pattern):
        calls["n"] += 1
        return [tmp_path / "a", tmp_path / "b"]

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)

    first = met.available_files
    second = met.available_files

    assert first == second
    assert calls["n"] == 1


def test_required_files_reports_broken_symlink_matches(tmp_path):
    broken = tmp_path / "20240601_00-05_hrrr"
    broken.symlink_to(tmp_path / "2024" / "06" / "20240601_00-05_hrrr")
    met = _make_met(tmp_path, "%Y%m%d_%H", "6 hours")

    with pytest.raises(MeteorologyError, match="broken symlinks"):
        met.required_files(r_time=dt.datetime(2024, 6, 1, 0), n_hours=-1)


def test_required_files_backward_non_boundary_includes_ceil_file(tmp_path):
    _touch_files(tmp_path, ["20230101_11", "20230101_12", "20230101_13"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")

    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12, 30), n_hours=-1)
    names = [f.name for f in files]

    assert "20230101_11" in names
    assert "20230101_12" in names
    assert "20230101_13" in names


def test_met_stream_stage_files_for_simulation_uses_archive_root(tmp_path):
    archive_root = tmp_path / "archive"
    met_dir = archive_root / "hrrr"
    met_dir.mkdir(parents=True)
    source = _touch_files(met_dir, ["20230101_11", "20230101_12"])

    met = MetStream(
        met_id="hrrr",
        directory="hrrr",
        file_format="%Y%m%d_%H",
        file_tres="1h",
        archive=MetArchive(archive_root),
    )

    staged = met.stage_files_for_simulation(
        r_time=dt.datetime(2023, 1, 1, 12),
        n_hours=-1,
        target_dir=tmp_path / "compute" / "met",
    )

    assert [p.name for p in staged] == [p.name for p in source]
    assert all(p.parent == tmp_path / "compute" / "met" for p in staged)
