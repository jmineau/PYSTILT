"""Tests for stilt.meteorology."""

import datetime as dt
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stilt.config.meteorology import MetConfig
from stilt.config.spatial import Bounds
from stilt.errors import MeteorologyError
from stilt.meteorology import MetStream

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


# ---------------------------------------------------------------------------
# MetConfig validation
# ---------------------------------------------------------------------------


def test_metconfig_archive_mode_requires_file_format(tmp_path):
    """Archive mode (no source) requires file_format and file_tres."""
    with pytest.raises(Exception, match="file_format and file_tres are required"):
        MetConfig(directory=tmp_path)


def test_metconfig_archive_mode_valid(tmp_path):
    cfg = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    assert cfg.file_format == "%Y%m%d_%H"
    assert cfg.source is None


def test_metconfig_source_mode_no_file_format_needed(tmp_path):
    """Source mode does not require file_format or file_tres."""
    cfg = MetConfig(directory=tmp_path, source="hrrr")
    assert cfg.source == "hrrr"
    assert cfg.file_format is None


def test_metconfig_unknown_source_raises(tmp_path):
    with pytest.raises(Exception, match="Unknown arlmet source"):
        MetConfig(directory=tmp_path, source="bogus_product")


def test_metconfig_subgrid_requires_bounds(tmp_path):
    with pytest.raises(Exception, match="subgrid_bounds is required"):
        MetConfig(
            directory=tmp_path,
            source="hrrr",
            subgrid_enable=True,
        )


def test_metconfig_subgrid_valid(tmp_path):
    cfg = MetConfig(
        directory=tmp_path,
        source="hrrr",
        subgrid_enable=True,
        subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
    )
    assert cfg.subgrid_enable is True


def test_metconfig_extra_fields_as_source_kwargs(tmp_path):
    """Extra inline fields land in source_kwargs (for e.g. NAMSSource domain)."""
    cfg = MetConfig(directory=tmp_path, source="nams", domain="ak")
    assert cfg.source_kwargs == {"domain": "ak"}


# ---------------------------------------------------------------------------
# MetStream source mode (download via arlmet)
# ---------------------------------------------------------------------------


def _make_source_met(tmp_path: Path, source: str = "hrrr", **kwargs) -> MetStream:
    return MetStream(
        met_id=source,
        directory=tmp_path,
        source_type=source,
        **kwargs,
    )


def test_metsource_download_calls_fetch(tmp_path):
    """In source mode, required_files delegates to arlmet source.fetch()."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = [tmp_path / "file1", tmp_path / "file2"]
    for f in mock_source.fetch.return_value:
        f.touch()

    met = _make_source_met(tmp_path)
    met._arlmet_source = mock_source  # inject mock

    files = met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    mock_source.fetch.assert_called_once()
    call_kwargs = mock_source.fetch.call_args
    assert call_kwargs.kwargs["local_dir"] == tmp_path
    assert call_kwargs.kwargs["backend"] == "s3"
    assert call_kwargs.kwargs["bbox"] is None
    assert len(files) == 2


def test_metsource_download_with_subgrid_passes_bbox(tmp_path):
    """source mode + subgrid_enable passes bbox to arlmet fetch."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=tmp_path,
        source_type="hrrr",
        subgrid_enable=True,
        subgrid_bounds=bounds,
        subgrid_buffer=0.5,
    )
    met._arlmet_source = mock_source

    met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    bbox = mock_source.fetch.call_args.kwargs["bbox"]
    assert bbox == (-114.5, 38.5, -109.5, 42.5)


def test_metsource_download_n_min_raises(tmp_path):
    """MeteorologyError when fetch returns fewer files than n_min."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = []

    met = _make_source_met(tmp_path, n_min=2)
    met._arlmet_source = mock_source

    with pytest.raises(MeteorologyError, match="Insufficient"):
        met.required_files(r_time="2024-07-18 12:00", n_hours=-24)


# ---------------------------------------------------------------------------
# MetStream archive subsetting via arlmet.extract_subset
# ---------------------------------------------------------------------------


def test_metsource_archive_subgrid_calls_extract_subset(tmp_path):
    """Archive-mode subsetting calls extract_subset and stages the cached copy."""
    source_dir = tmp_path / "archive"
    source_dir.mkdir()
    src_file = source_dir / "20230101_12"
    src_file.write_text("met")

    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=bounds,
        subgrid_buffer=0.0,
    )

    target_dir = tmp_path / "sim" / "met"
    with patch("arlmet.extract_subset") as mock_extract:
        # Make extract_subset create the cache file so staging can proceed
        def _fake_extract(src, dst, **kw):
            dst.write_text("subsetted")

        mock_extract.side_effect = _fake_extract
        staged = met.stage_files_for_simulation(
            r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1, target_dir=target_dir
        )

    mock_extract.assert_called_once()
    call_args = mock_extract.call_args
    assert call_args.args[0] == src_file.resolve()
    assert call_args.kwargs["bbox"] == (-114.0, 39.0, -110.0, 42.0)
    # staged file should be in target_dir
    assert len(staged) == 1
    assert staged[0].parent == target_dir


def test_metsource_archive_subgrid_reuses_cache(tmp_path):
    """extract_subset is not called again when the cached file already exists."""
    source_dir = tmp_path / "archive"
    source_dir.mkdir()
    src_file = source_dir / "20230101_12"
    src_file.write_text("met")

    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=bounds,
    )

    # Pre-populate the cache
    subgrid_dir = met._resolved_subgrid_dir()
    subgrid_dir.mkdir(parents=True)
    (subgrid_dir / src_file.name).write_text("cached")

    target_dir = tmp_path / "sim" / "met"
    with patch("arlmet.extract_subset") as mock_extract:
        met.stage_files_for_simulation(
            r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1, target_dir=target_dir
        )

    mock_extract.assert_not_called()


def test_metsource_archive_subgrid_levels(tmp_path):
    """subgrid_levels=N passes levels=list(range(N)) to extract_subset."""
    source_dir = tmp_path / "archive"
    source_dir.mkdir()
    src_file = source_dir / "20230101_12"
    src_file.write_text("met")

    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=bounds,
        subgrid_levels=5,
    )

    target_dir = tmp_path / "sim" / "met"
    with patch("arlmet.extract_subset") as mock_extract:

        def _fake_extract(src, dst, **kw):
            dst.write_text("subsetted")

        mock_extract.side_effect = _fake_extract
        met.stage_files_for_simulation(
            r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1, target_dir=target_dir
        )

    assert mock_extract.call_args.kwargs["levels"] == [0, 1, 2, 3, 4]


def test_metsource_archive_subgrid_auto_dir(tmp_path):
    """subgrid_dir defaults to directory/subgrid when not set."""
    source_dir = tmp_path / "archive"
    source_dir.mkdir()
    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=bounds,
    )
    assert met._resolved_subgrid_dir() == source_dir / "subgrid"


def test_metsource_archive_subgrid_custom_dir(tmp_path):
    """subgrid_dir is used when explicitly set."""
    source_dir = tmp_path / "archive"
    custom_dir = tmp_path / "custom_subgrid"
    source_dir.mkdir()
    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = MetStream(
        met_id="hrrr",
        directory=source_dir,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=bounds,
        subgrid_dir=custom_dir,
    )
    assert met._resolved_subgrid_dir() == custom_dir


def _touch_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for name in names:
        f = tmp_path / name
        f.touch()
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def test_stage_files_symlinks_or_copies(tmp_path):
    source_dir = tmp_path / "met"
    staged_dir = tmp_path / "compute" / "met"
    source_dir.mkdir(parents=True)
    src = source_dir / "20230101_12"
    src.write_text("met")

    met = _make_met(source_dir, "%Y%m%d_%H", "1h")
    staged = met._stage_files([src], staged_dir)

    assert staged == [staged_dir / src.name]
    assert staged[0].exists()
    assert staged[0].read_text() == "met"


def test_stage_files_deduplicates_duplicate_basenames(tmp_path, caplog):
    source_dir = tmp_path / "met"
    staged_dir = tmp_path / "compute" / "met"
    first_dir = source_dir / "a"
    second_dir = source_dir / "b"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "20230101_12"
    second = second_dir / "20230101_12"
    first.write_text("first")
    second.write_text("second")

    met = _make_met(source_dir, "%Y%m%d_%H", "1h")
    with caplog.at_level(logging.WARNING):
        staged = met._stage_files([first, second], staged_dir)

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


def test_required_files_error_reports_missing_patterns(tmp_path):
    """Error message names the unmatched pattern and the directory."""
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")
    with pytest.raises(MeteorologyError, match="Patterns not found"):
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


def test_required_files_backward_non_boundary_includes_ceil_file(tmp_path):
    _touch_files(tmp_path, ["20230101_11", "20230101_12", "20230101_13"])
    met = _make_met(tmp_path, "%Y%m%d_%H", "1h")

    files = met.required_files(r_time=dt.datetime(2023, 1, 1, 12, 30), n_hours=-1)
    names = [f.name for f in files]

    assert "20230101_11" in names
    assert "20230101_12" in names
    assert "20230101_13" in names
