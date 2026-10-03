"""Tests for stilt.meteorology."""

import datetime as dt
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stilt.config.meteorology import MetConfig
from stilt.exceptions import MeteorologyError
from stilt.meteorology import Met
from stilt.spatial import Bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_met(tmp_path: Path, file_format: str, tres: str, n_min: int = 1) -> Met:
    return Met(
        "hrrr",
        MetConfig(
            directory=tmp_path, file_format=file_format, file_tres=tres, n_min=n_min
        ),
    )


# ---------------------------------------------------------------------------
# MetConfig validation
# ---------------------------------------------------------------------------


def test_metconfig_local_files_require_file_format(tmp_path):
    """Local files (no download) require file_format and file_tres."""
    with pytest.raises(Exception, match="file_format and file_tres are required"):
        MetConfig(directory=tmp_path)


def test_metconfig_local_files_valid(tmp_path):
    cfg = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    assert cfg.file_format == "%Y%m%d_%H"
    assert cfg.download is None


def test_metconfig_download_needs_no_file_format(tmp_path):
    """Downloading does not require file_format or file_tres."""
    cfg = MetConfig(directory=tmp_path, download="hrrr")
    assert cfg.download == "hrrr"
    assert cfg.file_format is None


def test_metconfig_unknown_archive_raises(tmp_path):
    with pytest.raises(Exception, match="Unknown ARL archive"):
        MetConfig(directory=tmp_path, download="bogus_product")


def test_metconfig_subgrid_requires_bounds(tmp_path):
    with pytest.raises(Exception, match="subgrid_bounds is required"):
        MetConfig(
            directory=tmp_path,
            download="hrrr",
            subgrid_enable=True,
        )


def test_metconfig_subgrid_valid(tmp_path):
    cfg = MetConfig(
        directory=tmp_path,
        download="hrrr",
        subgrid_enable=True,
        subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
    )
    assert cfg.subgrid_enable is True


def test_metconfig_extra_fields_as_download_options(tmp_path):
    """Extra inline fields land in download_options (for e.g. NAMSSource domain)."""
    cfg = MetConfig(directory=tmp_path, download="nams", domain="ak")
    assert cfg.download_options == {"domain": "ak"}


def test_metconfig_rejects_an_unknown_key_without_download(tmp_path):
    """A typo in a plain met entry is an error, as elsewhere in config.yaml (#52)."""
    with pytest.raises(ValueError, match="subgrid_enabel"):
        MetConfig(
            directory=tmp_path,
            file_format="%Y%m%d_%H",
            file_tres="6h",
            subgrid_enabel=True,
        )


def test_metconfig_rejects_an_option_the_archive_does_not_take(tmp_path):
    with pytest.raises(ValueError, match="does not take"):
        MetConfig(directory=tmp_path, download="hrrr", domain="ak")
    with pytest.raises(ValueError, match="does not take"):
        MetConfig(directory=tmp_path, download="nams", domian="ak")


# ---------------------------------------------------------------------------
# Met download (via arlmet)
# ---------------------------------------------------------------------------


def _make_download_met(tmp_path: Path, download: str = "hrrr", **kwargs) -> Met:
    return Met(download, MetConfig(directory=tmp_path, download=download, **kwargs))


def test_met_download_calls_fetch(tmp_path):
    """With download, required_files delegates to the arlmet archive's fetch()."""
    mock_archive = MagicMock()
    mock_archive.fetch.return_value = [tmp_path / "file1", tmp_path / "file2"]
    for f in mock_archive.fetch.return_value:
        f.touch()

    met = _make_download_met(tmp_path)
    met._archive = mock_archive  # inject mock

    files = met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    mock_archive.fetch.assert_called_once()
    call_kwargs = mock_archive.fetch.call_args
    assert call_kwargs.kwargs["dest_dir"] == tmp_path
    assert call_kwargs.kwargs["mirror"] == "s3"
    assert call_kwargs.kwargs["bbox"] is None
    assert call_kwargs.kwargs["levels"] is None
    assert len(files) == 2


def test_met_download_from_is_passed_to_fetch(tmp_path):
    mock_archive = MagicMock()
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    met = _make_download_met(tmp_path, download_from="ftp")
    met._archive = mock_archive
    met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    assert mock_archive.fetch.call_args.kwargs["mirror"] == "ftp"


def test_met_download_with_subgrid_passes_bbox(tmp_path):
    """download + subgrid_enable passes bbox to arlmet fetch."""
    mock_archive = MagicMock()
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    bounds = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)
    met = Met(
        "hrrr",
        MetConfig(
            directory=tmp_path,
            download="hrrr",
            subgrid_enable=True,
            subgrid_bounds=bounds,
            subgrid_buffer=0.5,
        ),
    )
    met._archive = mock_archive

    met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    bbox = mock_archive.fetch.call_args.kwargs["bbox"]
    assert bbox == (-114.5, 38.5, -109.5, 42.5)
    assert mock_archive.fetch.call_args.kwargs["levels"] is None


def test_met_download_passes_subgrid_levels(tmp_path):
    """download + subgrid_levels=N asks arlmet to keep the lowest N levels."""
    mock_archive = MagicMock()
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    met = _make_download_met(
        tmp_path,
        subgrid_enable=True,
        subgrid_bounds=Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0),
        subgrid_levels=3,
    )
    met._archive = mock_archive

    met.required_files(r_time="2024-07-18 12:00", n_hours=-24)

    assert mock_archive.fetch.call_args.kwargs["levels"] == [0, 1, 2]


def test_met_download_n_min_raises(tmp_path):
    """MeteorologyError when fetch returns fewer files than n_min."""
    mock_archive = MagicMock()
    mock_archive.fetch.return_value = []

    met = _make_download_met(tmp_path, n_min=2)
    met._archive = mock_archive

    with pytest.raises(MeteorologyError, match="Insufficient"):
        met.required_files(r_time="2024-07-18 12:00", n_hours=-24)


# ---------------------------------------------------------------------------
# Met archive subsetting via arlmet.extract_subset
# ---------------------------------------------------------------------------


BOUNDS = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)


def _archive_met(tmp_path: Path, **kwargs) -> Met:
    """A local archive holding one 1 h file, cropped into tmp_path/crops."""
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "20230101_12").write_text("met")
    settings = {
        "subgrid_bounds": BOUNDS,
        "subgrid_buffer": 0.0,
        "subgrid_dir": tmp_path / "crops",
        **kwargs,
    }
    return Met(
        "hrrr",
        MetConfig(
            directory=archive,
            file_format="%Y%m%d_%H",
            file_tres="1h",
            subgrid_enable=True,
            **settings,
        ),
    )


def _files(met: Met) -> list[Path]:
    """Return the files one simulation at the test time reads."""
    return met.readable(
        met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)
    )


def _fake_extract(text: str = "cropped"):
    """Stand-in for arlmet.extract_subset that writes *text* to the destination."""

    def extract(src, dst, **kwargs):
        Path(dst).write_text(text)
        return Path(dst)

    return extract


def test_met_local_subgrid_calls_extract_subset(tmp_path):
    """Cropping local files crops into crop_dir and hands back the crop."""
    met = _archive_met(tmp_path)
    with patch("arlmet.extract_subset", side_effect=_fake_extract()) as mock_extract:
        files = _files(met)

    mock_extract.assert_called_once()
    call_args = mock_extract.call_args
    assert call_args.args[0] == (tmp_path / "archive" / "20230101_12").resolve()
    assert call_args.kwargs["bbox"] == (-114.0, 39.0, -110.0, 42.0)
    assert files == [met.crop_dir / "20230101_12"]
    assert files[0].read_text() == "cropped"


def test_met_local_subgrid_reuses_cache(tmp_path):
    """extract_subset is not called again when the crop already exists."""
    met = _archive_met(tmp_path)
    met.crop_dir.mkdir(parents=True)
    (met.crop_dir / "20230101_12").write_text("cached")

    with patch("arlmet.extract_subset") as mock_extract:
        files = _files(met)

    mock_extract.assert_not_called()
    assert files[0].read_text() == "cached"


def test_met_local_subgrid_levels(tmp_path):
    """subgrid_levels=N passes levels=list(range(N)) to extract_subset."""
    met = _archive_met(tmp_path, subgrid_levels=5)
    with patch("arlmet.extract_subset", side_effect=_fake_extract()) as mock_extract:
        _files(met)

    assert mock_extract.call_args.kwargs["levels"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "change",
    [
        {"subgrid_bounds": Bounds(xmin=-113.0, xmax=-110.0, ymin=39.0, ymax=42.0)},
        {"subgrid_buffer": 0.5},
        {"subgrid_levels": 20},
    ],
)
def test_changing_the_crop_changes_crop_dir(tmp_path, change):
    """A new crop box or level count never reuses crops made for another (#53)."""
    old = _archive_met(tmp_path)
    new = _archive_met(tmp_path, **change)
    assert new.crop_dir != old.crop_dir
    assert new.crop_dir.parent == old.crop_dir.parent == tmp_path / "crops"

    with patch("arlmet.extract_subset", side_effect=_fake_extract("old")):
        _files(old)
    with patch(
        "arlmet.extract_subset", side_effect=_fake_extract("new")
    ) as mock_extract:
        files = _files(new)

    mock_extract.assert_called_once()
    assert files[0].read_text() == "new"


def test_the_same_crop_shares_crop_dir(tmp_path):
    """Crop settings that give the same box share one folder, whatever the archive."""
    wide = Bounds(xmin=-114.5, xmax=-109.5, ymin=38.5, ymax=42.5)
    a = _archive_met(tmp_path, subgrid_buffer=0.5)
    b = _archive_met(
        tmp_path / "other", subgrid_bounds=wide, subgrid_dir=tmp_path / "crops"
    )
    assert a.crop_dir == b.crop_dir


def test_a_crop_appears_only_when_complete(tmp_path):
    """The crop is written to a temporary name and renamed into place (#53)."""
    met = _archive_met(tmp_path)
    final = met.crop_dir / "20230101_12"

    def extract(src, dst, **kwargs):
        assert Path(dst).parent == met.crop_dir
        assert Path(dst) != final
        Path(dst).write_text("half")
        assert not final.exists()
        Path(dst).write_text("cropped")
        return Path(dst)

    with patch("arlmet.extract_subset", side_effect=extract):
        _files(met)

    assert final.read_text() == "cropped"
    assert sorted(p.name for p in met.crop_dir.iterdir()) == ["20230101_12"]


def test_a_failed_crop_leaves_nothing_behind(tmp_path):
    """A crop that fails partway leaves no file, so the next run crops again."""
    met = _archive_met(tmp_path)

    def extract(src, dst, **kwargs):
        Path(dst).write_text("half")
        raise OSError("disk full")

    with (
        patch("arlmet.extract_subset", side_effect=extract),
        pytest.raises(OSError, match="disk full"),
    ):
        _files(met)

    assert list(met.crop_dir.iterdir()) == []


def test_a_project_needs_subgrid_dir_to_crop_local_files(tmp_path):
    """Crops are never written into the met archive by default (#53)."""
    from stilt.config import ProjectConfig

    met = MetConfig(
        directory=tmp_path,
        file_format="%Y%m%d_%H",
        file_tres="1h",
        subgrid_enable=True,
        subgrid_bounds=BOUNDS,
    )
    with pytest.raises(ValueError, match="subgrid_dir is required"):
        ProjectConfig(mets={"hrrr": met})


def test_a_project_needs_each_met_directory(tmp_path):
    """A met config without its directory reads back from a stored record, but a project needs it."""
    from stilt.config import ProjectConfig

    met = MetConfig(file_format="%Y%m%d_%H", file_tres="1h")
    assert met.directory is None
    with pytest.raises(ValueError, match="'hrrr' needs a directory"):
        ProjectConfig(mets={"hrrr": met})
    with pytest.raises(ValueError, match="has no directory"):
        Met("hrrr", met)


def test_metconfig_download_crop_needs_no_subgrid_dir(tmp_path):
    cfg = MetConfig(
        directory=tmp_path, download="hrrr", subgrid_enable=True, subgrid_bounds=BOUNDS
    )
    assert cfg.subgrid_dir is None


def _touch_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for name in names:
        f = tmp_path / name
        f.touch()
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# The files a run reads
# ---------------------------------------------------------------------------


def test_readable_returns_local_files_where_they_are(tmp_path):
    """Without cropping, HYSPLIT reads the met files in place: nothing is files."""
    source_dir = tmp_path / "met"
    source_dir.mkdir(parents=True)
    src = source_dir / "20230101_12"
    src.write_text("met")

    met = _make_met(source_dir, "%Y%m%d_%H", "1h")

    assert met.readable([src]) == [src]
    assert met.readable(
        met.required_files(r_time=dt.datetime(2023, 1, 1, 12), n_hours=-1)
    ) == [src]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["met"]


def test_readable_keeps_the_first_of_two_files_with_one_name(tmp_path, caplog):
    source_dir = tmp_path / "met"
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
        files = met.readable([first, second])

    assert files == [first]
    assert "duplicate basename" in caplog.text
    assert str(first) in caplog.text
    assert str(second) in caplog.text


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


def _six_hourly_hrrr(tmp_path):
    _touch_files(
        tmp_path,
        [
            "20190122_18-23_hrrr",
            "20190123_00-05_hrrr",
            "20190123_06-11_hrrr",
            "20190123_12-17_hrrr",
            "20190123_18-23_hrrr",
            "20190124_00-05_hrrr",
        ],
    )
    return _make_met(tmp_path, "%Y%m%d_%H", "6h")


def test_required_files_backward_mid_file_release_skips_next_file(tmp_path):
    """A 19:06 release sits inside 18-23, so the next day's 00z file is not needed (#29)."""
    met = _six_hourly_hrrr(tmp_path)
    files = met.required_files(r_time=dt.datetime(2019, 1, 23, 19, 6), n_hours=-24)
    assert [f.name for f in files] == [
        "20190122_18-23_hrrr",
        "20190123_00-05_hrrr",
        "20190123_06-11_hrrr",
        "20190123_12-17_hrrr",
        "20190123_18-23_hrrr",
    ]


def test_required_files_backward_last_hour_release_includes_next_file(tmp_path):
    """A 23:06 release interpolates against 00z, which is in the next file."""
    met = _six_hourly_hrrr(tmp_path)
    files = met.required_files(r_time=dt.datetime(2019, 1, 23, 23, 6), n_hours=-24)
    assert files[-1].name == "20190124_00-05_hrrr"


def test_required_files_backward_boundary_release_adds_nothing_later(tmp_path):
    met = _six_hourly_hrrr(tmp_path)
    files = met.required_files(r_time=dt.datetime(2019, 1, 23, 18), n_hours=-24)
    assert files[-1].name == "20190123_18-23_hrrr"


def test_required_files_ignores_backup_copies(tmp_path):
    """Archive backups (name~<timestamp>~) must not be staged beside the real file (#30)."""
    _touch_files(
        tmp_path,
        ["20200107_18-23_hrrr", "20200107_18-23_hrrr~20260403182134~"],
    )
    met = _make_met(tmp_path, "%Y%m%d_%H", "6h")
    files = met.required_files(r_time=dt.datetime(2020, 1, 7, 20), n_hours=-1)
    assert [f.name for f in files] == ["20200107_18-23_hrrr"]
