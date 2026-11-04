"""Tests for HYSPLIT's met config and its files (stilt.transport.hysplit.met)."""

import datetime as dt
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stilt.exceptions import MeteorologyError
from stilt.meteorology import run_window
from stilt.spatial import Bounds
from stilt.transport.hysplit import Met, MetConfig

from ...fixtures.factories import write_arl_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_files(met, r_time, n_hours):
    """Return the met files of one run, as the HYSPLIT model asks for them."""
    return met.files_for(run_window(r_time, n_hours))


def _make_met(tmp_path: Path) -> Met:
    return Met("hrrr", MetConfig(directory=tmp_path))


def _hrrr_files(directory: Path, names: list[str]) -> list[Path]:
    """Write a file per name, such as ``20240601_06-11_hrrr``, holding the 6 hourly steps it names."""
    return [
        write_arl_file(
            directory / name,
            times=pd.date_range(
                dt.datetime.strptime(name[:11], "%Y%m%d_%H"), periods=6, freq="h"
            ),
        )
        for name in names
    ]


# ---------------------------------------------------------------------------
# MetConfig validation
# ---------------------------------------------------------------------------


def test_a_string_is_the_mets_folder(tmp_path):
    """``hrrr: /path/to/hrrr`` is the whole entry of a met of local files."""
    assert MetConfig.model_validate(str(tmp_path)).directory == tmp_path
    assert MetConfig.model_validate(tmp_path).directory == tmp_path


def test_a_met_with_only_a_folder_is_valid(tmp_path):
    cfg = MetConfig(directory=tmp_path)
    assert cfg.directory == tmp_path
    assert cfg.download is None


@pytest.mark.parametrize(
    "key",
    [{"file_format": "%Y%m%d_%H"}, {"file_tres": "6h"}],
    ids=lambda k: next(iter(k)),
)
def test_the_old_file_settings_are_refused_with_what_to_do(tmp_path, key):
    with pytest.raises(ValueError, match=f"{next(iter(key))} is gone: delete it"):
        MetConfig(directory=tmp_path, **key)


def test_metconfig_download_keeps_its_folder(tmp_path):
    cfg = MetConfig(directory=tmp_path, download="hrrr")
    assert cfg.download == "hrrr"
    assert cfg.directory == tmp_path


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
        MetConfig(directory=tmp_path, subgrid_enabel=True)


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
    """With download, files_for delegates to the arlmet archive's fetch()."""
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = [tmp_path / "file1", tmp_path / "file2"]
    for f in mock_archive.fetch.return_value:
        f.touch()

    met = _make_download_met(tmp_path)
    met._archive = mock_archive  # inject mock

    files = _run_files(met, "2024-07-18 12:00", -24)

    mock_archive.fetch.assert_called_once()
    call_kwargs = mock_archive.fetch.call_args
    assert call_kwargs.kwargs["dest_dir"] == tmp_path
    assert call_kwargs.kwargs["mirror"] == "s3"
    assert call_kwargs.kwargs["bbox"] is None
    assert call_kwargs.kwargs["levels"] is None
    assert len(files) == 2


def test_met_download_from_is_passed_to_fetch(tmp_path):
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    met = _make_download_met(tmp_path, download_from="ftp")
    met._archive = mock_archive
    _run_files(met, "2024-07-18 12:00", -24)

    assert mock_archive.fetch.call_args.kwargs["mirror"] == "ftp"


def test_met_download_with_subgrid_passes_bbox(tmp_path):
    """``download`` + ``subgrid_enable`` passes bbox to arlmet fetch."""
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
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
        ),
    )
    met._archive = mock_archive

    _run_files(met, "2024-07-18 12:00", -24)

    bbox = mock_archive.fetch.call_args.kwargs["bbox"]
    assert bbox == (-114.0, 39.0, -110.0, 42.0)
    assert mock_archive.fetch.call_args.kwargs["levels"] is None


def test_met_download_passes_subgrid_levels(tmp_path):
    """``download`` + ``subgrid_levels=N`` asks arlmet to keep the lowest N levels."""
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()

    met = _make_download_met(
        tmp_path,
        subgrid_enable=True,
        subgrid_bounds=Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0),
        subgrid_levels=3,
    )
    met._archive = mock_archive

    _run_files(met, "2024-07-18 12:00", -24)

    assert mock_archive.fetch.call_args.kwargs["levels"] == [0, 1, 2]


def test_met_download_with_no_files_raises(tmp_path):
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = []

    met = _make_download_met(tmp_path)
    met._archive = mock_archive

    with pytest.raises(MeteorologyError, match="No hrrr files"):
        _run_files(met, "2024-07-18 12:00", -24)


# ---------------------------------------------------------------------------
# Met archive subsetting via arlmet.extract_subset
# ---------------------------------------------------------------------------


BOUNDS = Bounds(xmin=-114.0, xmax=-110.0, ymin=39.0, ymax=42.0)


def _archive_met(tmp_path: Path, **kwargs) -> Met:
    """A local archive holding one 6 h file, cropped into tmp_path/crops."""
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    write_arl_file(
        archive / "20230101_12",
        times=pd.date_range("2023-01-01 12:00", periods=6, freq="h"),
    )
    settings = {
        "subgrid_bounds": BOUNDS,
        "subgrid_dir": tmp_path / "crops",
        **kwargs,
    }
    return Met(
        "hrrr",
        MetConfig(directory=archive, subgrid_enable=True, **settings),
    )


def _files(met: Met) -> list[Path]:
    """Return the files a forward 1 h run at the test time reads: the one 6 h file."""
    return met.readable(_run_files(met, dt.datetime(2023, 1, 1, 12), 1))


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
    """The same crop shares one folder, whatever the archive."""
    a = _archive_met(tmp_path)
    b = _archive_met(tmp_path / "other", subgrid_dir=tmp_path / "crops")
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


def test_cropping_local_files_needs_subgrid_dir(tmp_path):
    """Crops are never written into the met archive by default (#53)."""
    with pytest.raises(ValueError, match="subgrid_dir is required"):
        MetConfig(directory=tmp_path, subgrid_enable=True, subgrid_bounds=BOUNDS)


def test_a_met_needs_its_directory():
    with pytest.raises(ValueError, match="directory"):
        MetConfig()


def test_metconfig_download_crop_needs_no_subgrid_dir(tmp_path):
    cfg = MetConfig(
        directory=tmp_path, download="hrrr", subgrid_enable=True, subgrid_bounds=BOUNDS
    )
    assert cfg.subgrid_dir is None


# ---------------------------------------------------------------------------
# The files a run reads
# ---------------------------------------------------------------------------


def test_readable_returns_local_files_where_they_are(tmp_path):
    """Without cropping, HYSPLIT reads the met files in place: nothing is copied."""
    source_dir = tmp_path / "met"
    before, src = _hrrr_files(
        source_dir, ["20230101_06-11_hrrr", "20230101_12-17_hrrr"]
    )

    met = _make_met(source_dir)

    assert met.readable([src]) == [src]
    run = _run_files(met, dt.datetime(2023, 1, 1, 12), -1)
    assert met.readable(run) == [before, src]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["met"]


def test_readable_keeps_the_first_of_two_files_with_one_name(tmp_path, caplog):
    first = tmp_path / "met" / "a" / "20230101_12"
    second = tmp_path / "met" / "b" / "20230101_12"

    met = _make_met(tmp_path / "met")
    with caplog.at_level(logging.WARNING):
        files = met.readable([first, second])

    assert files == [first]
    assert "duplicate basename" in caplog.text
    assert str(first) in caplog.text
    assert str(second) in caplog.text


# ---------------------------------------------------------------------------
# Backward and forward runs
# ---------------------------------------------------------------------------


def test_files_for_backward_24h(tmp_path):
    """A 24-h backward run spans the previous day, and no file is repeated."""
    names = [
        f"{day}_{h:02d}-{h + 5:02d}_hrrr"
        for day in ("20221231", "20230101")
        for h in (0, 6, 12, 18)
    ]
    _hrrr_files(tmp_path, names)
    files = _run_files(_make_met(tmp_path), dt.datetime(2023, 1, 1, 12), -24)
    assert [f.name for f in files] == names[2:7]


def test_files_for_forward_run(tmp_path):
    """A forward run includes the files after the receptor time."""
    names = ["20230101_06-11_hrrr", "20230101_12-17_hrrr", "20230101_18-23_hrrr"]
    _hrrr_files(tmp_path, names)
    files = _run_files(_make_met(tmp_path), dt.datetime(2023, 1, 1, 12), 7)
    assert [f.name for f in files] == names[1:]


# ---------------------------------------------------------------------------
# Insufficient files
# ---------------------------------------------------------------------------


def test_files_for_raises_when_no_files(tmp_path):
    with pytest.raises(MeteorologyError, match="No met file") as info:
        _run_files(_make_met(tmp_path), dt.datetime(2023, 1, 1, 12), -1)
    assert str(tmp_path) in str(info.value)


def test_one_missing_file_fails_the_run_naming_its_hours(tmp_path):
    """Every file the run needs must be there, or HYSPLIT stops the particles where the met ends (#169)."""
    _hrrr_files(tmp_path, ["20230101_00-05_hrrr", "20230101_12-17_hrrr"])
    met = _make_met(tmp_path)
    with pytest.raises(
        MeteorologyError, match="for 2023-01-01 05:00 to 2023-01-01 12:00"
    ) as info:
        _run_files(met, dt.datetime(2023, 1, 1, 13), -12)
    assert info.value.reason == "MET_COVERAGE"


def test_missing_is_what_files_for_fails_for(tmp_path):
    """The same search as files_for, that says what it did not find rather than raising."""
    _hrrr_files(tmp_path, ["20230101_00-05_hrrr", "20230101_12-17_hrrr"])
    met = _make_met(tmp_path)
    window = run_window(dt.datetime(2023, 1, 1, 13), -12)

    assert met.missing(window) == [
        (pd.Timestamp("2023-01-01 05:00"), pd.Timestamp("2023-01-01 12:00"))
    ]
    with pytest.raises(MeteorologyError, match="2023-01-01 05:00 to 2023-01-01 12:00"):
        met.files_for(window)
    _hrrr_files(tmp_path, ["20230101_06-11_hrrr"])
    assert _make_met(tmp_path).missing(window) == []


def test_nothing_is_missing_for_a_download(tmp_path):
    met = Met("hrrr", MetConfig(directory=tmp_path, download="hrrr"))

    assert met.missing(run_window(dt.datetime(2023, 1, 1, 12), -3)) == []


def test_checking_receptors_lists_the_folder_once(tmp_path, monkeypatch):
    """Receptors an hour apart, some missing their files, cost one listing of the folder."""
    from stilt.transport.hysplit import catalog as catalog_module

    _hrrr_files(tmp_path, ["20230101_06-11_hrrr"])
    walks = []
    real_walk = catalog_module.walk
    monkeypatch.setattr(
        catalog_module, "walk", lambda top: walks.append(top) or real_walk(top)
    )
    receptors = pd.DataFrame(
        {
            "receptor": [f"r{h}" for h in range(6, 16)],
            "time": [pd.Timestamp(f"2023-01-01 {h}:00") for h in range(6, 16)],
            "longitude": -111.85,
            "latitude": 40.77,
        }
    )

    found = _make_met(tmp_path).check_receptors(receptors, -3)

    assert set(found["receptor"]) == {f"r{h}" for h in [6, 7, 8, 12, 13, 14, 15]}
    assert walks == [tmp_path]


# ---------------------------------------------------------------------------
# The crop box: longitudes count modulo 360 (#326)
# ---------------------------------------------------------------------------


def _outside_crop(tmp_path, west, east, longitudes, latitude=40.0):
    """Return the longitudes that ``met_crop`` flags for a box from *west* to *east* and latitudes 30 to 50."""
    config = MetConfig(
        directory=tmp_path,
        download="hrrr",
        subgrid_enable=True,
        subgrid_bounds=Bounds(xmin=west, xmax=east, ymin=30, ymax=50),
    )
    receptors = pd.DataFrame(
        {
            "receptor": [f"r{i}" for i in range(len(longitudes))],
            "time": pd.Timestamp("2023-01-01 12:00"),
            "longitude": longitudes,
            "latitude": latitude,
        }
    )
    found = Met("hrrr", config).check_receptors(receptors, -3)
    assert set(found["check"]) <= {"met_crop"}
    return [longitudes[int(r[1:])] for r in found["receptor"]]


@pytest.mark.parametrize(
    ("west", "east", "longitudes", "flagged"),
    [
        # The four rows of the issue's table.
        (-125, -100, [-111.9, -90, -130], [-90, -130]),
        (170, -170, [175, -175, 0], [0]),
        (170, 190, [175, -175, 185, 0], [0]),
        (-190, -170, [-175, 175, 185], []),
        # A receptor on an edge is inside, written either way.
        (-125, -100, [-125, -100], []),
        (170, -170, [170, -170, 190, -190], []),
        (170, 190, [170, 190, -170, -190], []),
        (-190, -170, [-190, -170, 170, 190], []),
        # Just past an edge is outside.
        (170, -170, [169.99, -169.99], [169.99, -169.99]),
        (-125, -100, [-125.01, -99.99], [-125.01, -99.99]),
        # A box a turn wide or more holds every longitude, whatever its edges.
        (-180, 180, [-180, 0, 180, 90, -179.9], []),
        (0, 360, [-180, 0, 180, 359, -1], []),
        (-200, 200, [-180, 0, 180, 195, -195], []),
        (-10, 400, [-180, 0, 180, 195, -195], []),
        # A box that reaches past 360 wraps.
        (350, 370, [355, -355, 5, -5, 0, 10, 11, 349, -11], [11, 349, -11]),
        # A box with no width holds its meridian only.
        (-111, -111, [-111, 249, -110.9], [-110.9]),
        # West of east by more than half a turn: the box is the long way round.
        (-170, 170, [175, -175, 0, 170, -170], [175, -175]),
    ],
)
def test_crop_box_longitudes_count_modulo_360(
    tmp_path, west, east, longitudes, flagged
):
    """A receptor is in the box when its longitude, east of the box's west edge, is within the box's width."""
    assert _outside_crop(tmp_path, west, east, longitudes) == flagged


def test_a_receptor_outside_the_crop_in_latitude_is_flagged_across_the_dateline(
    tmp_path,
):
    assert _outside_crop(tmp_path, 170, -170, [175, -175], latitude=60) == [175, -175]


def test_crop_edges_written_in_another_turn_are_not_flagged_by_rounding(tmp_path):
    """The west edge plus 360 is the west edge, and the east edge plus 360 the east edge, however the sums round."""
    assert _outside_crop(tmp_path, -175.3, -170.1, [184.7, -175.3, 189.9]) == []


# ---------------------------------------------------------------------------
# Which files are met files: anything with a header, anywhere under the folder
# ---------------------------------------------------------------------------


def test_files_for_finds_files_whatever_they_are_named(tmp_path):
    """A name says nothing: the header says what a file holds."""
    times = pd.date_range("2024-06-01 00:00", periods=6, freq="h")
    write_arl_file(tmp_path / "our_own_product.arl", times=times)

    files = _run_files(_make_met(tmp_path), dt.datetime(2024, 6, 1, 3), -2)

    assert [f.name for f in files] == ["our_own_product.arl"]


def test_files_for_searches_recursively(tmp_path):
    _hrrr_files(tmp_path / "2024" / "06", ["20240601_00-05_hrrr"])

    files = _run_files(_make_met(tmp_path), dt.datetime(2024, 6, 1, 0), 1)

    assert [f.name for f in files] == ["20240601_00-05_hrrr"]


def test_files_are_in_time_order_whatever_their_names_sort_as(tmp_path):
    """HRRR's two naming schemes meet at 2019-06-12: ``hysplit.*`` sorts after ``2019*``."""
    old = write_arl_file(
        tmp_path / "hysplit.20190611.18z.hrrra",
        times=pd.date_range("2019-06-11 18:00", periods=6, freq="h"),
    )
    new = write_arl_file(
        tmp_path / "20190612_00-05_hrrr",
        times=pd.date_range("2019-06-12 00:00", periods=6, freq="h"),
    )

    files = _run_files(_make_met(tmp_path), dt.datetime(2019, 6, 12, 2), -6)

    assert files == [old, new]


def test_a_folder_read_once_is_not_read_again(tmp_path, monkeypatch):
    """A new Met, as each run makes, reads the kept table and opens no file."""
    import arlmet

    _hrrr_files(tmp_path, ["20230101_00-05_hrrr"])
    window = run_window(dt.datetime(2023, 1, 1, 3), -2)
    _make_met(tmp_path).files_for(window)
    monkeypatch.setattr(arlmet, "probe", MagicMock(side_effect=AssertionError))

    assert [p.name for p in _make_met(tmp_path).files_for(window)] == [
        "20230101_00-05_hrrr"
    ]


def _linked_archive(root: Path) -> Path:
    """
    An archive laid out as the shared HRRR one: files in year/month folders, a link to each at the top.

    Beside them, a link to a missing file and a link to a folder.
    """
    nested = root / "2024" / "06"
    names = ["20240601_00-05_hrrr", "20240601_06-11_hrrr"]
    _hrrr_files(nested, names)
    for name in names:
        (root / name).symlink_to(Path("2024") / "06" / name)
    (root / "20240601_12-17_hrrr").symlink_to(Path("2024") / "06" / "gone")
    (root / "20240601_18-folder").symlink_to(Path("2024"), target_is_directory=True)
    return nested


def test_a_folder_of_links_finds_each_file_once(tmp_path):
    """A file and the link to it are one file, found where the link points."""
    nested = _linked_archive(tmp_path)
    met = _make_met(tmp_path)

    files = met.files_for(("2024-06-01 00:00", "2024-06-01 10:00"))

    assert files == [
        (nested / "20240601_00-05_hrrr").resolve(),
        (nested / "20240601_06-11_hrrr").resolve(),
    ]


def test_a_link_out_of_the_folder_finds_the_file_it_points_to(tmp_path):
    """As in the shared HRRR archive: names of one scheme link to another's files beside the folder."""
    (real,) = _hrrr_files(tmp_path / "v1", ["20150615_00-05_hrrr"])
    (tmp_path / "hrrr").mkdir()
    (tmp_path / "hrrr" / "link").symlink_to(real)

    files = _make_met(tmp_path / "hrrr").files_for(
        ("2015-06-15 01:00", "2015-06-15 02:00")
    )

    assert files == [real]


def test_a_link_to_a_missing_file_or_a_folder_is_no_met_file(tmp_path):
    _linked_archive(tmp_path)
    met = _make_met(tmp_path)

    with pytest.raises(MeteorologyError, match="No met file"):
        met.files_for(("2024-06-01 12:00", "2024-06-01 13:00"))


def test_the_walk_lists_links_without_following_them(tmp_path):
    """Links are listed as names, and a link to a folder is not walked into."""
    from stilt.transport.hysplit.catalog import walk

    _linked_archive(tmp_path)

    walked = {Path(folder): sorted(names) for folder, names in walk(tmp_path)}

    assert walked == {
        tmp_path: [
            "20240601_00-05_hrrr",
            "20240601_06-11_hrrr",
            "20240601_12-17_hrrr",
            "20240601_18-folder",
        ],
        tmp_path / "2024": [],
        tmp_path / "2024" / "06": ["20240601_00-05_hrrr", "20240601_06-11_hrrr"],
    }


def test_the_walk_skips_a_folder_it_cannot_read(tmp_path):
    """As os.walk does: a missing directory lists nothing, and raises nothing."""
    from stilt.transport.hysplit.catalog import walk

    assert list(walk(tmp_path / "missing")) == []


def _six_hourly_hrrr(tmp_path):
    _hrrr_files(
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
    return _make_met(tmp_path)


def test_files_for_backward_mid_file_release_skips_next_file(tmp_path):
    """A 19:06 release sits inside 18-23, so the next day's 00z file is not needed (#29)."""
    met = _six_hourly_hrrr(tmp_path)
    files = _run_files(met, dt.datetime(2019, 1, 23, 19, 6), -24)
    assert [f.name for f in files] == [
        "20190122_18-23_hrrr",
        "20190123_00-05_hrrr",
        "20190123_06-11_hrrr",
        "20190123_12-17_hrrr",
        "20190123_18-23_hrrr",
    ]


def test_files_for_backward_last_hour_release_includes_next_file(tmp_path):
    """A 23:06 release interpolates against 00z, which is in the next file."""
    met = _six_hourly_hrrr(tmp_path)
    files = _run_files(met, dt.datetime(2019, 1, 23, 23, 6), -24)
    assert files[-1].name == "20190124_00-05_hrrr"


def test_files_for_backward_boundary_release_adds_nothing_later(tmp_path):
    met = _six_hourly_hrrr(tmp_path)
    files = _run_files(met, dt.datetime(2019, 1, 23, 18), -24)
    assert files[-1].name == "20190123_18-23_hrrr"


@pytest.mark.parametrize(
    "ending", ["~20260403182134~", ".~1~", "~", ".lock", ".partial", ".bad"]
)
def test_files_for_ignores_a_copy_set_aside(tmp_path, ending):
    """A sibling with a name like these is not a met file, though its header would pass (#30)."""
    block = "20240702_00-05_hrrr"
    _hrrr_files(tmp_path, [block])
    times = pd.date_range("2024-07-02 00:00", periods=6, freq="h")
    write_arl_file(tmp_path / (block + ending), times=times, source="NAM")

    files = _run_files(_make_met(tmp_path), dt.datetime(2024, 7, 2, 3), -2)

    assert [f.name for f in files] == [block]


def test_files_for_ignores_a_copy_set_aside_in_a_subfolder(tmp_path):
    """The HRRR archive keeps blocks in year and month folders, with their `.bad` copies."""
    block = "20260220_00-05_hrrr"
    _hrrr_files(tmp_path / "2026" / "02", [block])
    write_arl_file(
        tmp_path / "2026" / "02" / (block + ".bad"),
        times=pd.date_range("2026-02-20 00:00", periods=6, freq="h"),
        source="NAM",
    )

    files = _run_files(_make_met(tmp_path), dt.datetime(2026, 2, 20, 3), -2)

    assert [f.name for f in files] == [block]


def test_a_crop_being_written_is_not_a_met_file(tmp_path):
    """A crop mid-write can read as whole time steps, so its name has to keep it out."""
    write_arl_file(
        tmp_path / ".20240702_00-05_hrrr.tmp.abc.partial",
        times=pd.date_range("2024-07-02 00:00", periods=3, freq="h"),
    )

    with pytest.raises(MeteorologyError, match="No met file"):
        _run_files(_make_met(tmp_path), dt.datetime(2024, 7, 2, 1), -1)


def test_a_block_with_only_a_copy_set_aside_is_missing(tmp_path):
    """The copy does not stand in for the block: the run, and `stilt check`, find it missing."""
    write_arl_file(
        tmp_path / "20240702_00-05_hrrr.partial",
        times=pd.date_range("2024-07-02 00:00", periods=6, freq="h"),
    )
    met = _make_met(tmp_path)
    window = run_window(dt.datetime(2024, 7, 2, 3), -2)

    with pytest.raises(MeteorologyError, match="No met file") as info:
        met.files_for(window)
    assert info.value.reason == "MET_COVERAGE"
    assert met.missing(window) == [(window[0], window[1])]


def test_the_source_is_not_read_from_a_copy_set_aside(tmp_path):
    """The met's product comes from a file `files_for` can return, not from its `.bad` copy."""
    write_arl_file(tmp_path / "20240702_00-05_hrrr.bad", source="NAM")
    write_arl_file(tmp_path / "20240702_06-11_hrrr", source="HRRR")

    assert _make_met(tmp_path).config.source() == "HRRR"


def test_downloaded_met_gets_the_next_file_in_the_last_hour_of_a_file(tmp_path):
    """
    A receptor in the last hour of a met file also needs the next file, as for local met.

    At 17:30 with 6-hour HRRR files (12-17, 18-23), HYSPLIT interpolates
    between 17:00 and 18:00, so the 18:00 file is needed. STILT-R applies
    this rule, and so does the local path (see the test below); the
    download asks the archive for files through 18:00.
    """
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()
    met = _make_download_met(tmp_path)
    met._archive = mock_archive

    _run_files(met, "2024-07-18 17:30", -24)

    start, end = mock_archive.fetch.call_args.args[:2]
    assert start == pd.Timestamp("2024-07-17 17:30")
    assert end == pd.Timestamp("2024-07-18 18:00")


def test_local_met_gets_the_next_file_in_the_last_hour_of_a_file(tmp_path):
    _hrrr_files(tmp_path, ["20240718_12-17_hrrr", "20240718_18-23_hrrr"])
    met = _make_met(tmp_path)

    in_last_hour = _run_files(met, "2024-07-18 17:30", -1)
    earlier = _run_files(met, "2024-07-18 16:30", -1)

    assert [p.name for p in in_last_hour] == [
        "20240718_12-17_hrrr",
        "20240718_18-23_hrrr",
    ]
    assert [p.name for p in earlier] == ["20240718_12-17_hrrr"]


def _window_end_files(met, end):
    """Return the names of the files a forward run ending at *end*, and a backward run released then, read."""
    forward = _run_files(met, pd.Timestamp(end) - pd.Timedelta(hours=3), 3)
    backward = _run_files(met, end, -3)
    assert [p.name for p in forward][-1] == [p.name for p in backward][-1]
    return [p.name for p in forward]


@pytest.mark.parametrize(
    ("end", "files"),
    [
        # the issue's run: 20:31 for 3 hours ends between 23:00 and the next file's 00:00
        ("2021-01-14 23:31", ["20210114_18-23_hrrr", "20210115_00-05_hrrr"]),
        ("2021-01-14 23:59", ["20210114_18-23_hrrr", "20210115_00-05_hrrr"]),
        # on the file's last hour: HYSPLIT reads nothing past it (#283)
        ("2021-01-14 23:00", ["20210114_18-23_hrrr"]),
        ("2021-01-14 22:59", ["20210114_18-23_hrrr"]),
        ("2021-01-15 00:00", ["20210114_18-23_hrrr", "20210115_00-05_hrrr"]),
        ("2021-01-15 00:01", ["20210114_18-23_hrrr", "20210115_00-05_hrrr"]),
    ],
)
def test_a_forward_run_reads_the_hour_after_its_end_in_six_hour_files(
    tmp_path, end, files
):
    """A forward run needs the met time after its end, as a backward run does after its release (#275)."""
    _hrrr_files(
        tmp_path,
        ["20210114_12-17_hrrr", "20210114_18-23_hrrr", "20210115_00-05_hrrr"],
    )
    met = _make_met(tmp_path)

    assert _window_end_files(met, end)[-len(files) :] == files


@pytest.mark.parametrize(
    ("end", "files"),
    [
        ("2015-01-01 21:00", ["20150101_nam12"]),
        ("2015-01-01 22:00", ["20150101_nam12", "20150102_nam12"]),
        ("2015-01-01 23:30", ["20150101_nam12", "20150102_nam12"]),
        ("2015-01-02 00:00", ["20150101_nam12", "20150102_nam12"]),
    ],
)
def test_a_forward_run_reads_the_hour_after_its_end_in_daily_files(
    tmp_path, end, files
):
    for name, day in (
        ("20150101_nam12", "2015-01-01"),
        ("20150102_nam12", "2015-01-02"),
    ):
        write_arl_file(tmp_path / name, times=pd.date_range(day, periods=8, freq="3h"))
    met = _make_met(tmp_path)

    assert _window_end_files(met, end) == files


# ---------------------------------------------------------------------------
# Met steps longer than an hour (#283)
#
# HYSPLIT needs the met step at or before a run's start and the one at or
# after its end. For daily files of 3-hourly steps (NAM12, GDAS 0.5 deg), the
# step after a 22:00 end is the next day's 00:00, and a 21:00 end needs
# nothing past 21:00.
# ---------------------------------------------------------------------------


def _stepped_files(
    tmp_path: Path, names: list[str], file_format: str, tres: str, step: str
) -> list[Path]:
    """Write one ARL file per name in *names*, each holding the steps of *step* in its *tres*."""
    files = []
    for name in names:
        first = pd.Timestamp(dt.datetime.strptime(name, file_format))
        end = first + pd.to_timedelta(tres)
        times = pd.date_range(first, end, freq=step, inclusive="left")
        files.append(write_arl_file(tmp_path / name, times=times))
    return files


#: Daily files of 3-hourly steps, as NAM12's: 00:00 to 21:00 each day.
_NAM = {"names": ["20150101", "20150102"], "file_format": "%Y%m%d", "tres": "1D"}


@pytest.mark.parametrize(
    ("step", "end", "files"),
    [
        # the issue's two backward runs: after the last step, and on it
        ("3h", "2015-01-01 22:00", ["20150101", "20150102"]),
        ("3h", "2015-01-01 21:00", ["20150101"]),
        ("3h", "2015-01-01 21:01", ["20150101", "20150102"]),
        ("3h", "2015-01-01 20:59", ["20150101"]),
        ("3h", "2015-01-02 00:00", ["20150101", "20150102"]),
        # daily files of 6-hourly steps, as GFS 0.25 deg: 00:00 to 18:00
        ("6h", "2015-01-01 18:00", ["20150101"]),
        ("6h", "2015-01-01 19:00", ["20150101", "20150102"]),
    ],
)
@pytest.mark.parametrize("n_hours", [-3, 3])
def test_files_for_reads_the_next_file_only_past_the_last_met_step(
    tmp_path, step, end, files, n_hours
):
    """A backward run released at *end* and a forward run ending then get the same files."""
    _stepped_files(tmp_path, _NAM["names"], _NAM["file_format"], _NAM["tres"], step)
    met = _make_met(tmp_path)
    r_time = (
        pd.Timestamp(end) if n_hours < 0 else pd.Timestamp(end) - pd.Timedelta(hours=3)
    )
    window = run_window(r_time, n_hours)

    found = met.files_for(window)

    assert [p.name for p in found] == files
    met.check(found, window)


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_check_passes_a_run_that_ends_on_the_last_met_step(tmp_path, n_hours):
    """A run released at 21:00 (or ending then) runs on that day's file alone, as HYSPLIT runs it (#283)."""
    files = _stepped_files(
        tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h"
    )
    r_time = dt.datetime(2015, 1, 1, 21) if n_hours < 0 else dt.datetime(2015, 1, 1, 18)

    _make_met(tmp_path).check(files, run_window(r_time, n_hours))


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_check_fails_a_run_past_the_last_met_step_without_the_next_file(
    tmp_path, n_hours
):
    """A run released at 22:00 needs the next day's 00:00 step."""
    files = _stepped_files(
        tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h"
    )
    r_time = dt.datetime(2015, 1, 1, 22) if n_hours < 0 else dt.datetime(2015, 1, 1, 19)
    met = _make_met(tmp_path)

    with pytest.raises(
        MeteorologyError,
        match="hold 2015-01-01 00:00 to 2015-01-01 21:00. The run needs "
        "2015-01-01 19:00 to 2015-01-01 22:00",
    ):
        met.check(files, run_window(r_time, n_hours))


@pytest.mark.parametrize("n_hours", [-3, 3])
def test_missing_lists_the_step_a_run_past_the_last_step_needs(tmp_path, n_hours):
    """`stilt check` asks for the next day's file a 22:00 end needs, and none for 21:00."""
    _stepped_files(tmp_path, ["20150101"], _NAM["file_format"], _NAM["tres"], "3h")
    met = _make_met(tmp_path)

    def missing(end):
        r_time = pd.Timestamp(end)
        if n_hours > 0:
            r_time -= pd.Timedelta(hours=n_hours)
        return met.missing(run_window(r_time, n_hours))

    assert missing("2015-01-01 22:00") == [
        (pd.Timestamp("2015-01-01 21:00"), pd.Timestamp("2015-01-01 22:00"))
    ]
    assert missing("2015-01-01 21:00") == []


#: 6-hour files of hourly steps, as NOAA's HRRR archive.
_HRRR = {
    "names": ["20210115_00", "20210115_06"],
    "file_format": "%Y%m%d_%H",
    "tres": "6h",
}


@pytest.mark.parametrize(
    ("end", "files"),
    [
        # on the file's last hour: HYSPLIT reads nothing past 05:00
        ("2021-01-15 05:00", ["20210115_00"]),
        ("2021-01-15 05:30", ["20210115_00", "20210115_06"]),
        ("2021-01-15 04:59", ["20210115_00"]),
        ("2021-01-15 06:00", ["20210115_00", "20210115_06"]),
    ],
)
@pytest.mark.parametrize("n_hours", [-3, 3])
def test_files_for_hourly_steps_drop_the_file_after_the_last_hour(
    tmp_path, end, files, n_hours
):
    """With hourly steps only a run that ends exactly on a file's last hour changes: it no longer needs the next file."""
    _stepped_files(tmp_path, _HRRR["names"], _HRRR["file_format"], _HRRR["tres"], "h")
    met = _make_met(tmp_path)
    r_time = (
        pd.Timestamp(end) if n_hours < 0 else pd.Timestamp(end) - pd.Timedelta(hours=3)
    )
    window = run_window(r_time, n_hours)

    found = met.files_for(window)

    assert [p.name for p in found] == files
    met.check(found, window)


def test_a_file_that_is_not_arl_beside_the_met_changes_nothing(tmp_path):
    (tmp_path / "aaa_20210115").write_bytes(b"not an ARL index record" * 50)
    _hrrr_files(tmp_path, ["20210115_00-05_hrrr"])
    met = _make_met(tmp_path)
    window = run_window(dt.datetime(2021, 1, 15, 3, 30), -3)

    files = met.files_for(window)

    assert [p.name for p in files] == ["20210115_00-05_hrrr"]
    met.check(files, window)


def test_two_products_holding_one_time_are_refused_naming_both(tmp_path):
    """A run records one source, and which of the two it means cannot be told."""
    hrrr = write_arl_file(
        tmp_path / "hrrr_20210115",
        times=pd.date_range("2021-01-15 00:00", periods=6, freq="h"),
    )
    nam = write_arl_file(
        tmp_path / "nam_20210115",
        times=pd.date_range("2021-01-15 00:00", periods=8, freq="3h"),
        source="NAM",
    )
    with pytest.raises(MeteorologyError, match="its own folder") as info:
        _make_met(tmp_path).files_for(run_window(dt.datetime(2021, 1, 15, 3, 30), -3))
    assert str(hrrr) in str(info.value)
    assert str(nam) in str(info.value)


def test_two_products_at_different_times_share_a_folder(tmp_path):
    """gdas0p5's source changes over its life, so a change of product alone is no clash."""
    write_arl_file(
        tmp_path / "old",
        times=pd.date_range("2021-01-15 00:00", periods=6, freq="h"),
        source="GHDA",
    )
    write_arl_file(
        tmp_path / "new",
        times=pd.date_range("2021-01-15 06:00", periods=6, freq="h"),
        source="GFSG",
    )

    files = _make_met(tmp_path).files_for(("2021-01-15 04:00", "2021-01-15 07:00"))

    assert [p.name for p in files] == ["old", "new"]


def test_a_short_copy_of_one_product_does_not_hide_the_whole_file(tmp_path):
    """The file that holds the most is used when one holds only part of what another does."""
    (whole,) = _hrrr_files(tmp_path, ["20210115_00-05_hrrr"])
    write_arl_file(
        tmp_path / "short",
        times=pd.date_range("2021-01-15 00:00", periods=2, freq="h"),
    )

    assert _make_met(tmp_path).files_for(("2021-01-15 00:00", "2021-01-15 01:00")) == [
        whole
    ]


def test_a_3_hourly_met_beside_an_hourly_file_that_sorts_first(tmp_path):
    """The NAM12 run from 22:00 gets the next day's file though an hourly file sorts first (#283)."""
    write_arl_file(
        tmp_path / "20140101_00-05_hrrr",
        times=pd.date_range("2014-01-01 00:00", periods=6, freq="h"),
    )
    names = ["20150101_nam12", "20150102_nam12"]
    for name, day in zip(names, ["2015-01-01", "2015-01-02"], strict=True):
        write_arl_file(tmp_path / name, times=pd.date_range(day, periods=8, freq="3h"))
    met = _make_met(tmp_path)
    window = run_window(dt.datetime(2015, 1, 1, 22), -3)

    files = met.files_for(window)

    assert [p.name for p in files] == names
    met.check(files, window)


def test_the_met_step_skips_a_partial_download(tmp_path):
    """A cut-short download beside a met file, with other steps, is not read for the step."""
    _stepped_files(tmp_path, _NAM["names"], _NAM["file_format"], _NAM["tres"], "3h")
    write_arl_file(
        tmp_path / "20150101.partial",
        times=pd.date_range("2015-01-01", periods=24, freq="h"),
    )
    met = _make_met(tmp_path)

    files = met.files_for(run_window(dt.datetime(2015, 1, 1, 22), -3))

    assert [p.name for p in files] == _NAM["names"]


def test_a_file_that_is_not_arl_is_no_file_for_its_time(tmp_path):
    """Its header cannot say what it holds, so the run finds nothing for that time."""
    (tmp_path / "20150101").write_bytes(b"not an ARL index record" * 50)
    window = run_window(dt.datetime(2015, 1, 1, 22), -3)

    assert _make_met(tmp_path).missing(window) == [window]


def test_a_forward_run_missing_the_file_after_its_end_names_that_hour(tmp_path):
    _hrrr_files(tmp_path, ["20210114_18-23_hrrr"])
    met = _make_met(tmp_path)
    window = run_window(dt.datetime(2021, 1, 14, 20, 31), 3)

    with pytest.raises(
        MeteorologyError, match="for 2021-01-14 23:00 to 2021-01-14 23:31"
    ):
        met.files_for(window)
    assert met.missing(window) == [
        (pd.Timestamp("2021-01-14 23:00"), pd.Timestamp("2021-01-14 23:31"))
    ]


def test_a_forward_download_asks_for_the_met_step_after_its_end(tmp_path):
    mock_archive = MagicMock(step=pd.Timedelta(hours=1))
    mock_archive.fetch.return_value = [tmp_path / "file1"]
    (tmp_path / "file1").touch()
    met = _make_download_met(tmp_path)
    met._archive = mock_archive

    _run_files(met, "2021-01-14 20:31", 3)

    start, end = mock_archive.fetch.call_args.args[:2]
    assert start == pd.Timestamp("2021-01-14 20:31")
    assert end == pd.Timestamp("2021-01-15 00:00")


def _archive_files(tmp_path, download, release, hours):
    """The files of the real *download* archive that a run's download asks for (no network)."""
    from arlmet.archives import get_archive

    archive = get_archive(download)
    asked: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(start, end, **kwargs):
        asked.append((start, end))
        return [
            tmp_path / path.split("/")[-1]
            for path in archive.paths_for_range(start, end)
        ]

    archive.fetch = fetch  # type: ignore[method-assign]
    met = _make_download_met(tmp_path, download)
    met._archive = archive
    return [path.name for path in _run_files(met, release, hours)]


@pytest.mark.parametrize(
    ("download", "day_file"),
    [("nam12", "{}_nam12"), ("gdas0p5", "{}_gdas0p5")],
)
@pytest.mark.parametrize(
    ("release", "next_day"),
    [
        ("2015-01-01 22:00", True),  # needs the 00:00 step of January 2
        ("2015-01-01 21:30", True),  # between the last step and the next
        ("2015-01-01 21:00", False),  # exactly the last step of the file
        ("2015-01-01 20:00", False),
    ],
)
def test_a_3_hourly_download_gets_the_next_days_file_past_the_last_step(
    tmp_path, download, day_file, release, next_day
):
    """A backward run released after 21:00 needs the next 00:00 step, in the next day's file (#306)."""
    files = _archive_files(tmp_path, download, release, -3)

    expected = [day_file.format("20150101")]
    if next_day:
        expected.append(day_file.format("20150102"))
    assert files == expected


@pytest.mark.parametrize(
    ("download", "release", "n_files", "last"),
    [
        # 6-hour files, hourly steps: the next file only past the last hour of a file
        ("hrrr", "2024-07-18 17:30", 2, "20240718_18-23_hrrr"),
        ("hrrr", "2024-07-18 15:00", 1, "20240718_12-17_hrrr"),
        ("hrrr", "2024-07-18 05:30", 2, "20240718_06-11_hrrr"),
        ("hrrr", "2024-07-18 05:00", 1, "20240718_00-05_hrrr"),  # the last step
        ("hrrr", "2024-07-18 06:00", 2, "20240718_06-11_hrrr"),
        # daily files, hourly steps
        ("nams", "2015-01-01 23:30", 2, "20150102_hysplit.t00z.namsa"),
        ("nams", "2015-01-01 23:00", 1, "20150101_hysplit.t00z.namsa"),
    ],
)
def test_an_hourly_download_gets_the_next_file_only_past_the_last_step_of_a_file(
    tmp_path, download, release, n_files, last
):
    """A run that ends on the last step of a file does not fetch the next file, as for local met."""
    files = _archive_files(tmp_path, download, release, -1)

    assert len(files) == n_files
    assert files[-1] == last


def test_met_config_construction(tmp_path):
    mc = MetConfig(directory=tmp_path / "met")
    assert mc.directory == tmp_path / "met"


# ---------------------------------------------------------------------------
# What a run records of the met
# ---------------------------------------------------------------------------


def _local(directory: Path, **fields) -> MetConfig:
    return MetConfig(directory=directory, **fields)


def test_settings_are_the_source_in_the_file_header_and_the_crop(tmp_path):
    write_arl_file(tmp_path / "20240718_00", source="NAM")

    assert _local(tmp_path).settings() == {"source": "NAM", "crop": None}


def test_a_downloaded_met_records_its_archive_source_without_reading_a_file(tmp_path):
    met = MetConfig(directory=tmp_path / "nowhere", download="gfs0p25")

    assert met.settings() == {"source": "GFSQ", "crop": None}


def test_the_source_is_read_past_files_that_are_not_arl(tmp_path):
    (tmp_path / "README").write_text("HRRR files from NOAA")
    write_arl_file(tmp_path / "2024" / "20240718_00", source="HRRR")

    assert _local(tmp_path).source() == "HRRR"


def test_a_met_with_no_arl_file_cannot_say_its_source(tmp_path):
    (tmp_path / "20240718_00").touch()

    with pytest.raises(FileNotFoundError, match="No ARL file"):
        _local(tmp_path).source()


def test_a_met_whose_directory_does_not_exist_says_so(tmp_path):
    config = MetConfig(directory=tmp_path / "gone")

    with pytest.raises(FileNotFoundError) as info:
        config.source()
    assert str(info.value) == f"{tmp_path / 'gone'} does not exist."


def test_where_the_files_are_and_how_they_are_named_are_not_recorded(tmp_path):
    write_arl_file(tmp_path / "a" / "20240718_00-05_hrrr")
    write_arl_file(tmp_path / "b" / "hrrr.20240718.t00z")

    assert (
        MetConfig(directory=tmp_path / "a").settings()
        == MetConfig(directory=tmp_path / "b").settings()
    )


def test_two_products_named_alike_are_told_apart(tmp_path):
    """Two archives with the same file names record different mets (R2-24)."""
    write_arl_file(tmp_path / "hrrr" / "20240718_00", source="HRRR")
    write_arl_file(tmp_path / "nam" / "20240718_00", source="NAM")

    assert _local(tmp_path / "hrrr").settings() != _local(tmp_path / "nam").settings()


def test_the_crop_is_the_box(tmp_path):
    met = _local(
        tmp_path,
        subgrid_enable=True,
        subgrid_bounds=BOUNDS,
        subgrid_dir=tmp_path / "crops",
    )

    assert met.crop() == {
        "bbox": [BOUNDS.xmin, BOUNDS.ymin, BOUNDS.xmax, BOUNDS.ymax],
        "levels": None,
    }


def test_n_min_is_refused_with_why(tmp_path):
    with pytest.raises(ValueError, match="n_min is gone: a run needs every met file"):
        _local(tmp_path, n_min=5)


def test_a_buffer_is_refused_with_how_to_widen_the_box(tmp_path):
    with pytest.raises(ValueError, match="Widen the bounds"):
        _local(tmp_path, subgrid_buffer=0.2)


def test_a_crop_turned_off_is_no_crop(tmp_path):
    assert _local(tmp_path, subgrid_bounds=BOUNDS).crop() is None


def test_a_relative_directory_with_a_variable_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("MET_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    assert Met("hrrr", _local(Path("$MET_ROOT/hrrr"))).directory == tmp_path / "hrrr"
    assert Met("hrrr", _local(Path("hrrr"))).directory == tmp_path / "hrrr"


# ---------------------------------------------------------------------------
# Checking the files a run reads
# ---------------------------------------------------------------------------


def _six_hour_files(tmp_path: Path, held: dict[str, str], **kwargs) -> list[Path]:
    """Write a 6 h file per name in *held*, holding the hours from its name's hour to the given last hour."""
    files = []
    for name, last in held.items():
        first = pd.Timestamp(dt.datetime.strptime(name, "%Y%m%d_%H"))
        hours = pd.date_range(first, pd.Timestamp(last), freq="h")
        files.append(write_arl_file(tmp_path / name, times=hours, **kwargs))
    return files


#: 8 h back from 10:00, so the run needs the met steps from 02:00 to 10:00.
_WINDOW = run_window(dt.datetime(2023, 1, 1, 10), -8)


def test_check_passes_files_that_hold_every_hour_the_run_needs(tmp_path):
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 11:00"}
    )
    _make_met(tmp_path).check(files, _WINDOW)


def test_check_passes_files_of_two_time_periods(tmp_path):
    files = [
        write_arl_file(
            tmp_path / f"20230101_{hour:02d}",
            times=pd.date_range(f"2023-01-01 {hour:02d}:00", periods=2, freq="h"),
        )
        for hour in range(2, 11, 2)
    ]
    _make_met(tmp_path).check(files, _WINDOW)


def test_check_refuses_files_that_hold_one_time_period(tmp_path):
    """HYSPLIT stops on a file with one time period, so no run can use them (#286)."""
    files = [
        write_arl_file(
            tmp_path / f"20230101_{hour:02d}", times=[f"2023-01-01 {hour:02d}:00"]
        )
        for hour in range(2, 12)
    ]
    met = _make_met(tmp_path)
    with pytest.raises(
        MeteorologyError,
        match=f"met file {files[0]} holds one time period.*Only one time period of "
        "meteo data",
    ) as info:
        met.check(files, _WINDOW)
    assert info.value.reason == "MET_COVERAGE"


def test_check_refuses_one_file_with_one_time_period_among_longer_ones(tmp_path):
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 11:00"}
    )
    short = write_arl_file(tmp_path / "20230101_12", times=["2023-01-01 12:00"])
    met = _make_met(tmp_path)
    with pytest.raises(MeteorologyError, match=f"met file {short} holds one time"):
        met.check([*files, short], _WINDOW)


def test_a_file_cut_to_one_hour_is_reported_as_the_gap_it_leaves(tmp_path):
    """A file that is short because it was cut is a gap in the hours, which is the more useful thing to say."""
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 00:00", "20230101_06": "2023-01-01 11:00"}
    )
    met = _make_met(tmp_path)
    with pytest.raises(MeteorologyError, match="no time step between 2023-01-01 00:00"):
        met.check(files, _WINDOW)


def test_check_fails_a_file_cut_after_a_whole_hour(tmp_path):
    """A file that ends early looks whole to arlmet; the gap after it does not (#189)."""
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 03:00", "20230101_06": "2023-01-01 11:00"}
    )
    met = _make_met(tmp_path)
    with pytest.raises(
        MeteorologyError,
        match="no time step between 2023-01-01 03:00 and 2023-01-01 06:00",
    ) as info:
        met.check(files, _WINDOW)
    assert info.value.reason == "MET_COVERAGE"


def test_check_fails_files_that_end_before_the_run_does(tmp_path):
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 09:00"}
    )
    met = _make_met(tmp_path)
    with pytest.raises(
        MeteorologyError,
        match="hold 2023-01-01 00:00 to 2023-01-01 09:00. The run needs "
        "2023-01-01 02:00 to 2023-01-01 10:00",
    ):
        met.check(files, _WINDOW)


def test_check_needs_the_step_after_a_forward_run_ends(tmp_path):
    """A forward run from 20:31 for 3 hours ends at 23:31, between two files (#275)."""
    window = run_window(dt.datetime(2021, 1, 14, 20, 31), 3)
    held = {"20210114_18": "2021-01-14 23:00", "20210115_00": "2021-01-15 05:00"}
    files = _six_hour_files(tmp_path, held)
    met = _make_met(tmp_path)

    met.check(files, window)
    with pytest.raises(
        MeteorologyError,
        match="hold 2021-01-14 18:00 to 2021-01-14 23:00. The run needs "
        "2021-01-14 20:31 to 2021-01-14 23:31",
    ):
        met.check(files[:1], window)


def test_check_names_a_damaged_file(tmp_path):
    """A time step written partway opens cleanly; arlmet's check finds it."""
    files = _six_hour_files(
        tmp_path,
        {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 11:00"},
    )
    damaged = write_arl_file(
        tmp_path / "20230101_06",
        times=pd.date_range("2023-01-01 06:00", "2023-01-01 11:00", freq="h"),
        partial=True,
    )
    met = _make_met(tmp_path)
    with pytest.raises(
        MeteorologyError,
        match=f"met file {damaged} is damaged: 2023-01-01 11:00 has 1 of the 2 data "
        "records",
    ):
        met.check(files, _WINDOW)


def test_check_names_a_file_that_does_not_open(tmp_path):
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 11:00"}
    )
    files[1].write_bytes(files[1].read_bytes()[:-10])  # cut inside a record
    met = _make_met(tmp_path)
    with pytest.raises(MeteorologyError, match="is damaged: .*not a whole number"):
        met.check(files, _WINDOW)


def test_check_reads_each_file_once_until_it_changes(tmp_path, monkeypatch):
    """A check reads every record's header, so a process checks a file once, however many runs read it."""
    import arlmet

    opened: list[str] = []
    real_file = arlmet.File

    def counting_file(path, *args, **kwargs):
        if kwargs.get("mode", "r") == "r":  # not write_arl_file making its bytes
            opened.append(Path(path).name)
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(arlmet, "File", counting_file)
    files = _six_hour_files(
        tmp_path, {"20230101_00": "2023-01-01 05:00", "20230101_06": "2023-01-01 11:00"}
    )
    met = _make_met(tmp_path)
    met.check(files, _WINDOW)
    met.check(files, _WINDOW)
    assert opened == ["20230101_00", "20230101_06"]

    # Rewritten with fewer hours, the file is read again.
    _six_hour_files(tmp_path, {"20230101_06": "2023-01-01 09:00"})
    with pytest.raises(MeteorologyError, match="to 2023-01-01 09:00"):
        met.check(files, _WINDOW)
    assert opened == ["20230101_00", "20230101_06", "20230101_06"]
