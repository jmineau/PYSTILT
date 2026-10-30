"""Tests for Project.fetch and stilt fetch: the met files the unfinished simulations need, got once."""

import datetime as dt
from pathlib import Path

import arlmet.archives
import pandas as pd
import pytest
from arlmet.archives import HRRRArchive
from typer.testing import CliRunner

from stilt import transport
from stilt.cli import app
from stilt.project import Project
from stilt.transport.hysplit import Met, MetConfig

from .fixtures.factories import (
    make_met_config,
    make_met_files,
    make_receptor,
)

runner = CliRunner()

BOUNDS = {"xmin": -113.0, "xmax": -110.0, "ymin": 39.0, "ymax": 42.0}


class FakeHRRR(HRRRArchive):
    """
    HRRR's file names and 6 h blocks, with no network.

    ``_download`` and ``_fetch_and_crop`` write a few bytes and note the
    file, so a test can say what was downloaded, in how many requests, and
    that a second ask downloads nothing. A file whose URL holds a string in
    ``missing`` is not on the server.
    """

    def __init__(self) -> None:
        super().__init__()
        self.downloaded: list[str] = []
        self.requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.missing: set[str] = set()

    def fetch(self, start, end, **kwargs):
        self.requests.append((pd.Timestamp(start), pd.Timestamp(end)))
        return super().fetch(start, end, **kwargs)

    def _download(self, url, dest, opts, **kwargs):
        if any(text in url for text in self.missing):
            raise FileNotFoundError(f"{url} is not in the archive")
        self.downloaded.append(dest.name)
        dest.write_bytes(b"arl")

    def _fetch_and_crop(self, url, dest, opts, *, bbox, levels):
        self._download(url, dest, opts)


@pytest.fixture
def archive(monkeypatch) -> FakeHRRR:
    """The archive every met that downloads ``hrrr`` gets."""
    fake = FakeHRRR()
    monkeypatch.setattr(arlmet.archives, "get_archive", lambda name, **options: fake)
    return fake


def _receptor(day: int, hour: int):
    return make_receptor(dt.datetime(2024, 7, day, hour), -111.85, 40.77)


def _project(tmp_path: Path, receptors, *, n_hours: int = -3, **met) -> Project:
    """A project that downloads HRRR into ``tmp_path / "met"``, cropped, and makes particles only."""
    fields = {
        "directory": tmp_path / "met",
        "download": "hrrr",
        "subgrid_enable": True,
        "subgrid_bounds": BOUNDS,
        "subgrid_levels": 4,
        **met,
    }
    return Project.init(
        tmp_path / "proj",
        receptors=receptors,
        mets={"hrrr": fields},
        n_hours=n_hours,
        variants={"hrrr": {}},
    )


class _Stop(Exception):
    """Raised where HYSPLIT would start."""


def _files_a_run_gets(project: Project, receptor_ids, monkeypatch, tmp_path):
    """
    Return the met files that runs of *receptor_ids* ask for.

    Each is run through the transport model (:func:`stilt.transport.run_model`) up to
    the point where HYSPLIT would start, and the files :meth:`Met.files_for`
    returns are noted.
    """
    got: list[Path] = []
    files_for = Met.files_for

    def note(self, window):
        files = files_for(self, window)
        got.extend(files)
        return files

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(Met, "files_for", note)
    monkeypatch.setattr(Met, "check", lambda self, files, window: None)
    monkeypatch.setattr("stilt.transport.hysplit.model.write_inputs", stop)
    for rid in receptor_ids:
        sim = project.simulation(rid, "hrrr")
        with pytest.raises(_Stop):
            transport.run_model(
                sim.variant.model.name,
                sim.receptor,
                sim.transport,
                sim.variant.met_config,
                tmp_path / "workdir",
            )
    monkeypatch.undo()
    return got


# ---------------------------------------------------------------------------
# The files a run would ask for
# ---------------------------------------------------------------------------

#: Hours that reach across a block, fill one block, and are a week apart.
RECEPTORS = [
    _receptor(2, 3),
    _receptor(2, 7),
    _receptor(2, 8),
    _receptor(2, 20),
    _receptor(9, 1),
]


def test_fetch_gets_exactly_the_files_the_runs_ask_for(tmp_path, archive, monkeypatch):
    fetched = _project(tmp_path / "fetch", RECEPTORS, n_hours=-6)
    files = fetched.fetch()
    other = _project(tmp_path / "run", RECEPTORS, n_hours=-6)
    ran = _files_a_run_gets(
        other, other.receptors["receptor"], monkeypatch, tmp_path / "wd"
    )

    assert files["message"].isna().all()
    assert len(files) == 7
    # Same names, in two met directories: the crop is in the name.
    assert sorted(Path(f).name for f in files["file"]) == sorted({p.name for p in ran})
    assert all(Path(f).is_file() for f in files["file"])
    assert set(files["met"]) == {"hrrr"}


def test_runs_after_a_fetch_download_nothing(tmp_path, archive, monkeypatch):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)
    project.fetch()
    downloaded = list(archive.downloaded)

    ran = _files_a_run_gets(
        project, project.receptors["receptor"], monkeypatch, tmp_path / "wd"
    )

    assert len(downloaded) == 7
    assert archive.downloaded == downloaded
    assert len(ran) > 0


def test_a_fetch_asks_the_archive_once_for_windows_that_overlap(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)

    project.fetch()

    # 2 July 03:00, 07:00 and 08:00 (-6 h) overlap; 20:00 and 9 July do not.
    assert len(archive.requests) == 3


# ---------------------------------------------------------------------------
# What is left to run
# ---------------------------------------------------------------------------


def _finish(project: Project, receptor) -> None:
    """Write particles for *receptor*, which completes a variant that makes no footprint."""
    sim = project.simulation(receptor.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-111.0],
            "lat": [40.0],
            "zagl": [10.0],
        }
    )
    sim.output.write_particles(sim.variant, receptor, particles, [])


def test_a_finished_simulation_needs_no_met(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)
    _finish(project, RECEPTORS[-1])  # the only receptor of 9 July

    files = project.fetch()

    assert len(files) == 5
    assert not any("20240709" in name for name in archive.downloaded)


def test_a_project_with_nothing_left_to_run_fetches_nothing(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS[:1])
    _finish(project, RECEPTORS[0])

    files = project.fetch()

    assert files.empty
    assert list(files.columns) == ["met", "file", "message"]
    assert archive.requests == []


def test_a_selection_limits_the_fetch(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)
    sims = project.simulations

    files = project.fetch(sims[sims["receptor"] == RECEPTORS[-1].id])

    assert len(files) == 2  # 8 July 18-23 and 9 July 00-05
    assert all("202407" in f for f in files["file"])
    assert not any("20240702" in f for f in files["file"])


# ---------------------------------------------------------------------------
# Files that cannot be had
# ---------------------------------------------------------------------------


def test_a_file_the_archive_lacks_is_reported_for_its_times_alone(tmp_path, archive):
    # 2 July 06:00 and 12:00 (-12 h) overlap, so they are asked for together;
    # only the second needs the 12-17 block.
    receptors = [_receptor(2, 6), _receptor(2, 12)]
    project = _project(tmp_path, receptors, n_hours=-12)
    archive.missing = {"20240702_12-17"}

    files = project.fetch()

    failed = files[files["message"].notna()]
    assert len(failed) == 1
    assert "2024-07-02 00:00 to 2024-07-02 12:00" in failed["message"].iloc[0]
    assert "not in the archive" in failed["message"].iloc[0]
    assert failed["file"].isna().all()
    have = files[files["file"].notna()]
    assert len(have) == 3  # 1 July 18-23, 2 July 00-05 and 06-11
    assert all(Path(f).is_file() for f in have["file"])


def test_a_date_before_the_archive_begins_is_reported(tmp_path, archive):
    project = _project(
        tmp_path, [make_receptor(dt.datetime(2015, 1, 1, 12), -111.8, 40.7)]
    )

    files = project.fetch()

    assert files["file"].isna().all()
    assert "archive begins" in files["message"].iloc[0]


def test_a_met_without_download_is_only_checked(tmp_path):
    met_dir = tmp_path / "met"
    receptor = make_receptor(dt.datetime(2023, 1, 1, 12), -111.85, 40.77)
    make_met_files(met_dir, receptor.time, -1)  # a run of -3 h needs more
    project = Project.init(
        tmp_path / "proj",
        receptors=[receptor],
        mets={"hrrr": make_met_config(met_dir, file_tres="1h")},
        n_hours=-3,
        variants={"hrrr": {}},
    )
    before = sorted(met_dir.iterdir())

    files = project.fetch()
    found = project.check()

    problems = files[files["message"].notna()]
    assert len(problems) == 1
    assert "2023-01-01 09:00" in problems["message"].iloc[0]
    assert list(found["check"]) == ["met_files"]
    assert "2023-01-01 09:00" in found["message"].iloc[0]
    assert sorted(met_dir.iterdir()) == before
    assert files["file"].isna().all()  # a run with a file missing has none to read


def test_a_met_without_download_that_is_complete_is_ready(tmp_path):
    met_dir = tmp_path / "met"
    receptor = make_receptor(dt.datetime(2023, 1, 1, 12), -111.85, 40.77)
    make_met_files(met_dir, receptor.time, -3)
    project = Project.init(
        tmp_path / "proj",
        receptors=[receptor],
        mets={"hrrr": make_met_config(met_dir, file_tres="1h")},
        n_hours=-3,
        variants={"hrrr": {}},
    )

    files = project.fetch()

    assert files["message"].isna().all()
    assert len(files) == 4  # 09:00 to 12:00, the run's hours


# ---------------------------------------------------------------------------
# Met.fetch
# ---------------------------------------------------------------------------


def _window(start: str, end: str):
    return pd.Timestamp(start), pd.Timestamp(end)


def test_met_fetch_asks_once_for_overlapping_windows_and_apart_for_the_rest(
    tmp_path, archive
):
    met = Met("hrrr", MetConfig(directory=tmp_path, download="hrrr"))

    files = met.fetch(
        [
            _window("2024-07-02 00:00", "2024-07-02 06:00"),
            _window("2024-07-02 04:00", "2024-07-02 10:00"),
            _window("2024-07-02 10:00", "2024-07-02 11:00"),  # only touches
            _window("2024-07-02 00:00", "2024-07-02 06:00"),  # a repeat
        ]
    )

    assert len(archive.requests) == 2
    assert files["message"].isna().all()


def test_met_fetch_errors_other_than_a_missing_file_are_not_swallowed(
    tmp_path, archive, monkeypatch
):
    met = Met("hrrr", MetConfig(directory=tmp_path, download="hrrr"))

    def no_extra(*args, **kwargs):
        raise ImportError("fsspec is required")

    monkeypatch.setattr(archive, "fetch", no_extra)

    with pytest.raises(ImportError, match="fsspec"):
        met.fetch([_window("2024-07-02 00:00", "2024-07-02 06:00")])


def test_a_model_with_no_fetch_adds_no_row(tmp_path, monkeypatch):
    from .test_check import _NoFilesModel

    monkeypatch.setitem(transport.MODELS, "toy", _NoFilesModel)
    project = Project.init(
        tmp_path / "proj",
        receptors=[_receptor(2, 3)],
        mets={"hrrr": {"directory": tmp_path / "met", "download": "hrrr"}},
        n_hours=-3,
        variants={"toy": {"model": "toy", "met": "hrrr"}},
    )

    assert project.fetch().empty


# ---------------------------------------------------------------------------
# stilt fetch
# ---------------------------------------------------------------------------


def test_stilt_fetch_prints_the_files_and_exits_with_0(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 0, result.output
    assert "Met hrrr: 7 files ready" in result.output
    assert len(archive.downloaded) == 7


def test_stilt_fetch_exits_with_1_when_a_file_could_not_be_had(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS, n_hours=-6)
    archive.missing = {"20240709"}

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 1
    assert "Met hrrr: 5 files ready" in result.output
    assert "could not get the files for these times" in result.output
    assert "2024-07-08 19:00 to 2024-07-09 01:00" in result.output


def test_stilt_fetch_of_a_finished_project_says_so(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS[:1])
    _finish(project, RECEPTORS[0])

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 0
    assert "No unfinished simulation needs met files." in result.output


def test_stilt_fetch_exits_with_2_for_a_met_it_cannot_read(tmp_path):
    receptor = _receptor(2, 3)
    project = Project.init(
        tmp_path / "proj",
        receptors=[receptor],
        mets={"hrrr": make_met_config(tmp_path / "met", file_tres="1h")},
        n_hours=-3,
        variants={"hrrr": {}},
    )
    (tmp_path / "met" / "header.arl").unlink()

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 2


def test_stilt_fetch_exits_with_2_for_a_directory_that_is_not_a_project(tmp_path):
    assert runner.invoke(app, ["fetch", str(tmp_path)]).exit_code == 2


def test_stilt_fetch_exits_with_3_when_interrupted(tmp_path, monkeypatch):
    project = _project(tmp_path, RECEPTORS[:1])

    def interrupt(self, sel=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(Project, "fetch", interrupt)

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 3
    assert "Interrupted" in result.output


def test_stilt_fetch_uses_the_output_directory_it_is_given(tmp_path, archive):
    project = _project(tmp_path, RECEPTORS[:1])
    other = tmp_path / "other_output"
    _finish(project, RECEPTORS[0])  # finished in the project's own output

    result = runner.invoke(
        app, ["fetch", str(project.directory), "--output", str(other)]
    )

    assert result.exit_code == 0
    assert "ready" in result.output  # nothing is finished in the other one
