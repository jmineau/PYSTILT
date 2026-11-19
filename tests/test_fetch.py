"""Tests for Project.fetch and stilt fetch: the met files the unfinished simulations need, got once."""

import datetime as dt
from pathlib import Path

import arlmet.archives
import pandas as pd
import pytest
from arlmet.archives import HRRRArchive
from runtraj.project import Project
from typer.testing import CliRunner

from stilt.cli import app
from tests.fixtures.factories import blocks

from .fixtures.factories import (
    make_met_config,
    make_release,
)

runner = CliRunner()

BOUNDS = {"xmin": -113.0, "xmax": -110.0, "ymin": 39.0, "ymax": 42.0}


class FakeHRRR(HRRRArchive):
    """
    HRRR's file names and 6 h blocks, with no network.

    ``_download`` and ``_fetch_and_crop`` write a few bytes and note the
    file, so a test can say what was downloaded, in how many requests, and
    that a second ask downloads nothing. A file whose URL holds a string in
    ``missing`` is not on the server. When ``error`` is set, every download
    raises it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.downloaded: list[str] = []
        self.requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.missing: set[str] = set()
        self.error: Exception | None = None

    def fetch(self, start, end, **kwargs):
        self.requests.append((pd.Timestamp(start), pd.Timestamp(end)))
        return super().fetch(start, end, **kwargs)

    def _download(self, url, dest, opts, **kwargs):
        if self.error is not None:
            raise self.error
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


def _release(day: int, hour: int):
    return make_release(dt.datetime(2024, 7, day, hour), -111.85, 40.77)


def _project(tmp_path: Path, releases, *, n_hours: int = -3, **met) -> Project:
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
        releases=releases,
        **blocks(mets={"hrrr": fields}, n_hours=n_hours, variants={"hrrr": {}}),
    )


# ---------------------------------------------------------------------------
# The files a run would ask for
# ---------------------------------------------------------------------------

#: Hours that reach across a block, fill one block, and are a week apart.
RELEASES = [
    _release(2, 3),
    _release(2, 7),
    _release(2, 8),
    _release(2, 20),
    _release(9, 1),
]


# ---------------------------------------------------------------------------
# What is left to run
# ---------------------------------------------------------------------------


def _finish(project: Project, release) -> None:
    """Write particles for *release*, which completes a variant that makes no footprint."""
    sim = project.simulation(release.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-111.0],
            "lat": [40.0],
            "zagl": [10.0],
        }
    )
    sim.output.write_particles(sim.variant, release, particles, [])


def test_a_finished_simulation_needs_no_met(tmp_path, archive):
    project = _project(tmp_path, RELEASES, n_hours=-6)
    _finish(project, RELEASES[-1])  # the only release of 9 July

    files = project.fetch()

    assert len(files) == 5
    assert not any("20240709" in name for name in archive.downloaded)


def test_a_project_with_nothing_left_to_run_fetches_nothing(tmp_path, archive):
    project = _project(tmp_path, RELEASES[:1])
    _finish(project, RELEASES[0])

    files = project.fetch()

    assert files.empty
    assert list(files.columns) == ["met", "file", "message"]
    assert archive.requests == []


# ---------------------------------------------------------------------------
# Files that cannot be had
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Met.fetch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# stilt fetch
# ---------------------------------------------------------------------------


def test_stilt_fetch_prints_the_files_and_exits_with_0(tmp_path, archive):
    project = _project(tmp_path, RELEASES, n_hours=-6)

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 0, result.output
    assert "Met hrrr: 7 files ready" in result.output
    assert len(archive.downloaded) == 7


def test_stilt_fetch_exits_with_1_when_a_file_could_not_be_had(tmp_path, archive):
    project = _project(tmp_path, RELEASES, n_hours=-6)
    archive.missing = {"20240709"}

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 1
    assert "Met hrrr: 5 files ready" in result.output
    assert "could not get the files for these times" in result.output
    assert "2024-07-08 19:00 to 2024-07-09 01:00" in result.output


def _s3_answer(status, code):
    """Return what s3fs raises for S3's answer with this status and error code."""
    translate = pytest.importorskip("s3fs.errors").translate_boto_error
    botocore = pytest.importorskip("botocore.exceptions")
    answer = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code}}
    return translate(botocore.ClientError(answer, "GetObject"))


def _botocore(name):
    """Return botocore's error of this name, as a failed connection raises it."""
    return getattr(pytest.importorskip("botocore.exceptions"), name)(endpoint_url="u")


def _http_failed(status):
    """Return what fsspec's HTTP filesystem raises for any failure: ``FileNotFoundError``, from the error."""
    answered = RuntimeError("the server answered")
    answered.status = status
    error = FileNotFoundError("http://archive/file")
    error.__cause__ = answered
    return error


@pytest.mark.parametrize(
    ("make_error", "code"),
    [
        # The archive does not have the file: a retry would not help.
        pytest.param(lambda: _s3_answer(404, "NoSuchKey"), 1, id="NoSuchKey 404"),
        pytest.param(lambda: _http_failed(404), 1, id="http 404"),
        pytest.param(lambda: FileNotFoundError("gone"), 1, id="file not found"),
        # The archive is busy or the connection failed: run it again.
        pytest.param(lambda: _s3_answer(503, "SlowDown"), 3, id="SlowDown 503"),
        pytest.param(
            lambda: _s3_answer(500, "InternalError"), 3, id="InternalError 500"
        ),
        pytest.param(lambda: _http_failed(503), 3, id="http 503"),
        pytest.param(lambda: _http_failed(429), 3, id="http 429"),
        pytest.param(lambda: ConnectionResetError("reset"), 3, id="reset"),
        pytest.param(lambda: _botocore("ReadTimeoutError"), 3, id="read timeout"),
        pytest.param(lambda: _botocore("ConnectTimeoutError"), 3, id="connect timeout"),
        pytest.param(lambda: _botocore("EndpointConnectionError"), 3, id="refused"),
        # Anything else is not "the file is missing": the command group sorts it.
        pytest.param(lambda: PermissionError("denied"), 4, id="denied"),
        pytest.param(lambda: _s3_answer(403, "AccessDenied"), 4, id="AccessDenied 403"),
    ],
)
def test_stilt_fetch_exits_with_1_only_for_a_file_the_archive_lacks(
    tmp_path, archive, make_error, code
):
    project = _project(tmp_path, RELEASES[:1])
    archive.error = make_error()

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == code, result.output
    if code == 1:
        assert "could not get the files for these times" in result.output


def test_a_busy_archive_stops_a_fetch_at_the_first_window(tmp_path, archive):
    """A throttled archive is asked once, not once for each window."""
    project = _project(tmp_path, RELEASES, n_hours=-6)
    archive.error = _s3_answer(503, "SlowDown")

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 3
    assert len(archive.requests) == 1


def test_stilt_fetch_of_a_finished_project_says_so(tmp_path, archive):
    project = _project(tmp_path, RELEASES[:1])
    _finish(project, RELEASES[0])

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 0
    assert "No unfinished simulation needs met files." in result.output


def test_stilt_fetch_exits_with_2_for_a_met_it_cannot_read(tmp_path):
    release = _release(2, 3)
    project = Project.init(
        tmp_path / "proj",
        releases=[release],
        **blocks(
            mets={"hrrr": make_met_config(tmp_path / "met")},
            n_hours=-3,
            variants={"hrrr": {}},
        ),
    )
    (tmp_path / "met" / "header.arl").unlink()

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 2


def test_stilt_fetch_exits_with_2_for_a_directory_that_is_not_a_project(tmp_path):
    assert runner.invoke(app, ["fetch", str(tmp_path)]).exit_code == 2


def test_stilt_fetch_exits_with_3_when_interrupted(tmp_path, monkeypatch):
    project = _project(tmp_path, RELEASES[:1])

    def interrupt(self, sel=None, **_):
        raise KeyboardInterrupt

    monkeypatch.setattr(Project, "fetch", interrupt)

    result = runner.invoke(app, ["fetch", str(project.directory)])

    assert result.exit_code == 3
    assert "Interrupted" in result.output


def test_stilt_fetch_uses_the_output_directory_it_is_given(tmp_path, archive):
    project = _project(tmp_path, RELEASES[:1])
    other = tmp_path / "other_output"
    _finish(project, RELEASES[0])  # finished in the project's own output

    result = runner.invoke(
        app, ["fetch", str(project.directory), "--output", str(other)]
    )

    assert result.exit_code == 0
    assert "ready" in result.output  # nothing is finished in the other one


def test_stilt_fetch_gets_only_the_releases_it_is_given(tmp_path, archive):
    project = _project(tmp_path, RELEASES, n_hours=-6)
    ids = tmp_path / "ids.txt"
    ids.write_text(RELEASES[-1].id + "\n")

    result = runner.invoke(
        app, ["fetch", str(project.directory), "--releases", str(ids)]
    )

    assert result.exit_code == 0, result.output
    assert "releases=1" in result.output
    assert len(archive.downloaded) == 2  # 8 July 18-23 and 9 July 00-05
    assert not any("20240702" in name for name in archive.downloaded)


def test_stilt_fetch_reads_a_submissions_releases_parquet(tmp_path, archive):
    """The fetch job of ``stilt submit`` reads the rows the array reads (#436)."""
    import pyarrow as pa

    from stilt._paths import write_parquet

    project = _project(tmp_path, RELEASES, n_hours=-6)
    rows = project._rows_of([RELEASES[-1].id])
    parquet = tmp_path / "releases.parquet"
    write_parquet(pa.Table.from_pandas(rows, preserve_index=False), parquet)

    result = runner.invoke(
        app, ["fetch", str(project.directory), "--releases", str(parquet)]
    )

    assert result.exit_code == 0, result.output
    assert len(archive.downloaded) == 2


def test_stilt_fetch_refuses_a_release_not_in_the_project(tmp_path, archive):
    project = _project(tmp_path, RELEASES[:1])
    ids = tmp_path / "ids.txt"
    ids.write_text("not-a-release\n")

    result = runner.invoke(
        app, ["fetch", str(project.directory), "--releases", str(ids)]
    )

    assert result.exit_code == 2
    assert "not in this project" in result.output
    assert archive.downloaded == []


def test_stilt_fetch_no_skip_fetches_for_runs_whose_particles_exist(
    tmp_path, archive, monkeypatch
):
    project = _project(tmp_path, RELEASES, n_hours=-6)
    real_unrun = Project._unrun

    def unrun(self, sel=None, *, skip_existing=True):
        if skip_existing:  # as if every particles file existed
            return self._selected(sel).iloc[:0]
        return real_unrun(self, sel, skip_existing=False)

    monkeypatch.setattr(Project, "_unrun", unrun)

    assert runner.invoke(app, ["fetch", str(project.directory)]).exit_code == 0
    assert archive.downloaded == []
    result = runner.invoke(app, ["fetch", str(project.directory), "--no-skip"])
    assert result.exit_code == 0, result.output
    assert archive.downloaded
