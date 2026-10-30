"""Tests for stilt fetch stopped by a signal: SIGTERM as a pod or Slurm stops it, and Ctrl-C."""

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from .fixtures.factories import write_arl_file
from .test_fetch import _project, _receptor

#: Runs ``stilt fetch`` in a process of its own, against an HRRR archive that
#: "downloads" a local file (arlmet's own download code runs, with its partial
#: file) and, unless told ``fast``, takes about ten seconds to copy it.
DRIVER = textwrap.dedent(
    """
    import shutil, sys, time
    from pathlib import Path

    import arlmet.archives as archives

    project, source, speed = sys.argv[1], Path(sys.argv[2]), sys.argv[3]


    class Local(archives.HRRRArchive):
        def _url(self, path, mirror):
            return str(source)

        def _storage_options(self, mirror):
            return {}


    def copy(src, dst, length=0):
        while chunk := src.read(1024):
            dst.write(chunk)
            if speed == "slow":
                time.sleep(0.01)


    shutil.copyfileobj = copy
    archives.get_archive = lambda name, **options: Local()
    sys.argv = ["stilt", "fetch", project]
    from stilt.cli import app

    app()
    """
)


def _fetch_process(tmp_path: Path, speed: str) -> subprocess.Popen:
    """Start ``stilt fetch`` for the project in *tmp_path*, and return its process."""
    source = tmp_path / "remote"
    if not source.exists():
        write_arl_file(source)
        source.write_bytes(source.read_bytes() * 100)  # about 1 MB; copied whole
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    return subprocess.Popen(
        [sys.executable, str(driver), str(tmp_path / "proj"), str(source), speed],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_partial(met: Path, process: subprocess.Popen) -> None:
    """Return once the download has begun, which is when its partial file exists."""
    for _ in range(600):
        if met.is_dir() and any(p.name.endswith(".partial") for p in met.iterdir()):
            return
        assert process.poll() is None, process.communicate()[0]
        time.sleep(0.05)
    raise AssertionError("the download never started")


@pytest.mark.parametrize(
    "signals",
    [
        [signal.SIGTERM],
        [signal.SIGINT],
        [signal.SIGTERM, signal.SIGTERM],
        [signal.SIGINT, signal.SIGINT],
        [signal.SIGTERM, signal.SIGINT],
    ],
    ids=["term", "int", "term-twice", "int-twice", "term-then-int"],
)
def test_a_stopped_fetch_exits_with_3_and_leaves_no_partial_file(tmp_path, signals):
    _project(tmp_path, [_receptor(2, 3)], subgrid_enable=False)
    met = tmp_path / "met"
    process = _fetch_process(tmp_path, "slow")
    _wait_for_partial(met, process)

    for sig in signals:
        process.send_signal(sig)
        time.sleep(0.02)
    output = process.communicate(timeout=60)[0]

    assert process.returncode == 3, output
    assert "Interrupted" in output
    assert sorted(p.name for p in met.iterdir()) == []


def test_fetching_again_after_a_stop_gets_the_files_and_exits_with_0(tmp_path):
    _project(tmp_path, [_receptor(2, 3)], subgrid_enable=False)
    met = tmp_path / "met"
    process = _fetch_process(tmp_path, "slow")
    _wait_for_partial(met, process)
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=60)

    again = _fetch_process(tmp_path, "fast")
    output = again.communicate(timeout=120)[0]

    assert again.returncode == 0, output
    assert "Met hrrr: 1 file ready" in output
    assert [p.name for p in met.iterdir()] == ["20240702_00-05_hrrr"]
