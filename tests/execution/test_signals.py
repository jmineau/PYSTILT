"""Tests for the worker's SIGTERM handling."""

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from stilt.execution.worker import _sigterm_as_interrupt


def test_sigterm_as_interrupt_restores_handler():
    """Original SIGTERM handler is restored after the context exits."""
    original = signal.getsignal(signal.SIGTERM)
    with _sigterm_as_interrupt():
        assert signal.getsignal(signal.SIGTERM) != original
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_raises_keyboard_interrupt():
    """SIGTERM delivered inside the context raises KeyboardInterrupt."""
    with pytest.raises(KeyboardInterrupt), _sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)


def test_sigterm_as_interrupt_restores_after_exception():
    """Handler is restored even when an exception interrupts the block."""
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), _sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_is_noop_outside_main_thread():
    """Called from a non-main thread, the SIGTERM handler is left untouched."""
    original = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def _worker() -> None:
        with _sigterm_as_interrupt():
            seen["inside"] = signal.getsignal(signal.SIGTERM)
        seen["after"] = signal.getsignal(signal.SIGTERM)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert seen["inside"] == original
    assert seen["after"] == original
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_leaves_an_existing_handler_alone():
    """A SIGTERM handler the caller set stays in charge."""
    seen: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    try:
        handler = signal.getsignal(signal.SIGTERM)
        with _sigterm_as_interrupt():
            assert signal.getsignal(signal.SIGTERM) is handler
            signal.raise_signal(signal.SIGTERM)  # no KeyboardInterrupt
        assert seen == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------
# stilt run --cpus N, stopped by SIGTERM (#309)
# ---------------------------------------------------------------------------

#: A transport model whose run is a fake ``hycs_std`` that sleeps while ``slow`` exists.
_SLOW_PACKAGE = """
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from stilt.transport import ModelRun, TransportConfig
from stilt.transport.hysplit.driver import run_hycs_std

class SlowConfig(TransportConfig):
    pass

class SlowMet(BaseModel):
    weather: str

    def settings(self):
        return {"weather": self.weather}

class SlowModel:
    name = "slow"
    config_class = SlowConfig
    met_config_class = SlowMet

    def version(self, config):
        return "1.0"

    def data_files(self, config):
        return None

    def run(self, receptor, config, met, window, workdir=None, timeout=None):
        here = Path(os.environ["SLOW_DIR"])
        exe = workdir / "hycs_std"
        exe.write_text(
            "#!/bin/sh\\n"
            f"if [ -e {here}/slow ]; then echo $$ >> {here}/pids; exec sleep 60; fi\\n"
        )
        exe.chmod(0o755)
        run_hycs_std(workdir)
        return ModelRun(particles=pd.DataFrame({
            "particle": [1, 2, 1, 2],
            "age": [0, 0, -60, -60],
            "lon": receptor.longitude,
            "lat": receptor.latitude,
            "zagl": receptor.altitude,
            "foot": [0.0, 0.0, 0.01, 0.01],
        }))
"""

#: Starts ``stilt run`` with one multiprocessing start method.
_LAUNCH = """
import multiprocessing, sys
if __name__ == "__main__":
    multiprocessing.set_start_method(sys.argv[1])
    sys.argv = ["stilt", *sys.argv[2:]]
    from stilt.cli import app
    app()
"""

START_METHODS = [
    pytest.param(
        "fork",
        marks=pytest.mark.skipif(
            sys.platform == "darwin", reason="fork is unsafe on macOS"
        ),
    ),
    "spawn",
    "forkserver",
]


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals and sh")
@pytest.mark.parametrize("method", START_METHODS)
def test_sigterm_stops_a_run_on_two_cpus_with_exit_3(tmp_path, method):
    """SIGTERM ends the HYSPLIT runs and the workers, and ``stilt run`` exits 3 (#309)."""
    (tmp_path / "slowpool").mkdir()
    (tmp_path / "slowpool" / "__init__.py").write_text(_SLOW_PACKAGE)
    (tmp_path / "launch.py").write_text(_LAUNCH)
    (tmp_path / "slow").touch()
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(
        "model: slowpool.SlowModel\n"
        "n_hours: -1\n"
        "mets:\n  slow:\n    weather: hrrr\n"
        "variants:\n  slow: {}\n"
    )
    (project / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n"
        + "".join(f"2023-01-01 {h:02d}:00:00,-111.85,40.77,5.0\n" for h in range(2))
    )
    command = [
        sys.executable,
        str(tmp_path / "launch.py"),
        method,
        "run",
        str(project),
        "--cpus",
        "2",
        "--compute-root",
        str(tmp_path / "work"),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), os.environ.get("PYTHONPATH", "")]
        ),
        "SLOW_DIR": str(tmp_path),
    }
    pids = tmp_path / "pids"
    # Its own process group, so a failure leaves nothing running.
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            assert proc.poll() is None, proc.communicate()[0]
            assert time.monotonic() < deadline, "the fake hycs_std never ran twice"
            time.sleep(0.1)

        proc.send_signal(signal.SIGTERM)
        started = time.monotonic()
        try:
            output, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("stilt run did not stop within 20 s of SIGTERM")
        assert time.monotonic() - started < 20
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        # The fake hycs_std runs have sessions of their own.
        for pid in pids.read_text().split() if pids.exists() else []:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)

    assert proc.returncode == 3, output
    assert "stopped after 0 of 2 receptors; the rest did not run" in output
    assert "interrupted=2" in output
    # The fake hycs_std runs were stopped, not left behind.
    for pid in map(int, pids.read_text().split()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    # Nothing was written that a rerun would take for finished.
    assert not list((project / "output").glob("particles/*/date=*/*.parquet"))

    (tmp_path / "slow").unlink()
    done = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(list((project / "output").glob("particles/*/date=*/*.parquet"))) == 2
