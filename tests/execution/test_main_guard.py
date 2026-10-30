"""A script that runs a project on several cpus needs ``if __name__ == "__main__":`` (#274)."""

import contextlib
import os
import signal
import subprocess
import sys

import pytest

from stilt.execution import worker

from .test_worker import _TOY_PACKAGE

#: A script that makes a project and runs it on two cpus, under one start method.
_SCRIPT = """
import multiprocessing, sys
multiprocessing.set_start_method(sys.argv[1], force=True)

import stilt

{body}
"""

_RUN = """\
receptors = [
    stilt.PointReceptor(time=f"2023-01-01 0{h}:00", longitude=-111.85, latitude=40.77, altitude=5)
    for h in range(3)
]
project = stilt.Project.init(
    "project",
    receptors=receptors,
    model="toypool.ToyModel",
    mets={"toy": {"weather": "hrrr"}},
    variants={"toy": {}},
    n_hours=-1,
    hnf_plume=False,
    execution={"backend": "local", "cpus": 2},
)
project.run()
print("RAN")
"""

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the script runs in a process group"
)

START_METHODS = ["spawn", "forkserver"]


def _run_script(tmp_path, body, method):
    """Run the script in its own process group, and kill the group if it does not end."""
    (tmp_path / "toypool").mkdir()
    (tmp_path / "toypool" / "__init__.py").write_text(_TOY_PACKAGE)
    script = tmp_path / "script.py"
    script.write_text(_SCRIPT.format(body=body))
    proc = subprocess.Popen(
        [sys.executable, str(script), method],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"the script hung instead of stopping:\n{output[-2000:]}")
    finally:
        # A pool that kept starting workers would leave some behind.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    return proc.returncode, output


@pytest.mark.parametrize("method", START_METHODS)
def test_an_unguarded_script_stops_and_shows_the_fix(tmp_path, method):
    """The workers run the script again and fail; the run stops and shows the guard."""
    code, output = _run_script(tmp_path, _RUN, method)

    assert code != 0
    assert 'if __name__ == "__main__":' in output
    assert "RAN" not in output


@pytest.mark.parametrize("method", START_METHODS)
def test_a_guarded_script_runs(tmp_path, method):
    """The same script under the guard runs every receptor."""
    body = 'if __name__ == "__main__":\n' + "".join(
        f"    {line}\n" for line in _RUN.splitlines()
    )
    code, output = _run_script(tmp_path, body, method)

    assert code == 0, output
    assert "RAN" in output
    particles = (tmp_path / "project").glob("output/particles/*/date=*/*.parquet")
    assert len(list(particles)) == 3


def test_fork_skips_the_check(monkeypatch):
    """Fork runs no script again, so no worker is started to try it."""
    monkeypatch.setattr(worker.multiprocessing, "get_start_method", lambda: "fork")
    monkeypatch.setattr(
        worker.multiprocessing,
        "Process",
        lambda *a, **k: pytest.fail("a worker was started"),
    )
    worker._check_workers_can_start(2)


class _FakeWorker:
    """A worker that cannot start, or that exits with the given code."""

    def __init__(self, *, start_error=None, exitcode=None, **kwargs):
        self.start_error = start_error
        self.exitcode = exitcode

    def start(self):
        if self.start_error is not None:
            raise self.start_error

    def join(self):
        pass


@pytest.mark.parametrize(
    "worker_kwargs",
    [
        {"exitcode": 1},
        {"start_error": RuntimeError("did not receive acknowledgement of fd")},
        {"start_error": ConnectionResetError("Connection reset by peer")},
    ],
)
def test_a_worker_that_cannot_start_raises_with_the_guard(monkeypatch, worker_kwargs):
    """A failed start (an exit code, or the error forkserver raises) shows the guard."""
    monkeypatch.setattr(worker.multiprocessing, "get_start_method", lambda: "spawn")
    monkeypatch.setattr(
        worker.multiprocessing,
        "Process",
        lambda *a, **k: _FakeWorker(**worker_kwargs),
    )
    with pytest.raises(RuntimeError, match=r'if __name__ == "__main__":'):
        worker._check_workers_can_start(2)


def test_a_worker_that_starts_lets_the_run_go_on(monkeypatch):
    """A worker that exits with 0 means the script can run again in a worker."""
    monkeypatch.setattr(worker.multiprocessing, "get_start_method", lambda: "spawn")
    monkeypatch.setattr(
        worker.multiprocessing,
        "Process",
        lambda *a, **k: _FakeWorker(exitcode=0),
    )
    worker._check_workers_can_start(2)
