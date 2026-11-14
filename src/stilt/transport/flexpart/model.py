"""FLEXPART 11 as a transport model."""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.exceptions import SimulationError
from stilt.transport import ModelRun

from .config import FlexpartConfig, FlexpartMetConfig, flexpart_version
from .inputs import write_inputs
from .met import catalog, files_for
from .output import read_partoutput

if TYPE_CHECKING:
    from stilt.receptors import Receptor

#: FLEXPART's output, kept in the workdir as the run's log.
LOG_FILE = "flexpart.log"

#: The line FLEXPART ends a run that worked with.
SUCCESS = "CONGRATULATIONS"

#: The lines of FLEXPART's output kept in a failure's message.
TAIL_LINES = 20


class FlexpartModel:
    """
    FLEXPART 11 as a transport model, the object ``get_model("flexpart")`` returns.

    :meth:`run` picks the GRIB files a receptor needs, writes FLEXPART's
    input files into the workdir (:func:`~stilt.transport.flexpart.inputs.write_inputs`),
    runs the checkout's executable there, and reads its particle output
    (:func:`~stilt.transport.flexpart.read_partoutput`). The particles have
    no ``foot``: PYSTILT computes it from FLEXPART's ``mlht`` and the GRIB
    files' pressure (:func:`stilt.particles.add_foot`).
    """

    name = "flexpart"
    config_class = FlexpartConfig
    #: FLEXPART reads GRIB files, by valid time.
    met_config_class = FlexpartMetConfig
    #: FLEXPART writes its inputs and outputs as files in a directory.
    needs_workdir = True
    #: One FLEXPART process per receptor.
    batched = False

    def version(self, config: FlexpartConfig) -> str:
        """Return the version of the checkout in ``config.flexpart_dir``."""
        return flexpart_version(config.checkout)

    def data_files(self, config: FlexpartConfig) -> dict[str, str] | None:
        """Return ``None``: the land-use and surface tables are the checkout's, part of its version."""
        return None

    def run(
        self,
        receptor: Receptor,
        config: FlexpartConfig,
        met: FlexpartMetConfig,
        window: tuple[dt.datetime, dt.datetime],
        workdir: Path | None = None,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Run FLEXPART for one receptor in *workdir* and return its particles.

        Raises
        ------
        MeteorologyError
            Before FLEXPART runs, when the met files do not cover *window*
            or one between is missing.
        SimulationError
            With FLEXPART's output in ``log``: ``reason`` ``TIMEOUT`` when
            the run exceeded *timeout* seconds, ``EXIT_STATUS`` when
            FLEXPART stopped without finishing, and ``NO_PARTICLE_DATA``
            when it finished but wrote no particle file.
        """
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix="stilt-") as tmp:
                return self.run(receptor, config, met, window, Path(tmp), timeout)
        met_files = files_for(catalog(met.directory, met.pattern), window)
        write_inputs(workdir, receptor, config, met, window, met_files)
        try:
            run_flexpart(workdir, config, timeout)
            outputs = sorted((workdir / "output").glob("partoutput_*.nc"))
            if not outputs:
                raise SimulationError(
                    "FLEXPART wrote no partoutput file.", reason="NO_PARTICLE_DATA"
                )
            particles = read_partoutput(outputs, receptor.time)
        except SimulationError as error:
            error.log = _read_log(workdir)
            raise
        return ModelRun(
            particles=particles,
            log=_read_log(workdir),
            met_files=list(met_files.values()),
        )


def run_flexpart(
    workdir: Path, config: FlexpartConfig, timeout: int | None = None
) -> None:
    """
    Run the checkout's FLEXPART in *workdir*, writing its output to ``flexpart.log``.

    It runs with ``config.threads`` OpenMP threads. If this process is
    interrupted while it waits, FLEXPART is stopped too.

    Raises
    ------
    SimulationError
        ``TIMEOUT`` when it ran longer than *timeout* seconds, or
        ``EXIT_STATUS`` when it ended without its closing line, with the
        last lines of its output.
    """
    log_path = workdir / LOG_FILE
    env = {**os.environ, "OMP_NUM_THREADS": str(config.threads)}
    with (
        log_path.open("w", encoding="utf-8") as handle,
        subprocess.Popen(
            [str(config.executable), "pathnames"],
            cwd=workdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        ) as proc,
    ):
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _terminate(proc)
            raise SimulationError(
                f"FLEXPART ran longer than the {timeout} s timeout.", reason="TIMEOUT"
            ) from e
        except BaseException:
            # FLEXPART runs in a session of its own, so no signal reached it.
            _terminate(proc)
            raise
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0 or SUCCESS not in output:
        tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])
        raise SimulationError(
            f"FLEXPART stopped with exit status {proc.returncode}:\n{tail}",
            reason="EXIT_STATUS",
        )


def _terminate(proc: subprocess.Popen) -> None:
    """Stop FLEXPART's process group, and wait for it."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _read_log(workdir: Path) -> str:
    """Return FLEXPART's output in *workdir*, or an empty string when there is none."""
    path = workdir / LOG_FILE
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


__all__ = ["FlexpartModel", "run_flexpart"]
