"""Set up and run one HYSPLIT simulation."""

import os
import platform
import signal
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stilt.config import (
    STILTParams,
    kmsl_from_vertical_reference,
)
from stilt.exceptions import (
    HYSPLITFailureError,
    HYSPLITNotFoundError,
    HYSPLITTimeoutError,
    NoParticleOutputError,
)
from stilt.receptors import Receptor
from stilt.transport.hysplit.control import ControlFile
from stilt.transport.hysplit.failures import (
    FAILURE_PHRASES,
    MET_TRUNCATED_WARNING,
    FailureReason,
)
from stilt.transport.hysplit.namelist import NameList

CONTROL_FILE = "CONTROL"
SETUP_FILE = "SETUP.CFG"
HYCS_STD_FILE = "hycs_std"
PARTICLE_STILT_FILE = "PARTICLE_STILT.DAT"
PARTICLE_FILE = "PARTICLE.DAT"
WINDERR_FILE = "WINDERR"
ZIERR_FILE = "ZIERR"
ZICONTROL_FILE = "ZICONTROL"


def _bundled_exe_dir() -> Path:
    """
    Return the directory of the bundled ``hycs_std`` for this platform.

    The bundled builds are x86-64 Linux and x86-64 macOS. The macOS build
    also runs on Apple Silicon through Rosetta. Any other platform raises,
    rather than running a binary the operating system cannot execute. So
    does an install from the source archive, which carries no binary.
    """
    system = platform.system()
    machine = platform.machine().lower()
    subdir = None
    if system == "Linux" and machine in ("x86_64", "amd64"):
        subdir = "linux_x64"
    elif system == "Darwin" and machine in ("x86_64", "arm64"):
        subdir = "macos_x64"
    if subdir is not None:
        exe_dir = Path(str(pkg_files("stilt.transport.hysplit") / "bin" / subdir))
        if (exe_dir / HYCS_STD_FILE).is_file():
            return exe_dir
    raise HYSPLITNotFoundError(
        f"No bundled HYSPLIT binary for {system} {platform.machine()}. "
        "Build hycs_std for this machine and set exe_dir in config.yaml "
        "to the directory that holds it."
    )


def _bundled_data_dir() -> Path:
    """Return the directory of HYSPLIT's bundled data tables."""
    return Path(str(pkg_files("stilt.transport.hysplit") / "data"))


def _read_particle_dat(path: Path, names: Sequence[str]) -> pd.DataFrame:
    """
    Read a ``PARTICLE_STILT.DAT`` file, with ``names`` as its columns.

    ``numpy.loadtxt`` reads large files much faster than pandas and handles
    HYSPLIT's variable-width spacing.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="loadtxt: input contained no data",
            category=UserWarning,
        )
        values = np.loadtxt(path, skiprows=1, ndmin=2)

    if values.size == 0:
        return pd.DataFrame(columns=pd.Index(names))
    if values.shape[1] != len(names):
        raise ValueError(
            f"{path.name} has {values.shape[1]} columns, expected {len(names)} "
            f"from varsiwant={list(names)!r}."
        )
    return pd.DataFrame(values, columns=pd.Index(names))


@dataclass
class HYSPLITResult:
    """
    Output of one HYSPLIT run.

    Attributes
    ----------
    particles : pandas.DataFrame
        Particle table read from ``PARTICLE_STILT.DAT``, one column per
        ``varsiwant`` variable.
    log_path : Path
        Log file holding the run's standard output.
    """

    particles: pd.DataFrame
    log_path: Path


def _write_values(path: Path, values: list[float] | None) -> None:
    """Write one value per line to ``path``, or remove it when ``values`` is ``None``."""
    if values is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")


class HYSPLITDriver:
    """
    Run HYSPLIT's ``hycs_std`` once, for one receptor in one directory.

    :meth:`prepare` writes the input files (``CONTROL``, ``SETUP.CFG``, and
    the error files) and :meth:`execute` runs the binary and reads the
    particles it writes. :class:`~stilt.transport.hysplit.HysplitModel` makes one per
    run. Use it directly only to run HYSPLIT outside a project.

    Parameters
    ----------
    receptor : Receptor
        Receptor to release particles from.
    params : STILTParams
        Transport and error settings.
    met_files : list of Path
        Meteorology files, in the order HYSPLIT should read them.
    directory : Path, optional
        Simulation directory to run in.
    exe_dir : Path, optional
        Directory holding ``hycs_std``. Defaults to ``params.exe_dir``, then
        to the bundled build.
    data_dir : Path, optional
        Directory of HYSPLIT data tables. Defaults to the bundled tables.
    """

    def __init__(
        self,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
        directory: Path | None = None,
        exe_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        self.directory = (
            Path(tempfile.mkdtemp(prefix="pystilt_"))
            if directory is None
            else Path(directory).expanduser().resolve()
        )
        self.control_path = self.directory / CONTROL_FILE
        self.setup_path = self.directory / SETUP_FILE
        self.hycs_std_path = self.directory / HYCS_STD_FILE
        self.log_path = self.directory / "stilt.log"
        self.particle_stilt_path = self.directory / PARTICLE_STILT_FILE
        self.particle_path = self.directory / PARTICLE_FILE
        self.winderr_path = self.directory / WINDERR_FILE
        self.zierr_path = self.directory / ZIERR_FILE
        self.zicontrol_path = self.directory / ZICONTROL_FILE
        self.receptor = receptor
        self.params = params
        self.met_files = met_files
        # explicit argument > STILTParams.exe_dir > binary bundled with the package
        chosen = exe_dir if exe_dir is not None else params.exe_dir
        self.exe_dir = Path(chosen) if chosen is not None else _bundled_exe_dir()
        self.data_dir = Path(data_dir) if data_dir is not None else _bundled_data_dir()

    def prepare(self) -> None:
        """
        Create the simulation directory and write HYSPLIT's input files.

        Links ``hycs_std`` and the data tables into the directory and writes
        ``CONTROL`` and ``SETUP.CFG``, plus ``ZICONTROL``, ``WINDERR``, and
        ``ZIERR`` when the settings call for them.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        # Symlink the binary from exe_dir and data files from data_dir (mirrors
        # STILT-R). Only hycs_std is taken from exe_dir: a custom build directory
        # usually holds a whole HYSPLIT exec/ tree we have no business linking.
        exe = self.exe_dir / HYCS_STD_FILE
        if not exe.is_file():
            raise HYSPLITNotFoundError(
                f"No {HYCS_STD_FILE!r} executable in {self.exe_dir}. "
                "Check STILTParams.exe_dir."
            )
        # A reused simulation directory may still point at a different build.
        if self.hycs_std_path.is_symlink() and (
            self.hycs_std_path.resolve() != exe.resolve()
        ):
            self.hycs_std_path.unlink()
        for f in [exe, *self.data_dir.iterdir()]:
            link = self.directory / f.name
            if not link.exists():
                link.symlink_to(f.resolve())

        # Write HYSPLIT CONTROL
        ControlFile(
            receptor=self.receptor,
            n_hours=self.params.n_hours,
            emisshrs=self.params.emisshrs,
            w_option=self.params.w_option,
            z_top=self.params.z_top,
            met_files=self.met_files,
        ).write(self.control_path)

        # SETUP.CFG carries winderrtf; WINDERR / ZIERR exist only when perturbed.
        self._write_setup()
        self._write_zicontrol()
        self._write_winderr()
        self._write_zierr()

    def execute(self, timeout: int | None = None) -> HYSPLITResult:
        """
        Run HYSPLIT once and read its particle output.

        Parameters
        ----------
        timeout : int or None
            Time limit for the run, in seconds. ``None`` waits indefinitely.

        Returns
        -------
        HYSPLITResult
            The particles and the log path.

        Raises
        ------
        HYSPLITTimeoutError
            The run exceeded ``timeout``.
        HYSPLITFailureError
            The log shows a known HYSPLIT failure, or a met file was cut
            short and the particles stop before the end of the run.
        NoParticleOutputError
            HYSPLIT wrote no ``PARTICLE_STILT.DAT``.
        """
        self.particle_stilt_path.unlink(missing_ok=True)
        self.particle_path.unlink(missing_ok=True)
        log_start = self._run(timeout, label="hycs_std")
        particles = self._read_particles()
        self._check_met_reached_end(particles, log_start)
        return HYSPLITResult(particles=particles, log_path=self.log_path)

    # -- Private helpers -------------------------------------------------------

    def _run(self, timeout: int | None, *, label: str = "hycs_std") -> int:
        """Run ``hycs_std``, appending its output to the log, and return the offset it starts at."""
        if not self.hycs_std_path.exists():
            raise HYSPLITNotFoundError(
                f"HYSPLIT executable not found for {self.directory}: {self.hycs_std_path}"
            )
        segment_start = self.log_path.stat().st_size if self.log_path.exists() else 0
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {label} run ===\n")
            handle.flush()
            with subprocess.Popen(
                [str(self.hycs_std_path)],
                cwd=self.directory,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as proc:
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired as e:
                    self._terminate_process(proc)
                    raise HYSPLITTimeoutError(
                        f"hycs_std timed out after {timeout}s for {self.directory}"
                    ) from e
        self._check_log_for_failure(segment_start)
        return segment_start

    def _terminate_process(self, proc: subprocess.Popen[Any]) -> None:
        """Stop a HYSPLIT process group with SIGTERM, then SIGKILL if it does not exit."""
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.wait()

    def _check_log_for_failure(self, start: int) -> None:
        """
        Raise if the log written since byte ``start`` shows a known HYSPLIT failure.

        Seeking to a byte offset in text mode is safe here because
        ``hycs_std`` writes no multi-byte characters.
        """
        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start)
            for line in handle:
                for phrase, reason in FAILURE_PHRASES.items():
                    if phrase in line:
                        raise HYSPLITFailureError(reason, self.log_path)

    def _check_met_reached_end(self, particles: pd.DataFrame, log_start: int) -> None:
        """
        Raise if a met file was cut short and no particle reaches the end of the run.

        HYSPLIT only warns when a met file holds one time period, and its
        particles stop where the met runs out. The warning alone is not a
        failure, since the damaged file may cover hours the particles never
        reach. Particles that leave the met domain do not trigger this
        either, unless the warning is also in the log.
        """
        if particles.empty:
            return
        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(log_start)
            if MET_TRUNCATED_WARNING not in handle.read():
                return
        end = abs(self.params.n_hours) * 60
        reach = float(np.abs(particles["time"].to_numpy()).max())
        if reach >= end - max(self.params.outdt, 0):
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"Meteorology ends early: the particles stop {reach / 60:g} h "
                f"into a {end / 60:g} h run.\n"
            )
        raise HYSPLITFailureError(FailureReason.MET_TRUNCATED, self.log_path)

    def _read_particles(self) -> pd.DataFrame:
        """Read ``PARTICLE_STILT.DAT``."""
        particle_path = self.particle_stilt_path
        if not particle_path.exists():
            raise NoParticleOutputError(
                f"{particle_path.name} not produced for {self.directory}"
            )

        return _read_particle_dat(particle_path, self.params.varsiwant)

    def _write_setup(self) -> None:
        """Write ``SETUP.CFG``."""
        entries = self.params.setup_entries()
        entries["kmsl"] = self._resolved_kmsl()
        entries["ivmax"] = len(self.params.varsiwant)  # number of output variables
        entries["winderrtf"] = self.params.winderrtf

        nl = NameList("SETUP")
        nl.update(entries)
        nl.write(self.setup_path)

    def _resolved_kmsl(self) -> int:
        """Return ``KMSL`` for this receptor, raising if ``params.kmsl`` disagrees."""
        receptor_kmsl = kmsl_from_vertical_reference(self.receptor.altitude_ref)
        if self.params.kmsl is None:
            return receptor_kmsl
        if self.params.kmsl != receptor_kmsl:
            raise ValueError(
                "TransportParams.kmsl conflicts with receptor altitude_ref: "
                f"kmsl={self.params.kmsl}, altitude_ref={self.receptor.altitude_ref!r}."
            )
        return self.params.kmsl

    def _write_winderr(self) -> None:
        """Write ``WINDERR`` when wind perturbations are enabled, else remove it."""
        _write_values(self.winderr_path, self.params.winderr)

    def _write_zierr(self) -> None:
        """Write ``ZIERR`` when mixed-layer perturbations are enabled, else remove it."""
        _write_values(self.zierr_path, self.params.zierr)

    def _write_zicontrol(self) -> None:
        """Write ``ZICONTROL`` when ``ziscale`` scales the mixed layer, else remove it."""
        values = self.params.ziscale_factors
        if values is None:
            self.zicontrol_path.unlink(missing_ok=True)
            return
        text = "\n".join([str(len(values)), *(str(v) for v in values)]) + "\n"
        self.zicontrol_path.write_text(text)
