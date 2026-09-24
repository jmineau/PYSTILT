"""HYSPLIT driver: binary resolution and simulation execution."""

import os
import platform
import signal
import subprocess
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
from stilt.errors import (
    FAILURE_PHRASES,
    HYSPLITFailureError,
    HYSPLITTimeoutError,
    NoParticleOutputError,
)
from stilt.hysplit.control import ControlFile
from stilt.hysplit.namelist import NameList
from stilt.project import resolve_directory
from stilt.receptors import Receptor

CONTROL_FILE = "CONTROL"
SETUP_FILE = "SETUP.CFG"
HYCS_STD_FILE = "hycs_std"
PARTICLE_STILT_FILE = "PARTICLE_STILT.DAT"
PARTICLE_FILE = "PARTICLE.DAT"
WINDERR_FILE = "WINDERR"
ZIERR_FILE = "ZIERR"
ZICONTROL_FILE = "ZICONTROL"


def _bundled_exe_dir() -> Path:
    """Return the bundled binary directory for the current platform."""
    system = platform.system()
    if system == "Linux":
        subdir = "linux_x64"
    elif system == "Darwin":
        subdir = "macos_x64"
    else:
        raise RuntimeError(
            f"No bundled HYSPLIT binary for {system}. "
            "Build hycs_std from source and place it in a directory on your PATH."
        )
    return Path(str(pkg_files("stilt.hysplit") / "bin" / subdir))


def _bundled_data_dir() -> Path:
    """Return the bundled HYSPLIT data files directory."""
    return Path(str(pkg_files("stilt.hysplit") / "data"))


def _read_particle_dat(path: Path, names: Sequence[str]) -> pd.DataFrame:
    """
    Read one whitespace-delimited HYSPLIT particle output file.

    ``numpy.loadtxt`` is substantially cheaper than regex-based pandas parsing
    for large numeric ``PARTICLE_STILT.DAT`` files while still handling
    variable-width whitespace emitted by HYSPLIT.
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
    Raw output from a single HYSPLIT execution.

    Attributes
    ----------
    particles : pd.DataFrame or None
        Main particle positions and footprint columns from the PARDUMP file.
        ``None`` when the main run was skipped (error-only execution).
    error_particles : dict[int, pd.DataFrame]
        Error-trajectory particle data keyed by realization index. Empty when
        no error run was performed; a realization whose run failed is absent.
    log_path : Path
        Combined log path containing streamed standard output from the run(s).
    """

    particles: pd.DataFrame | None
    error_particles: dict[int, pd.DataFrame]
    log_path: Path


class HYSPLITDriver:
    """Prepares and executes one HYSPLIT run in a simulation directory."""

    def __init__(
        self,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
        directory: Path | None = None,
        exe_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        self.directory = resolve_directory(directory)
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
        """Create sim directory, symlink executables and data files, write CONTROL and SETUP.CFG."""
        self.directory.mkdir(parents=True, exist_ok=True)

        # Symlink the binary from exe_dir and data files from data_dir (mirrors
        # STILT-R). Only hycs_std is taken from exe_dir: a custom build directory
        # usually holds a whole HYSPLIT exec/ tree we have no business linking.
        exe = self.exe_dir / HYCS_STD_FILE
        if not exe.is_file():
            raise FileNotFoundError(
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

        # Write SETUP.CFG with winderrtf=0 (error trajectory rewrites this later)
        self._write_setup(winderrtf=0)
        self._write_zicontrol()

    def execute(
        self,
        timeout: int | None,
        rm_dat: bool,
        *,
        error_only: bool = False,
        error_realizations: Sequence[int] = (0,),
    ) -> HYSPLITResult:
        """
        Run HYSPLIT, optionally followed by one or more error trajectory runs.

        The error trajectory runs only proceed after main particles are
        successfully parsed - WINDERR/ZIERR are never written otherwise.

        Parameters
        ----------
        timeout : int or None
            Wall-time limit in seconds for each HYSPLIT call. ``None`` disables
            the timeout.
        rm_dat : bool
            If ``True``, delete the raw ``PARTICLE.DAT`` file after parsing to
            save disk space.
        error_only : bool
            If ``True``, skip the main run and run only the (independent) error
            trajectories. Used to backfill error trajectories without
            recomputing an existing main trajectory. ``particles`` is ``None``
            in this case.
        error_realizations : sequence of int
            Which error realizations to run, one HYSPLIT call each. They are
            distinct draws under ``krand=4``, where HYSPLIT seeds each run from
            the clock, or under ``krand=2`` with a seed, where ``SETUP.CFG`` is
            rewritten with ``params.realization_seed(k)`` (``seed + k``) before
            pass ``k``;
            the config validator enforces one of the two.

        Returns
        -------
        HYSPLITResult
            Parsed particles, error particles by realization, and the log path.
        """
        particles = None
        if not error_only:
            self._run(timeout, label="main")
            particles = self._read_particles(rm_dat)

        # --- Error trajectories (independent of the main run and of each other) ---
        error_particles: dict[int, pd.DataFrame] = {}
        if self.params.winderrtf > 0:
            self._write_winderr()
            self._write_zierr()
            seeded = self.params.seed is not None
            if not seeded:
                self._write_setup(winderrtf=self.params.winderrtf)
            for k in error_realizations:
                if seeded:
                    self._write_setup(
                        winderrtf=self.params.winderrtf,
                        seed=self.params.realization_seed(k),
                    )
                self.particle_stilt_path.unlink(missing_ok=True)
                self.particle_path.unlink(missing_ok=True)

                try:
                    self._run(timeout, label=f"error[{k}]")
                except (HYSPLITTimeoutError, HYSPLITFailureError) as e:
                    with self.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"\n=== error run [{k}] failed ===\n{e}\n")

                if self.particle_stilt_path.exists():
                    error_particles[k] = self._read_particles(rm_dat)

        return HYSPLITResult(
            particles=particles,
            error_particles=error_particles,
            log_path=self.log_path,
        )

    # -- Private helpers -------------------------------------------------------

    def _run(self, timeout: int | None, *, label: str = "main") -> None:
        """Run hycs_std, streaming output to the log file."""
        if not self.hycs_std_path.exists():
            raise FileNotFoundError(
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

    def _terminate_process(self, proc: subprocess.Popen[Any]) -> None:
        """Terminate one HYSPLIT process group with SIGTERM→wait→SIGKILL escalation."""
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
        Scan the new log segment for known HYSPLIT failure phrases.

        ``start`` is a byte offset from ``stat().st_size``; ``seek()`` in text
        mode is safe for byte-aligned positions on POSIX (no multi-byte chars
        are written by hycs_std).
        """
        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start)
            for line in handle:
                for phrase, reason in FAILURE_PHRASES.items():
                    if phrase in line:
                        raise HYSPLITFailureError(reason, str(self.directory))

    def _read_particles(self, rm_dat: bool) -> pd.DataFrame:
        """Read and optionally remove ``PARTICLE_STILT.DAT`` output."""
        particle_path = self.particle_stilt_path
        if not particle_path.exists():
            raise NoParticleOutputError(
                f"{particle_path.name} not produced for {self.directory}"
            )

        particles = _read_particle_dat(particle_path, self.params.varsiwant)

        if rm_dat:
            particle_path.unlink(missing_ok=True)
            self.particle_path.unlink(missing_ok=True)
        return particles

    def _write_setup(self, winderrtf: int, seed: int | None = None) -> None:
        """
        Write ``SETUP.CFG`` for the current HYSPLIT run.

        ``seed`` overrides the configured user seed for one run (an error
        realization); it goes through the same ``STILTParams.setup_seed``
        mapping as the configured one.
        """
        entries = self.params.setup_entries()
        if seed is not None:
            entries["seed"] = self.params.setup_seed(seed)
        entries["kmsl"] = self._resolved_kmsl()
        entries["ivmax"] = len(self.params.varsiwant)  # number of output variables
        entries["winderrtf"] = winderrtf

        nl = NameList("SETUP")
        nl.update(entries)
        nl.write(self.setup_path)

    def _resolved_kmsl(self) -> int:
        """Return the effective KMSL value for this simulation."""
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
        """Write ``WINDERR`` when wind perturbations are enabled."""
        params = self.params._xyerr_params()
        if all(v is not None for v in params.values()):
            self.winderr_path.write_text(
                "\n".join(str(v) for v in params.values()) + "\n",
                encoding="utf-8",
            )

    def _write_zierr(self) -> None:
        """Write ``ZIERR`` when mixed-layer perturbations are enabled."""
        params = self.params._zierr_params()
        if all(v is not None for v in params.values()):
            self.zierr_path.write_text(
                "\n".join(str(v) for v in params.values()) + "\n",
                encoding="utf-8",
            )

    def _write_zicontrol(self) -> None:
        """Write ZICONTROL when mixed-layer scaling is enabled."""
        values = self.params.ziscale_factors
        if values is None:
            self.zicontrol_path.unlink(missing_ok=True)
            return
        text = "\n".join([str(len(values)), *(str(v) for v in values)]) + "\n"
        self.zicontrol_path.write_text(text)
