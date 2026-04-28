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
    build_setup_entries,
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
from stilt.receptor import Receptor
from stilt.storage import resolve_directory

CONTROL_FILE = "CONTROL"
SETUP_FILE = "SETUP.CFG"
HYCS_STD_FILE = "hycs_std"
PARTICLE_STILT_FILE = "PARTICLE_STILT.DAT"
PARTICLE_FILE = "PARTICLE.DAT"
WINDERR_FILE = "WINDERR"
ZIERR_FILE = "ZIERR"
ZICONTROL_FILE = "ZICONTROL"


# Mapping from (system, machine) → bundled subdirectory inside stilt/hysplit/bin/
_PLATFORM_MAP: dict[tuple[str, str], str] = {
    ("Linux", "x86_64"): "linux_x64",
    ("Darwin", "x86_64"): "macos_x64",
}


def _bundled_exe_dir() -> Path:
    """Return the bundled binary directory for the current platform."""
    key = (platform.system(), platform.machine())
    subdir = _PLATFORM_MAP.get(key)
    if subdir is None:
        raise RuntimeError(
            f"No bundled HYSPLIT binary for {platform.system()} {platform.machine()}. "
            "Build hycs_std from source and place it in a directory on your PATH."
        )
    return Path(str(pkg_files("stilt.hysplit") / "bin" / subdir))


def _bundled_data_dir() -> Path:
    """Return the bundled HYSPLIT data files directory."""
    return Path(str(pkg_files("stilt.hysplit") / "data"))


def _read_particle_dat(path: Path, names: Sequence[str]) -> pd.DataFrame:
    """Read one whitespace-delimited HYSPLIT particle output file.

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
    """Raw output from a single HYSPLIT execution.

    Attributes
    ----------
    particles : pd.DataFrame
        Main particle positions and footprint columns from the PARDUMP file.
    error_particles : pd.DataFrame or None
        Error-trajectory particle data, or ``None`` if no error run was performed.
    log_path : Path
        Combined log path containing streamed standard output from the run(s).
    """

    particles: pd.DataFrame
    error_particles: pd.DataFrame | None
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
        self.exe_dir = Path(exe_dir) if exe_dir is not None else _bundled_exe_dir()
        self.data_dir = Path(data_dir) if data_dir is not None else _bundled_data_dir()

    def prepare(self) -> None:
        """Create sim directory, symlink executables and data files, write CONTROL and SETUP.CFG."""
        self.directory.mkdir(parents=True, exist_ok=True)

        # Symlink binary from exe_dir, data files from data_dir (mirrors R-STILT)
        for src_dir in [self.exe_dir, self.data_dir]:
            for f in src_dir.iterdir():
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

    def execute(self, timeout: int | None, rm_dat: bool) -> HYSPLITResult:
        """
        Run HYSPLIT, optionally followed by an error trajectory run.

        The error trajectory run only proceeds after main particles are
        successfully parsed - WINDERR/ZIERR are never written otherwise.

        Parameters
        ----------
        timeout : int or None
            Wall-time limit in seconds for each HYSPLIT call. ``None`` disables
            the timeout.
        rm_dat : bool
            If ``True``, delete the raw ``PARTICLE.DAT`` file after parsing to
            save disk space.

        Returns
        -------
        HYSPLITResult
            Parsed particles, optional error particles, and streamed log path.
        """
        self._run(timeout, label="main")

        particles = self._read_particles(rm_dat)

        # --- Error trajectory (only if main succeeded) ---
        error_particles = None
        if self.params.winderrtf > 0:
            self._write_winderr()
            self._write_zierr()
            self._write_setup(winderrtf=self.params.winderrtf)
            self.particle_stilt_path.unlink(missing_ok=True)
            self.particle_path.unlink(missing_ok=True)

            try:
                self._run(timeout, label="error")
            except (HYSPLITTimeoutError, HYSPLITFailureError) as e:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n=== error run failed ===\n{e}\n")

            if self.particle_stilt_path.exists():
                error_particles = self._read_particles(rm_dat)

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
        output = self._read_log_segment(segment_start)
        for phrase, reason in FAILURE_PHRASES.items():
            if phrase in output:
                raise HYSPLITFailureError(reason, str(self.directory))

    def _terminate_process(self, proc: subprocess.Popen[Any]) -> None:
        """Terminate one HYSPLIT process group with a bounded escalation path."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            finally:
                proc.wait(timeout=5)

    def _read_log_segment(self, start: int) -> str:
        """Read one appended segment from the combined log file."""
        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start)
            return handle.read()

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

    def _write_setup(self, winderrtf: int) -> None:
        """Write ``SETUP.CFG`` for the current HYSPLIT run."""
        entries = build_setup_entries(self.params)
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
                "\n".join(str(v) for v in params.values()) + "\n"
            )

    def _write_zierr(self) -> None:
        """Write ``ZIERR`` when mixed-layer perturbations are enabled."""
        params = self.params._zierr_params()
        if all(v is not None for v in params.values()):
            self.zierr_path.write_text(
                "\n".join(str(v) for v in params.values()) + "\n"
            )

    def _ziscale_values(self) -> list[float] | None:
        """Return the ZICONTROL scale vector for this run, if enabled."""
        if not self.params.zicontroltf:
            return None

        raw = self.params.ziscale
        n_hours = max(abs(int(self.params.n_hours)), 1)

        if isinstance(raw, int | float):
            return [float(raw)] * n_hours

        if not raw:
            raise ValueError("ziscale cannot be empty when zicontroltf is enabled.")

        first = raw[0]
        if isinstance(first, list):
            if len(raw) != 1:
                raise ValueError(
                    "Per-simulation ziscale lists are not supported in PYSTILT's "
                    "flat config yet. Pass one shared vector for all simulations."
                )
            values = [float(v) for v in first]
        else:
            values = [float(v) for v in raw]  # type: ignore[arg-type]

        if not values:
            raise ValueError("ziscale cannot be empty when zicontroltf is enabled.")
        return values

    def _write_zicontrol(self) -> None:
        """Write ZICONTROL when mixed-layer scaling is enabled."""
        values = self._ziscale_values()
        if values is None:
            self.zicontrol_path.unlink(missing_ok=True)
            return
        text = "\n".join([str(len(values)), *(str(v) for v in values)]) + "\n"
        self.zicontrol_path.write_text(text)
