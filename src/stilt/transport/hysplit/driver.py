"""Set up and run one HYSPLIT simulation."""

import os
import platform
import signal
import subprocess
import warnings
from collections.abc import Sequence
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stilt.exceptions import (
    HYSPLITFailureError,
    HYSPLITNotFoundError,
    HYSPLITTimeoutError,
    NoParticleOutputError,
)
from stilt.receptors import Receptor
from stilt.transport.hysplit.config import HysplitConfig, fields_in
from stilt.transport.hysplit.control import ControlFile
from stilt.transport.hysplit.failures import (
    MET_TRUNCATED_WARNING,
    FailureReason,
    failure_in,
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


# -- what each input file holds -----------------------------------------------


def setup_entries(params: HysplitConfig) -> dict[str, Any]:
    """
    Return the ``SETUP.CFG`` namelist entries of *params*, leaving out unset ones.

    The driver adds ``KMSL``, ``IVMAX``, and ``WINDERRTF``, which depend on
    the receptor or follow from other settings.
    """
    entries = {
        name: getattr(params, name)
        for name in fields_in("SETUP.CFG")
        if getattr(params, name) is not None
    }
    entries["maxpar"] = params.effective_maxpar
    entries["zicontroltf"] = zicontroltf(params)
    if params.seed is not None:
        entries["seed"] = setup_seed(params.seed)
    return entries


def setup_seed(seed: int) -> int:
    """
    Return the ``SEED`` value written to ``SETUP.CFG`` for a user seed.

    HYSPLIT sets its generator state to ``-1 + SEED``. Under ``krand=2`` it
    reinitializes only from a negative state, and every state of -1 or more
    gives the same stream. Writing ``-(|seed| + 1)`` puts the state at
    ``-(|seed| + 2)``. That is negative, different for each ``|seed|``, and
    never the unseeded default (``SEED = 0``). A patched HYSPLIT that uses
    ``SEED`` directly maps a negative ``SEED`` to the same state, so the
    value works with both builds.
    """
    return -(abs(seed) + 1)


def ziscale_factors(params: HysplitConfig) -> list[float] | None:
    """
    Return the hourly mixed-layer factors for ``ZICONTROL``, or ``None`` when unscaled.

    A single ``ziscale`` value is repeated for every hour of the run and a
    list is used as given. Factors that are all 1.0 give ``None``.
    """
    values = params.hourly_ziscale()
    if all(v == 1.0 for v in values):
        return None
    return values


def zicontroltf(params: HysplitConfig) -> int:
    """Return HYSPLIT's ``ZICONTROLTF`` flag: 1 when ``ziscale`` scales the mixed layer."""
    return int(ziscale_factors(params) is not None)


def winderr(params: HysplitConfig) -> list[float] | None:
    """Return the ``WINDERR`` values, in the file's order, or ``None`` when unset."""
    values = [getattr(params, name) for name in fields_in("WINDERR")]
    return None if values[0] is None else values


def zierr(params: HysplitConfig) -> list[float] | None:
    """Return the ``ZIERR`` values, in the file's order, or ``None`` when unset."""
    values = [getattr(params, name) for name in fields_in("ZIERR")]
    return None if values[0] is None else values


def winderrtf(params: HysplitConfig) -> int:
    """Return HYSPLIT's ``WINDERRTF`` flag: 1 for wind errors, 2 for mixed-layer errors, 3 for both."""
    return (winderr(params) is not None) + 2 * (zierr(params) is not None)


def _write_values(path: Path, values: list[float] | None) -> None:
    """Write one value per line to ``path``; nothing is written when ``values`` is ``None``."""
    if values is not None:
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
    params : HysplitConfig
        Transport and error settings.
    met_files : list of Path
        Meteorology files, in the order HYSPLIT should read them.
    directory : Path
        Directory to run in. It is created if needed.
    exe_dir : Path, optional
        Directory holding ``hycs_std``. Defaults to ``params.exe_dir``, then
        to the bundled build.
    data_dir : Path, optional
        Directory of HYSPLIT data tables that replace the bundled ones.
        Defaults to ``params.data_dir``; without either, the bundled tables
        are used.
    """

    def __init__(
        self,
        receptor: Receptor,
        params: HysplitConfig,
        met_files: list[Path],
        directory: Path,
        exe_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        self.directory = Path(directory).expanduser().resolve()
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
        # explicit argument > HysplitConfig.exe_dir > binary bundled with the package
        chosen = exe_dir if exe_dir is not None else params.exe_dir
        self.exe_dir = Path(chosen) if chosen is not None else _bundled_exe_dir()
        chosen = data_dir if data_dir is not None else params.data_dir
        self.data_dir = Path(chosen) if chosen is not None else None

    def prepare(self) -> None:
        """
        Create the simulation directory and write HYSPLIT's input files.

        Links ``hycs_std`` and the data tables into the directory and writes
        ``CONTROL`` and ``SETUP.CFG``, plus ``ZICONTROL``, ``WINDERR``, and
        ``ZIERR`` when the settings call for them.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        # Symlink the binary from exe_dir and the data tables (mirrors STILT-R):
        # the bundled tables, replaced by any in data_dir. Only hycs_std is
        # taken from exe_dir: a custom build directory usually holds a whole
        # HYSPLIT exec/ tree we have no business linking.
        exe = self.exe_dir / HYCS_STD_FILE
        if not exe.is_file():
            raise HYSPLITNotFoundError(
                f"No {HYCS_STD_FILE!r} executable in {self.exe_dir}. "
                "Check HysplitConfig.exe_dir."
            )
        links = {HYCS_STD_FILE: exe}
        links.update({f.name: f for f in _bundled_data_dir().iterdir()})
        if self.data_dir is not None:  # its tables replace the bundled ones
            links.update({f.name: f for f in self.data_dir.iterdir() if f.is_file()})
        for name, target in links.items():
            (self.directory / name).symlink_to(target.resolve())

        # Write HYSPLIT CONTROL
        ControlFile(
            receptor=self.receptor,
            n_hours=self.params.n_hours,
            emisshrs=self.params.emisshrs,
            w_option=self.params.w_option,
            z_top=self.params.z_top,
            met_files=self.met_files,
        ).write(self.control_path)

        # SETUP.CFG carries winderrtf; WINDERR / ZIERR are written only when
        # perturbed. The directory is empty, so nothing stale needs removing.
        self._write_setup()
        self._write_zicontrol()
        self._write_winderr()
        self._write_zierr()

    def execute(self, timeout: int | None = None) -> pd.DataFrame:
        """
        Run HYSPLIT once and read its particle output.

        Parameters
        ----------
        timeout : int or None
            Time limit for the run, in seconds. ``None`` waits indefinitely.

        Returns
        -------
        pandas.DataFrame
            The particles read from ``PARTICLE_STILT.DAT``, one column per
            ``varsiwant`` variable. The run's log is at :attr:`log_path`.

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
        self._run(timeout)
        particles = self._read_particles()
        self._check_met_reached_end(particles)
        return particles

    # -- Private helpers -------------------------------------------------------

    def _run(self, timeout: int | None) -> None:
        """Run ``hycs_std``, writing its output to the log."""
        with (
            self.log_path.open("w", encoding="utf-8") as handle,
            subprocess.Popen(
                [str(self.hycs_std_path)],
                cwd=self.directory,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as proc,
        ):
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired as e:
                self._terminate_process(proc)
                raise HYSPLITTimeoutError(
                    f"hycs_std timed out after {timeout}s for {self.directory}"
                ) from e
        self._check_log_for_failure()

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

    def _check_log_for_failure(self) -> None:
        """Raise if the log shows a known HYSPLIT failure."""
        reason = failure_in(self.log_path.read_text(encoding="utf-8", errors="replace"))
        if reason is not None:
            raise HYSPLITFailureError(reason, self.log_path)

    def _check_met_reached_end(self, particles: pd.DataFrame) -> None:
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
        log = self.log_path.read_text(encoding="utf-8", errors="replace")
        if MET_TRUNCATED_WARNING not in log:
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
        entries = setup_entries(self.params)
        entries["kmsl"] = self._resolved_kmsl()
        entries["ivmax"] = len(self.params.varsiwant)  # number of output variables
        entries["winderrtf"] = winderrtf(self.params)

        nl = NameList("SETUP")
        nl.update(entries)
        nl.write(self.setup_path)

    def _resolved_kmsl(self) -> int:
        """Return ``KMSL`` for this receptor, raising if ``params.kmsl`` disagrees."""
        # HYSPLIT's KMSL: 0 for heights above ground, 1 above sea level.
        receptor_kmsl = 1 if self.receptor.altitude_ref == "msl" else 0
        if self.params.kmsl is None:
            return receptor_kmsl
        if self.params.kmsl != receptor_kmsl:
            raise ValueError(
                "HysplitConfig.kmsl conflicts with receptor altitude_ref: "
                f"kmsl={self.params.kmsl}, altitude_ref={self.receptor.altitude_ref!r}."
            )
        return self.params.kmsl

    def _write_winderr(self) -> None:
        """Write ``WINDERR`` when wind perturbations are enabled."""
        _write_values(self.winderr_path, winderr(self.params))

    def _write_zierr(self) -> None:
        """Write ``ZIERR`` when mixed-layer perturbations are enabled."""
        _write_values(self.zierr_path, zierr(self.params))

    def _write_zicontrol(self) -> None:
        """Write ``ZICONTROL`` when ``ziscale`` scales the mixed layer."""
        values = ziscale_factors(self.params)
        if values is None:
            return
        text = "\n".join([str(len(values)), *(str(v) for v in values)]) + "\n"
        self.zicontrol_path.write_text(text)
