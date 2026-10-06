"""HYSPLIT as a transport model."""

from __future__ import annotations

import datetime as dt
import hashlib
import tempfile
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.exceptions import SimulationError
from stilt.meteorology import Met, MetConfig
from stilt.transport import ModelRun

from .config import HysplitConfig
from .driver import (
    LOG_FILE,
    PARTICLE_STILT_FILE,
    _bundled_data_dir,
    _check_met_reached_end,
    _run_hycs_std,
    read_particle_dat,
    write_inputs,
)
from .failures import FailureReason

if TYPE_CHECKING:
    from stilt.receptors import Receptor


def hysplit_version(exe_dir: str | Path | None = None) -> str:
    """
    Return the version of the HYSPLIT build in *exe_dir*, or of the bundled build.

    A build directory names its version in a ``version`` file beside
    ``hycs_std``, as the bundled one does. Two builds with the same settings
    can give different particles, so the version is part of a run's
    identity.

    Raises
    ------
    FileNotFoundError
        If *exe_dir* has no ``version`` file.
    """
    if exe_dir is None:
        path = Path(str(pkg_files("stilt.transport.hysplit") / "bin" / "version"))
    else:
        path = Path(exe_dir) / "version"
        if not path.exists():
            raise FileNotFoundError(
                f"{exe_dir} has no 'version' file. A custom hycs_std build needs "
                "one beside the binary, holding its version string (such as "
                "v5.3.2+t0-rows), so its runs are told apart from other builds'."
            )
    return path.read_text().strip()


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HysplitModel:
    """
    HYSPLIT as a transport model, the object ``get_model("hysplit")`` returns.

    :meth:`run` picks the met files a receptor needs, writes HYSPLIT's input
    files (:func:`~stilt.transport.hysplit.write_inputs`), runs ``hycs_std``,
    and reads the particles it writes
    (:func:`~stilt.transport.hysplit.read_particle_dat`). HYSPLIT reads the
    met where it is (the cropped copies when the met is cropped), so nothing
    is staged per run.
    """

    name = "hysplit"
    config_class = HysplitConfig

    def version(self, config: HysplitConfig) -> str:
        """Return the version of the bundled build, or of the one in ``config.exe_dir``."""
        return hysplit_version(config.exe_dir)

    def data_files(self, config: HysplitConfig) -> dict[str, str] | None:
        """
        Return the SHA-256 of each table in ``config.data_dir`` that differs from the bundled one.

        ``None`` when there is no ``data_dir`` or every table in it matches
        the bundled table of the same name.
        """
        if config.data_dir is None:
            return None
        bundled = _bundled_data_dir()
        changed = {}
        for path in sorted(Path(config.data_dir).iterdir()):
            if not path.is_file():
                continue
            digest = _sha256(path)
            own = bundled / path.name
            if not own.is_file() or _sha256(own) != digest:
                changed[path.name] = digest
        return changed or None

    def run(
        self,
        receptor: Receptor,
        config: HysplitConfig,
        met: MetConfig,
        window: tuple[dt.datetime, dt.datetime],
        workdir: Path | None = None,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Run HYSPLIT for one receptor in *workdir* and return its particles.

        HYSPLIT finds the met files that cover *window* (and, for a backward
        run, the hour after the release, which it interpolates), cropped
        when the met is. Its output is ``stilt.log`` in *workdir*, returned
        as the run's log. Without *workdir* it runs in a temporary
        directory. The release heights and the near-field correction are
        added by the caller, as for any model
        (:func:`stilt.transport.run_model`).

        Raises
        ------
        SimulationError
            With HYSPLIT's log in ``log``, and ``reason`` ``TIMEOUT`` when
            the run exceeded *timeout* seconds; the :class:`FailureReason`
            of a known failure message in the log; ``MET_TRUNCATED`` when a
            met file was cut short and the particles stop before the end of
            the run; or ``NO_PARTICLE_DATA`` when HYSPLIT wrote no particle
            file.
        """
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix="stilt-") as tmp:
                return self.run(receptor, config, met, window, Path(tmp), timeout)
        files = Met("met", met)
        source = files.files_for(window, hour_after=config.n_hours < 0)
        write_inputs(workdir, receptor, config, files.readable(source))
        try:
            _run_hycs_std(workdir, timeout)
            path = workdir / PARTICLE_STILT_FILE
            if not path.exists():
                raise SimulationError(
                    f"HYSPLIT wrote no {PARTICLE_STILT_FILE}.",
                    reason=FailureReason.NO_PARTICLE_DATA,
                )
            particles = read_particle_dat(path, config.varsiwant)
            _check_met_reached_end(particles, workdir / LOG_FILE, config)
        except SimulationError as error:
            error.log = _read_log(workdir)
            raise
        # The record names the source files; the crop settings are in the
        # run's settings.
        return ModelRun(particles=particles, log=_read_log(workdir), met_files=source)


def _read_log(workdir: Path) -> str:
    """Return HYSPLIT's log in *workdir*, or an empty string when it wrote none."""
    path = workdir / LOG_FILE
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


__all__ = ["HysplitModel", "hysplit_version"]
