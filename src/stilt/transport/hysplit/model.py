"""HYSPLIT as a transport model."""

from __future__ import annotations

import hashlib
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from stilt.particles import calc_plume_dilution
from stilt.receptors import PointReceptor
from stilt.transport import ModelRun

from .config import HysplitConfig
from .driver import HYSPLITDriver, _bundled_data_dir
from .release import add_release_heights

if TYPE_CHECKING:
    from stilt.meteorology import Met
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


def finish_particles(
    particles: pd.DataFrame, receptor: Receptor, config: HysplitConfig
) -> pd.DataFrame:
    """
    Return HYSPLIT's particles with what the config asks added.

    Each particle gets its release height (``xhgt``) for a column or
    multipoint receptor, and ``foot`` is corrected for plume dilution near
    the receptor when ``hnf_plume`` is set
    (:func:`stilt.particles.calc_plume_dilution`).
    """
    particles = add_release_heights(particles, receptor)
    if config.hnf_plume:
        r_zagl = receptor.altitude if isinstance(receptor, PointReceptor) else None
        particles = calc_plume_dilution(particles, r_zagl, config.veght)
    return particles


class HysplitModel:
    """
    HYSPLIT as a transport model, the object ``get_model("hysplit")`` returns.

    :meth:`run` picks the met files a receptor needs and runs HYSPLIT
    through a :class:`~stilt.transport.hysplit.HYSPLITDriver`, which writes the input
    files, runs ``hycs_std``, and reads the particles. HYSPLIT reads the met
    where it is (the cropped copies when the met is cropped), so nothing is
    staged per run.
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
        met: Met,
        workdir: Path,
        timeout: int | None = None,
    ) -> ModelRun:
        """Write the input files, run ``hycs_std``, and return the finished particles (:func:`finish_particles`)."""
        source = met.required_files(r_time=receptor.time, n_hours=config.n_hours)
        driver = HYSPLITDriver(
            directory=workdir,
            receptor=receptor,
            params=config,
            met_files=met.readable(source),
        )
        driver.prepare()
        particles = driver.execute(timeout=timeout)
        # The record names the source files; the crop settings are in the
        # run's settings.
        return ModelRun(
            particles=finish_particles(particles, receptor, config), met_files=source
        )


__all__ = ["HysplitModel", "finish_particles", "hysplit_version"]
