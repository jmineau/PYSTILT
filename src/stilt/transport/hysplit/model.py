"""HYSPLIT as a transport model."""

from __future__ import annotations

import hashlib
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.transport import ModelRun

from .driver import HYSPLITDriver, _bundled_data_dir
from .release import add_release_heights

if TYPE_CHECKING:
    from stilt.config import TransportParams
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

    def version(self, params: TransportParams) -> str:
        """Return the version of the bundled build, or of the one in ``params.exe_dir``."""
        return hysplit_version(params.exe_dir)

    def data_files(self, params: TransportParams) -> dict[str, str] | None:
        """
        Return the SHA-256 of each table in ``params.data_dir`` that differs from the bundled one.

        ``None`` when there is no ``data_dir`` or every table in it matches
        the bundled table of the same name.
        """
        if params.data_dir is None:
            return None
        bundled = _bundled_data_dir()
        changed = {}
        for path in sorted(Path(params.data_dir).iterdir()):
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
        params: TransportParams,
        met: Met,
        workdir: Path,
        timeout: int | None = None,
    ) -> ModelRun:
        """Write the input files, run ``hycs_std``, and return the particles."""
        source = met.required_files(r_time=receptor.time, n_hours=params.n_hours)
        driver = HYSPLITDriver(
            directory=workdir,
            receptor=receptor,
            params=params,
            met_files=met.readable(source),
        )
        driver.prepare()
        particles = driver.execute(timeout=timeout)
        # The record names the source files; the crop settings are in the
        # run's settings.
        return ModelRun(
            particles=add_release_heights(particles, receptor),
            met_files=source,
        )


__all__ = ["HysplitModel", "hysplit_version"]
