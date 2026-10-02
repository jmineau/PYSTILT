"""HYSPLIT as a transport model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from stilt.config.transport import hysplit_version
from stilt.transport import ModelRun

from .driver import HYSPLITDriver

if TYPE_CHECKING:
    from stilt.config import TransportParams
    from stilt.meteorology import Met
    from stilt.receptors import Receptor


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
        result = driver.execute(timeout=timeout)
        # The record names the source files; the crop settings are in the
        # run's settings.
        return ModelRun(particles=result.particles, met_files=source)


__all__ = ["HysplitModel"]
