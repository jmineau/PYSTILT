"""HYSPLIT as a transport engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from stilt.config.transport import hysplit_version
from stilt.engine import EngineRun

from .driver import HYSPLITDriver

if TYPE_CHECKING:
    from stilt.config import STILTParams
    from stilt.meteorology import Met
    from stilt.receptors import Receptor


class HysplitEngine:
    """
    Run HYSPLIT's ``hycs_std`` for one receptor.

    The run's ``CONTROL``, ``SETUP.CFG``, and particle file live in the
    working directory for the duration of the run. HYSPLIT reads the
    meteorology where it is (cropped copies when the met is cropped), so
    nothing is staged per run.
    """

    name = "hysplit"

    def version(self, params: STILTParams) -> str:
        """Return the version of the bundled build, or of the one in ``params.exe_dir``."""
        return hysplit_version(params.exe_dir)

    def run(
        self, receptor: Receptor, params: STILTParams, met: Met, workdir: Path
    ) -> EngineRun:
        """Write the input files, run ``hycs_std``, and return the particles."""
        source = met.required_files(r_time=receptor.time, n_hours=params.n_hours)
        driver = HYSPLITDriver(
            directory=workdir,
            receptor=receptor,
            params=params,
            met_files=met.readable(source),
        )
        driver.prepare()
        result = driver.execute(timeout=params.timeout, rm_dat=params.rm_dat)
        # The record names the source files; the crop settings are in the
        # run's settings.
        return EngineRun(particles=result.particles, met_files=source)


__all__ = ["HysplitEngine"]
