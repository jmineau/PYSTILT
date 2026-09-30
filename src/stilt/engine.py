"""
The transport engine boundary.

A transport engine turns one receptor and one set of transport settings into
a particle table. HYSPLIT (:class:`stilt.hysplit.HysplitEngine`) is the only
engine today. The worker reaches it through :func:`get_engine` by the name a
run's settings record, so a second engine needs no change to the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from stilt.config import STILTParams
    from stilt.meteorology import Met
    from stilt.receptors import Receptor


@dataclass(frozen=True)
class EngineRun:
    """
    What one engine run produced.

    Attributes
    ----------
    particles : pandas.DataFrame
        The particle table, one row per particle per time step.
    met_files : list of Path
        The meteorology files the run read, for the record kept with the
        particles.
    """

    particles: pd.DataFrame
    met_files: list[Path]


class TransportEngine(Protocol):
    """
    A transport model that follows particles from a receptor.

    Attributes
    ----------
    name : str
        Name recorded in a run's settings, such as ``"hysplit"``.
    """

    name: str

    def version(self, params: STILTParams) -> str:
        """Return the version of the build *params* would run, recorded with the run."""
        ...

    def run(
        self, receptor: Receptor, params: STILTParams, met: Met, workdir: Path
    ) -> EngineRun:
        """
        Run one receptor and return its particles.

        Parameters
        ----------
        receptor : Receptor
            Where and when particles are released.
        params : STILTParams
            Transport settings.
        met : Met
            The meteorology to read.
        workdir : Path
            Scratch directory for the run's files. It exists and is empty.
            The caller discards it afterwards and keeps ``stilt.log`` from it
            when the engine writes one.
        """
        ...


def get_engine(name: str = "hysplit") -> TransportEngine:
    """
    Return the transport engine called *name*.

    Raises
    ------
    ValueError
        If no engine has that name.
    """
    if name == "hysplit":
        from stilt.hysplit import HysplitEngine

        return HysplitEngine()
    raise ValueError(f"Unknown transport engine {name!r}. The engines are ['hysplit'].")


__all__ = ["EngineRun", "TransportEngine", "get_engine"]
