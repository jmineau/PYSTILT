"""
The transport model interface.

A transport model turns one receptor and one set of transport settings into
a particle table. Each model is a subpackage here. HYSPLIT
(:class:`stilt.transport.hysplit.HysplitModel`) is the only one today. The
worker reaches it through :func:`get_model` by the name a run's settings
record, so a second model needs no change to the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stilt.config import TransportParams
    from stilt.meteorology import Met
    from stilt.receptors import Receptor


@dataclass(frozen=True)
class ModelRun:
    """
    What one model run produced.

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


class ModelInfo(BaseModel):
    """The transport model build a run was made with, recorded in the run's settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="hysplit", description="Model name.")
    version: str = Field(
        ..., description="Version string of the build, such as ``v5.1.0``."
    )
    data_files: dict[str, str] | None = Field(
        None,
        description=(
            "SHA-256 of each data file the run used in place of the model's "
            "own, by file name. ``None`` when it used the model's own."
        ),
    )


class TransportModel(Protocol):
    """
    A transport model that follows particles from a receptor.

    Attributes
    ----------
    name : str
        Name recorded in a run's settings, such as ``"hysplit"``.
    """

    name: str

    def version(self, params: TransportParams) -> str:
        """Return the version of the build *params* would run, recorded with the run."""
        ...

    def data_files(self, params: TransportParams) -> dict[str, str] | None:
        """
        Return the checksum of each data file *params* replaces, by file name.

        ``None`` when the run uses the model's own data files. These are part
        of the run's identity, since the data change the particles.
        """
        ...

    def run(
        self,
        receptor: Receptor,
        params: TransportParams,
        met: Met,
        workdir: Path,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Run one receptor and return its particles.

        Parameters
        ----------
        receptor : Receptor
            Where and when particles are released.
        params : TransportParams
            Transport settings.
        met : Met
            The meteorology to read.
        workdir : Path
            Scratch directory for the run's files. It exists and is empty.
            The caller discards it afterwards and keeps ``stilt.log`` from it
            when the model writes one.
        timeout : int, optional
            Time limit in seconds. ``None`` waits indefinitely.
        """
        ...


def get_model(name: str = "hysplit") -> TransportModel:
    """
    Return the transport model called *name*.

    Raises
    ------
    ValueError
        If no model has that name.
    """
    if name == "hysplit":
        from stilt.transport.hysplit import HysplitModel

        return HysplitModel()
    raise ValueError(f"Unknown transport model {name!r}. The models are ['hysplit'].")


__all__ = ["ModelInfo", "ModelRun", "TransportModel", "get_model"]
