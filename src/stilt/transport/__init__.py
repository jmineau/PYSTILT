"""
The transport model interface.

A transport model turns one receptor and one set of transport settings into
a particle table. Each model is a subpackage here. HYSPLIT
(:class:`stilt.transport.hysplit.HysplitModel`) is the only one today. The
worker reaches it through :func:`get_model` by the name a run's settings
record, so a second model needs no change to the worker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
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


class TransportConfig(Protocol):
    """
    What PYSTILT needs from any transport model's config.

    A model's config is a pydantic model of its own parameters, which the
    model names as its ``config_class``. In ``config.yaml`` they are flat,
    top-level keys for the project's model, and a variant that names another
    model gives that model's parameters itself.

    Attributes
    ----------
    n_hours : int
        Length of each simulation, in hours. Negative runs backward in time.
    UNRECORDED : frozenset of str
        Fields that change no particle, such as where the build is, left
        out of a run's recorded settings.
    """

    n_hours: int
    UNRECORDED: ClassVar[frozenset[str]]

    def settings(self) -> dict[str, Any]:
        """Return what a run records of this config: the fields that change its particles."""
        ...

    def realizations(self, n: int) -> list[Self]:
        """Return *n* realizations of this config, for an ensemble, raising if they would repeat."""
        ...

    # Every model's config is a pydantic model; the core uses these two of its methods.

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return the config as a dict (pydantic's ``model_dump``)."""
        ...

    def model_dump_json(self, **kwargs: Any) -> str:
        """Return the config as JSON (pydantic's ``model_dump_json``)."""
        ...


class TransportModel(Protocol):
    """
    A transport model that follows particles from a receptor.

    Attributes
    ----------
    name : str
        Name recorded in a run's settings, such as ``"hysplit"``, and
        written as ``model:`` in ``config.yaml``.
    config_class : type
        The model's config (a :class:`TransportConfig`), which validates its
        parameters.
    """

    name: str
    config_class: type[Any]

    def version(self, config: Any) -> str:
        """Return the version of the build *config* would run, recorded with the run."""
        ...

    def data_files(self, config: Any) -> dict[str, str] | None:
        """
        Return the checksum of each data file *config* replaces, by file name.

        ``None`` when the run uses the model's own data files. These are part
        of the run's identity, since the data change the particles.
        """
        ...

    def run(
        self,
        receptor: Receptor,
        config: Any,
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
        config : TransportConfig
            The model's config, of its ``config_class``.
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


def _hysplit() -> TransportModel:
    """Return HYSPLIT, importing its package only when it is asked for."""
    from stilt.transport.hysplit import HysplitModel

    return HysplitModel()


#: The transport models PYSTILT can run, by the name ``model:`` takes in
#: ``config.yaml``. A port of another model adds its entry here.
MODELS: dict[str, Callable[[], TransportModel]] = {"hysplit": _hysplit}


def get_model(name: str = "hysplit") -> TransportModel:
    """
    Return the transport model called *name*.

    Raises
    ------
    ValueError
        If no model has that name.
    """
    factory = MODELS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown transport model {name!r}. The models are {sorted(MODELS)}."
        )
    return factory()


__all__ = [
    "MODELS",
    "ModelInfo",
    "ModelRun",
    "TransportConfig",
    "TransportModel",
    "get_model",
]
