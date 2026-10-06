"""
The transport model interface.

A transport model turns one receptor and one set of transport settings into
a particle table. Each model is a subpackage here. HYSPLIT
(:class:`stilt.transport.hysplit.HysplitModel`) is the only one today. The
worker reaches it through :func:`get_model` by the name a run's settings
record, so a second model needs no change to the worker.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stilt.meteorology import Met, MetConfig

if TYPE_CHECKING:
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


class TransportConfig(BaseModel):
    """
    The base of every transport model's config.

    A model's config subclasses it with its own parameters and is the
    model's ``config_class``. In ``config.yaml`` the parameters are flat,
    top-level keys for the project's model, and a variant that names another
    model gives that model's parameters itself. A subclass overrides
    :meth:`settings` or :meth:`realizations` only where its model differs.
    """

    model_config = ConfigDict(extra="forbid")

    #: Fields that change no particle, such as where the build is, left out
    #: of a run's recorded settings.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset()

    n_hours: int = Field(
        -24,
        description="Length of each simulation, in hours. Negative runs backward in time.",
    )
    seed: int | None = Field(
        None,
        description=(
            "Seed of the model's random numbers, for a reproducible run. "
            "Realization ``k`` of a variant runs with ``seed + k``."
        ),
    )

    def settings(self) -> dict[str, Any]:
        """Return what a run records of this config: every field but the :attr:`UNRECORDED` ones."""
        return self.model_dump(mode="json", exclude=set(self.UNRECORDED))

    def realizations(self, n: int) -> list[Self]:
        """
        Return *n* realizations of this config, realization ``k`` with ``seed + k``.

        Realization 0 uses the configured seed. Without a seed every
        realization is the same config, and the model must draw its own
        random numbers for them to differ.
        """
        seed = self.seed
        return [
            self.model_copy(update={"seed": None if seed is None else seed + k})
            for k in range(n)
        ]


class TransportModel(Protocol):
    """
    A transport model that follows particles from a receptor.

    Attributes
    ----------
    name : str
        Name recorded in a run's settings, such as ``"hysplit"``, and
        written as ``model:`` in ``config.yaml``.
    """

    name: str

    @property
    def config_class(self) -> type[TransportConfig]:
        """The model's config, a :class:`TransportConfig`, which validates its parameters."""
        ...

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


def run_trajectories(
    receptor: Receptor,
    met: MetConfig | Mapping[str, Any],
    *,
    model: str = "hysplit",
    workdir: str | Path | None = None,
    timeout: int | None = None,
    **params: Any,
) -> pd.DataFrame:
    """
    Run the transport model for one receptor and return its particles.

    This is what a project runs for each receptor, without a project: the
    particles are returned, not stored. STILT-R calls it
    ``calc_trajectory``.

    Parameters
    ----------
    receptor : Receptor
        Where and when particles are released.
    met : MetConfig or dict
        The meteorology, as one entry under ``mets:`` in ``config.yaml``
        (:class:`stilt.MetConfig`). A relative ``directory`` starts from
        the working directory.
    model : str, default "hysplit"
        The transport model.
    workdir : str or Path, optional
        Directory the model runs in, kept afterwards for a look at its
        files. It must be empty or not exist yet. Without it the model runs
        in a temporary directory that is removed.
    timeout : int, optional
        Time limit in seconds. ``None`` waits indefinitely.
    **params
        The transport model's parameters, such as ``n_hours`` and
        ``numpar`` (:class:`stilt.transport.hysplit.HysplitConfig`).

    Returns
    -------
    pandas.DataFrame
        The particle table, one row per particle per output step.

    Raises
    ------
    ValueError
        If *workdir* is not empty, or a parameter is not one of the
        model's.
    SimulationError
        If the run fails; ``reason`` says why.
    MeteorologyError
        If the met files the run needs are missing.

    Examples
    --------
    >>> receptor = stilt.PointReceptor(
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
    >>> particles = stilt.run_trajectories(receptor, met, n_hours=-24, numpar=200)
    """
    transport = get_model(model)
    config = transport.config_class(**params)
    meteorology = Met("met", MetConfig.model_validate(met))
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="stilt-") as tmp:
            run = transport.run(receptor, config, meteorology, Path(tmp), timeout)
            return run.particles
    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError(
            f"{workdir} is not empty. The model would read files left from "
            "another run; give an empty or new directory."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    return transport.run(receptor, config, meteorology, workdir, timeout).particles


__all__ = [
    "MODELS",
    "ModelInfo",
    "ModelRun",
    "TransportConfig",
    "TransportModel",
    "get_model",
    "run_trajectories",
]
