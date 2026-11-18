"""
A simulation, one release run under one variant, and where its results are.

This is the particle part, which the workers and the stages use.
:class:`stilt.Simulation` (:mod:`stilt.stilt_simulation`), which a STILT
project gives you, adds the reads of what STILT makes from the particles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from stilt.output import Output, completed
from stilt.particles import read_particles
from stilt.releases import Release
from stilt.stages import reads

if TYPE_CHECKING:
    from stilt._paths import Location
    from stilt.config import Variant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Simulation:
    """
    One release run under one variant, as one realization of an ensemble or alone.

    A simulation is a value: its release, its variant, and the output
    directory its results are in. From those it knows where its particles,
    the file of each stage (:meth:`path`), and its log are, whether they
    exist (:meth:`has`, :attr:`is_complete`), and how to read them. It runs
    nothing itself; the workers in :mod:`stilt.execution` run the transport
    model and the stages.

    :class:`stilt.Simulation` subclasses it with the reads of what STILT
    makes from the particles: the footprint, the background, and the
    transport error. You rarely build one yourself. Get it from a project instead, as in
    ``project.simulation(release_id, "hrrr")``.

    Parameters
    ----------
    release : Release
        Where and when particles are released.
    variant : Variant
        The variant, resolved: its configs, and the hashes that find its
        particles and footprint folders.
    output : Output
        The output directory.
    directory : Path, optional
        The project directory, where a relative file name in the settings
        starts, such as an averaging-kernel table. ``None`` starts it from
        the working directory.
    realization : int, optional
        Which realization of an ensemble variant (``realizations: N``),
        ``0`` to ``N - 1``. ``None`` for a variant that runs once.

    Raises
    ------
    ValueError
        If *realization* is not one the variant runs as.
    """

    release: Release
    variant: Variant
    output: Output
    directory: Path | None = None
    realization: int | None = None

    def __post_init__(self) -> None:
        if self.realization not in self.variant.realization_numbers:
            raise ValueError(
                f"Variant {self.variant.name!r} runs as realizations "
                f"{self.variant.realization_numbers}, not {self.realization!r}."
            )

    def __repr__(self) -> str:
        realization = (
            "" if self.realization is None else f", realization={self.realization}"
        )
        return (
            f"Simulation(release={self.release.id!r}, "
            f"variant={self.variant.name!r}{realization})"
        )

    def __str__(self) -> str:
        name = f"{self.release.id}/{self.variant.name}"
        return name if self.realization is None else f"{name}/{self.realization}"

    def __eq__(self, other: object) -> bool:
        # One simulation whatever its class: the workers build this class and
        # stilt.Project the subclass, which adds only reads.
        if not isinstance(other, Simulation):
            return NotImplemented
        mine = tuple(getattr(self, f.name) for f in fields(self))
        return mine == tuple(getattr(other, f.name) for f in fields(other))

    def __hash__(self) -> int:
        return hash((self.id, self.output))

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> tuple[str, str, int | None]:
        """``(release.id, variant.name, realization)``, the row of ``project.simulations`` it is."""
        return (self.release.id, self.variant.name, self.realization)

    @property
    def transport(self) -> Any:
        """The transport config this simulation runs with. It is the variant's, with ``seed + realization`` for an ensemble."""
        return self.variant.transport_for(self.realization)

    # -- where the results are ---------------------------------------------

    def path(self, kind: str) -> Location | None:
        """
        Return the path of this simulation's file of *kind* in the output directory.

        *kind* is ``"particles"`` or the name of a stage, such as
        ``"footprint"``. ``None`` before the variant's folder of that kind
        exists, and for a stage the variant does not use. The file may not
        exist yet (:meth:`has`). A stage that reads another stage's file
        finds it here, and reads it with that stage's own reader.

        Examples
        --------
        >>> sim.path("footprint")  # doctest: +SKIP
        """
        return self.output.path(kind, self.variant, self.release.id, self.realization)

    def has(self, kind: str) -> bool:
        """
        Return whether this simulation's file of *kind* exists.

        *kind* is ``"particles"`` or the name of a stage. A stage's file is
        a finished result whatever it holds: an empty footprint is one.
        """
        path = self.path(kind)
        return path is not None and path.exists()

    @property
    def particles_path(self) -> Location | None:
        """Path of the particle file in the output directory, or ``None`` before its folder exists."""
        return self.path("particles")

    @property
    def log_path(self) -> Location | None:
        """Path of the transport model's log in the output directory, or ``None`` before its folder exists."""
        return self.output.log_path(self.variant, self.release.id, self.realization)

    @property
    def kept_workdir(self) -> Location | None:
        """
        Where a failed run's working directory is kept, or ``None`` before its folder exists.

        The directory, under ``scratch/`` in the output directory, holds
        the transport model's files as the run left them: for HYSPLIT,
        CONTROL, SETUP.CFG, and its own output. It exists only after a
        failed run, or after any run with ``keep_workdir`` set.
        """
        return self.output.kept_workdir(self.variant, self.release.id, self.realization)

    # -- presence and completion -------------------------------------------

    @property
    def has_particles(self) -> bool:
        """Whether the particle file exists."""
        return self.has("particles")

    @property
    def is_complete(self) -> bool:
        """
        Whether every expected result exists.

        That is the particles, and the file of every stage the variant
        uses, such as the footprint. It is :func:`stilt.output.completed`,
        the one definition of done, for one release, from a file check for
        each.
        """
        rid = self.release.id
        mine: frozenset[str] = frozenset([rid])
        none: frozenset[str] = frozenset()
        kinds = ("particles", *self.variant.stage_order)
        return rid in completed(*(mine if self.has(k) else none for k in kinds))

    # -- status ------------------------------------------------------------

    @property
    def failure(self) -> dict[str, Any] | None:
        """
        Why this simulation is not complete, as the worker recorded it, or ``None``.

        ``None`` when it is complete, or has not run, or what failed has
        since succeeded. Otherwise a dict with ``stage`` (``"particles"``
        for the transport model run, or the name of the stage that failed,
        such as ``"footprint"``), ``reason`` (a short cause such as
        ``"MET_COVERAGE"``, or the error's class when it has none),
        ``message``, and ``time``. An unexpected error also has a
        ``traceback``. A failed transport model run fails every variant
        that shares its particles. Of several stages that failed, the
        first in the order they run is given. :attr:`log` reads the model's log and
        :attr:`kept_workdir` is where its workdir was kept.

        Examples
        --------
        >>> sim.failure  # doctest: +SKIP
        {'stage': 'particles', 'reason': 'MET_COVERAGE',
         'message': 'HYSPLIT: start point not within (x,y,t) any data file', ...}
        """
        variant, rid = self.variant, self.release.id
        if not self.has("particles"):
            return self.output.failure("particles", variant, rid, self.realization)
        # A stage can have failed only where what it reads exists: the first
        # such stage with a record, in the order the stages run.
        there = {"particles"}
        for name in variant.stage_order:
            if self.has(name):
                there.add(name)
            elif there.issuperset(reads(variant.stage(name))):
                record = self.output.failure(name, variant, rid, self.realization)
                if record is not None:
                    return record
        return None

    # -- reading the results -----------------------------------------------

    @property
    def log(self) -> str:
        """
        Text of the transport model's log.

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            raise FileNotFoundError(f"No log for {self} yet.")
        return log_path.read_text()

    @cached_property
    def particles(self) -> pd.DataFrame:
        """
        The particle table, one row per particle per output step.

        The columns are the variables in ``varsiwant``, under the particle
        table's names (``particle``, ``age`` in minutes since release,
        ``lon``, ``lat``, ``zagl``, ``foot``, ...), plus ``time`` (UTC)
        and ``release_height`` (for column and multipoint releases). Read from the output directory on first
        access and kept. Check
        :attr:`has_particles` first when the run may not have finished.

        Raises
        ------
        FileNotFoundError
            If the particles have not been written yet. Nothing is kept, so
            a read after the run finishes loads them.
        """
        path = self.particles_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"{self} has no particles yet.")
        return read_particles(path)


__all__ = ["Simulation"]
