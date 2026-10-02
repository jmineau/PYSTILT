"""A simulation, one receptor run under one variant, and where its results are."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import xarray as xr

from stilt.config import FootprintConfig, VariantConfig
from stilt.exceptions import EmptyFootprint
from stilt.footprint import calculate
from stilt.output import Footprints, Output, Particles
from stilt.particles import particles_metadata
from stilt.receptors import Receptor, parse_receptor_id
from stilt.transforms import ParticleTransform, TransformContext
from stilt.transport.hysplit.failures import identify_failure_reason

if TYPE_CHECKING:
    from stilt.visualization import SimulationPlotAccessor

logger = logging.getLogger(__name__)


class SimID(NamedTuple):
    """
    Id of one simulation, a ``(receptor, variant)`` pair.

    Its string form is ``"<receptor_id>/<variant>"``.

    Examples
    --------
    >>> sid = SimID.parse("202307151800_-111.848_40.766_10/hrrr")
    >>> sid.variant
    'hrrr'
    >>> str(sid)
    '202307151800_-111.848_40.766_10/hrrr'
    """

    receptor: str
    variant: str

    def __str__(self) -> str:
        return f"{self.receptor}/{self.variant}"

    def __fspath__(self) -> str:
        """Return the string form, so ``root / sim_id`` gives a working directory."""
        return str(self)

    @classmethod
    def parse(cls, value: str | SimID | tuple[str, str]) -> SimID:
        """
        Build a :class:`SimID` from its string form or a ``(receptor, variant)`` pair.

        Raises
        ------
        ValueError
            If a string is not of the form ``"<receptor_id>/<variant>"``, or
            the receptor id is malformed.
        """
        if isinstance(value, SimID):
            return value
        if isinstance(value, tuple):
            receptor, variant = value
        else:
            receptor, sep, variant = str(value).partition("/")
            if not sep or not variant:
                raise ValueError(
                    f"Invalid sim id {value!r}; expected '{{receptor_id}}/{{variant}}'."
                )
        parse_receptor_id(receptor)  # raises on a malformed id
        return cls(str(receptor), variant)


@dataclass(frozen=True)
class Simulation:
    """
    One receptor run under one variant.

    A simulation is a value: its receptor, its variant, and the output
    directory its results are in. From those it knows where its particles,
    footprint, and log are, whether they exist (:meth:`is_complete`), and how
    to read them. It runs nothing itself; the workers in
    :mod:`stilt.execution` run HYSPLIT and write the results.

    You rarely build one yourself. Get it from a project instead, as in
    ``project.simulation(receptor_id, "hrrr")``.

    Parameters
    ----------
    receptor : Receptor
        Where and when particles are released.
    variant : VariantConfig
        The variant: its transport settings, which name the particles folder, and its
        footprint settings, if any.
    output : Output
        The output directory.
    """

    receptor: Receptor
    variant: VariantConfig
    output: Output

    def __repr__(self) -> str:
        return f"Simulation(id={str(self.id)!r})"

    def __hash__(self) -> int:
        # Transport settings are not hashable (STILTParams is mutable), so hash
        # what identifies the simulation: its id and where its results are.
        return hash((self.id, self.output))

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> SimID:
        """``(receptor.id, variant.name)``."""
        return SimID(self.receptor.id, self.variant.name)

    # -- where the results are ---------------------------------------------

    # The folders hold many receptors' results; a simulation is one receptor,
    # so they stay private and only this simulation's own paths are public.

    @property
    def _particle_set(self) -> Particles | None:
        """The folder holding this simulation's particles, or ``None`` until it exists."""
        return self.output.find_particles(self.variant.transport)

    @property
    def _footprint_set(self) -> Footprints | None:
        """The folder holding this variant's footprints, or ``None`` until it exists."""
        config = self.variant.footprint
        particles = self._particle_set
        if config is None or particles is None:
            return None
        return self.output.find_footprints(particles.hash, config)

    @property
    def particles_path(self) -> Path | None:
        """Path of the particle file in the output directory, or ``None`` before its folder exists."""
        folder = self._particle_set
        return None if folder is None else folder.file(self.receptor.id)

    @property
    def footprint_path(self) -> Path | None:
        """Path of the footprint file in the output directory, or ``None`` before its folder exists."""
        feet = self._footprint_set
        return None if feet is None else feet.file(self.receptor.id)

    @property
    def log_path(self) -> Path | None:
        """Path of the HYSPLIT log in the output directory, or ``None`` before its folder exists."""
        folder = self._particle_set
        return None if folder is None else folder.log_path(self.receptor.id)

    # -- presence and completion -------------------------------------------

    @property
    def has_particles(self) -> bool:
        """Whether the particle file exists."""
        folder = self._particle_set
        return folder is not None and folder.has(self.receptor.id)

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint file exists.

        An empty footprint (no particles over the grid) is a finished result.
        """
        feet = self._footprint_set
        return feet is not None and feet.has(self.receptor.id)

    @property
    def makes_footprint(self) -> bool:
        """Whether this simulation makes a footprint, which it does when its variant has a grid."""
        return self.variant.footprint is not None

    def is_complete(self) -> bool:
        """
        Return whether every expected result exists.

        That is the particles, and the footprint when the variant has a grid.
        """
        return self.has_particles and (not self.makes_footprint or self.has_footprint)

    # -- status ------------------------------------------------------------

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Start and end of the period the particles cover.

        Returns
        -------
        tuple of datetime
            ``(start, stop)`` with ``start < stop`` for backward and forward
            runs alike.
        """
        r_time = self.receptor.time
        other_end = r_time + dt.timedelta(hours=self.variant.transport.n_hours)
        return min(r_time, other_end), max(r_time, other_end)

    @property
    def empty_reason(self) -> str | None:
        """
        Why the footprint is empty, or ``None`` when it is not, or does not exist.

        ``"outside_domain"`` means no particle reached the grid and
        ``"no_particles"`` that there were none.
        """
        feet = self._footprint_set
        if feet is None or not feet.has(self.receptor.id):
            return None
        return feet.empty_reason(self.receptor.id)

    @property
    def outcome(self) -> str | None:
        """
        How this simulation ended, read from its results and log.

        Returns
        -------
        str or None
            ``"complete"`` if every expected result exists,
            ``"failed:<reason>"`` if a log exists but results are missing
            (see :class:`~stilt.transport.hysplit.FailureReason`), or ``None`` if the
            simulation has not run.
        """
        if self.is_complete():
            return "complete"
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            return None
        return f"failed:{identify_failure_reason(log_path)}"

    # -- reading the results -----------------------------------------------

    @property
    def log(self) -> str:
        """
        Text of the HYSPLIT log.

        Raises
        ------
        FileNotFoundError
            If the log has not been written yet.
        """
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            raise FileNotFoundError(f"No log for {self.id} yet.")
        return log_path.read_text()

    @property
    def met_files(self) -> list[Path]:
        """
        The meteorology files HYSPLIT read for these particles.

        Raises
        ------
        FileNotFoundError
            If the particles have not been written yet.
        """
        path = self.particles_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"{self.id} has no particles yet.")
        return particles_metadata(path).met_files

    @cached_property
    def particles(self) -> pd.DataFrame:
        """
        The particle table, one row per particle per output step.

        The columns are the variables in ``varsiwant`` (``indx``, ``time`` in
        minutes since release, ``long``, ``lati``, ``zagl``, ``foot``, ...),
        plus ``datetime`` (UTC) and ``xhgt`` (release height, for column and
        multipoint receptors). Read from the output directory on first
        access and kept. Check
        :attr:`has_particles` first when the run may not have finished.

        Raises
        ------
        FileNotFoundError
            If the particles have not been written yet. Nothing is kept, so
            a read after the run finishes loads them.
        """
        folder = self._particle_set
        if folder is None or not folder.has(self.receptor.id):
            raise FileNotFoundError(f"{self.id} has no particles yet.")
        return folder.read(self.receptor.id)

    @cached_property
    def footprint(self) -> xr.DataArray | None:
        """
        The footprint, or ``None`` when the variant has no grid or it is empty.

        An :class:`xarray.DataArray` with dimensions ``(time, lat, lon)``,
        whose ``.stilt`` accessor has the PYSTILT methods.

        Read from the output directory on first access and kept. An empty
        footprint (no particle reached the grid) is ``None`` with the reason
        in :attr:`empty_reason`. Check :attr:`has_footprint` first when the
        run may not have finished.

        Raises
        ------
        FileNotFoundError
            If the footprint has not been written yet. Nothing is kept, so a
            read after the run finishes loads it.
        """
        if not self.makes_footprint:
            return None
        feet = self._footprint_set
        if feet is None or not feet.has(self.receptor.id):
            raise FileNotFoundError(f"{self.id} has no footprint yet.")
        return feet.read(self.receptor.id)

    @cached_property
    def plot(self) -> SimulationPlotAccessor:
        """Plotting methods, such as ``sim.plot.map()``."""
        from stilt.visualization import SimulationPlotAccessor

        return SimulationPlotAccessor(self)

    def generate_footprint(
        self,
        config: FootprintConfig | None = None,
        transforms: Sequence[ParticleTransform] | None = None,
        context: TransformContext | None = None,
    ) -> xr.DataArray | None:
        """
        Calculate a footprint from the stored particles, without writing it.

        Use it to try other settings than the variant's, for example
        ``sim.variant.footprint.model_copy(update={"smooth_factor": 0.5})``.
        Nothing is written; the workers write the variant's own footprint.

        Parameters
        ----------
        config : FootprintConfig, optional
            Footprint settings. Defaults to the variant's own.
        transforms : sequence of ParticleTransform, optional
            Extra particle transforms, applied after ``config.transforms``
            and recorded with them.
        context : TransformContext, optional
            Passed to every transform. Defaults to one with the receptor and
            variant and no directory; pass ``project.transform_context(sim)``
            when a transform names a file relative to the project.

        Returns
        -------
        xarray.DataArray or None
            The footprint, or ``None`` when no particle reaches the grid.

        Raises
        ------
        TypeError
            If the variant has no grid and no *config* is given.
        FileNotFoundError
            If the particles have not been written yet.
        """
        if config is None:
            config = self.variant.footprint
        if config is None:
            raise TypeError(
                f"{self.id} has no footprint settings; pass a FootprintConfig."
            )
        particles = self.particles
        if transforms:
            config = config.model_copy(
                update={"transforms": [*config.transforms, *transforms]}
            )
        if context is None:
            context = TransformContext(
                receptor=self.receptor, variant=self.variant.name
            )
        try:
            return calculate(
                particles,
                self.receptor,
                config,
                name=self.variant.name,
                context=context,
            )
        except EmptyFootprint:
            return None


__all__ = ["SimID", "Simulation"]
