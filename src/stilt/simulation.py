"""A simulation, one receptor run under one variant, and where its results are."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from stilt.exceptions import EmptyFootprint
from stilt.footprint import gridding
from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import read_footprint
from stilt.output import Output, completed
from stilt.particles import (
    Background,
    TransportError,
    background,
    read_particles,
    transport_error,
)
from stilt.receptors import Receptor
from stilt.spatial import Grid

if TYPE_CHECKING:
    from stilt._paths import Location
    from stilt.config import Variant
    from stilt.visualization import SimulationPlotAccessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Simulation:
    """
    One receptor run under one variant, as one realization of an ensemble or alone.

    A simulation is a value: its receptor, its variant, and the output
    directory its results are in. From those it knows where its particles,
    footprint, and log are, whether they exist (:meth:`is_complete`), and how
    to read them. It runs nothing itself; the workers in
    :mod:`stilt.execution` run the transport model and write the results.

    You rarely build one yourself. Get it from a project instead, as in
    ``project.simulation(receptor_id, "hrrr")``.

    Parameters
    ----------
    receptor : Receptor
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

    receptor: Receptor
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
            f"Simulation(receptor={self.receptor.id!r}, "
            f"variant={self.variant.name!r}{realization})"
        )

    def __str__(self) -> str:
        name = f"{self.receptor.id}/{self.variant.name}"
        return name if self.realization is None else f"{name}/{self.realization}"

    def __hash__(self) -> int:
        return hash((self.id, self.output))

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> tuple[str, str, int | None]:
        """``(receptor.id, variant.name, realization)``, the row of ``project.simulations`` it is."""
        return (self.receptor.id, self.variant.name, self.realization)

    @property
    def transport(self) -> Any:
        """The transport config this simulation runs with: the variant's, with ``seed + realization`` for an ensemble."""
        return self.variant.transport_for(self.realization)

    # -- where the results are ---------------------------------------------

    @property
    def particles_path(self) -> Location | None:
        """Path of the particle file in the output directory, or ``None`` before its folder exists."""
        return self.output.path(
            "particles", self.variant, self.receptor.id, self.realization
        )

    @property
    def footprint_path(self) -> Location | None:
        """Path of the footprint file in the output directory, or ``None`` before its folder exists."""
        return self.output.path(
            "footprints", self.variant, self.receptor.id, self.realization
        )

    @property
    def log_path(self) -> Location | None:
        """Path of the transport model's log in the output directory, or ``None`` before its folder exists."""
        return self.output.log_path(self.variant, self.receptor.id, self.realization)

    @property
    def kept_workdir(self) -> Location | None:
        """
        Where a failed run's working directory is kept, or ``None`` before its folder exists.

        The directory, under ``scratch/`` in the output directory, holds
        the transport model's files as the run left them: for HYSPLIT,
        CONTROL, SETUP.CFG, and its own output. It exists only after a
        failed run, or after any run with ``keep_scratch`` set.
        """
        return self.output.kept_workdir(
            self.variant, self.receptor.id, self.realization
        )

    # -- presence and completion -------------------------------------------

    @property
    def has_particles(self) -> bool:
        """Whether the particle file exists."""
        path = self.particles_path
        return path is not None and path.exists()

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint file exists.

        An empty footprint (no particles over the grid) is a finished result.
        """
        path = self.footprint_path
        return path is not None and path.exists()

    @property
    def is_complete(self) -> bool:
        """
        Whether every expected result exists.

        That is the particles, and the footprint when the variant has a grid.
        It is :func:`stilt.output.completed`, the one definition of done,
        for one receptor, from two file checks.
        """
        rid = self.receptor.id
        mine = frozenset([rid])
        particles = mine if self.has_particles else frozenset()
        if self.variant.footprint is None:
            return rid in completed(particles, None)
        return rid in completed(particles, mine if self.has_footprint else frozenset())

    # -- status ------------------------------------------------------------

    @property
    def failure(self) -> dict[str, Any] | None:
        """
        Why this simulation is not complete, as the worker recorded it, or ``None``.

        ``None`` when it is complete, or has not run, or the step that
        failed has since succeeded. Otherwise a dict with ``step``
        (``"particles"`` or ``"footprint"``), ``reason`` (a short cause
        such as ``"MET_COVERAGE"``, or the error's class when it has
        none), ``message``, and ``time``. An unexpected error also has a
        ``traceback``. A failed transport model run fails every variant
        that shares its particles. :attr:`log` reads the model's log and
        :attr:`kept_workdir` is where its workdir was kept.

        Examples
        --------
        >>> sim.failure
        {'step': 'particles', 'reason': 'MET_COVERAGE',
         'message': 'HYSPLIT: start point not within (x,y,t) any data file', ...}
        """
        if self.is_complete:
            return None
        kind = "footprints" if self.has_particles else "particles"
        return self.output.failure(
            kind, self.variant, self.receptor.id, self.realization
        )

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

        The columns are the variables in ``varsiwant`` (``particle``, ``time`` in
        minutes since release, ``lon``, ``lat``, ``zagl``, ``foot``, ...),
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
        path = self.particles_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"{self} has no particles yet.")
        return read_particles(path)

    @cached_property
    def footprint(self) -> xr.DataArray | None:
        """
        The footprint, or ``None`` when the variant has no grid or it is empty.

        An :class:`xarray.DataArray` with dimensions ``(time, lat, lon)``,
        whose ``.stilt`` accessor has the PYSTILT methods.

        Read from the output directory on first access and kept. An empty
        footprint, where no particle reached the grid, is ``None`` while
        :attr:`has_footprint` is true. Check :attr:`has_footprint` first
        when the run may not have finished.

        Raises
        ------
        FileNotFoundError
            If the footprint has not been written yet. Nothing is kept, so a
            read after the run finishes loads it.
        """
        if self.variant.footprint is None:
            return None
        path = self.footprint_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"{self} has no footprint yet.")
        return read_footprint(path)

    @cached_property
    def plot(self) -> SimulationPlotAccessor:
        """Plotting methods, such as ``sim.plot.map()``."""
        from stilt.visualization import SimulationPlotAccessor

        return SimulationPlotAccessor(self)

    def calc_footprint(
        self,
        *,
        grid: Grid | None = None,
        smooth_factor: float | None = None,
        time_integrate: bool | None = None,
        transforms: Sequence[Any] | None = None,
    ) -> xr.DataArray | None:
        """
        Calculate a footprint from the stored particles, without writing it.

        This is :func:`stilt.calc_footprint` with this simulation's
        particles, receptor, and settings filled in. Each setting given
        replaces the variant's own. Use it to try other settings; the
        workers write the variant's own footprint. A relative file name in
        a transform's settings starts from the project directory.

        Returns
        -------
        xarray.DataArray or None
            The footprint, or ``None`` when no particle reaches the grid.

        Raises
        ------
        TypeError
            If the variant has no grid and none is given.
        FileNotFoundError
            If the particles have not been written yet.

        Examples
        --------
        >>> smoother = sim.calc_footprint(smooth_factor=2.0)
        >>> fine = sim.calc_footprint(grid=hexes.to_grid(cells_per_target=4))
        """
        own = self.variant.footprint
        if own is None:
            own = FootprintConfig.model_validate({})
        # The variant's geometry names its own grid, not another one.
        geometry_hash = self.variant.geometry_hash if grid is None else None
        grid = grid if grid is not None else own.grid
        if grid is None:
            raise TypeError(f"{self} has no footprint settings; give a grid.")
        particles = self.particles
        try:
            return gridding.calc_footprint(
                particles,
                self.receptor,
                grid,
                smooth_factor=own.smooth_factor
                if smooth_factor is None
                else smooth_factor,
                time_integrate=own.time_integrate
                if time_integrate is None
                else time_integrate,
                transforms=own.transforms if transforms is None else transforms,
                name=self.variant.name,
                directory=self.directory,
                geometry_hash=geometry_hash,
            )
        except EmptyFootprint:
            return None

    @property
    def _transforms(self) -> Sequence[Any]:
        """The footprint's particle transforms, which weight the particles; none without a footprint."""
        return (
            () if self.variant.footprint is None else self.variant.footprint.transforms
        )

    def background(self, field: xr.DataArray | pd.Series) -> Background:
        """
        Return the background mole fraction at the receptor.

        The background is the field where each particle ends, averaged over
        the particles with the footprint's weights (its transforms), so it
        adds to the enhancement. See :func:`stilt.particles.background`.

        Parameters
        ----------
        field : xarray.DataArray or pandas.Series
            A mole-fraction field such as CarbonTracker, whose vertical
            dimension is named after a particle column (``pres`` or
            ``zagl``), or one value per particle, indexed by ``particle``.

        Examples
        --------
        >>> ct = xr.open_dataarray("ct_ch4.nc").rename(level="pres")
        >>> sim.background(ct).value
        """
        return background(
            self.particles,
            field,
            transforms=self._transforms,
            receptor=self.receptor,
            directory=self.directory,
        )

    def transport_error(
        self,
        error: Simulation | Sequence[Simulation],
        flux: xr.DataArray,
        **options: Any,
    ) -> TransportError:
        """
        Return the transport error of this simulation's modeled enhancement.

        It is the extra spread in the per-particle enhancement that a
        wind-error variant of the same receptor adds (Lin and Gerbig, 2005).
        See :func:`stilt.particles.transport_error`, which documents the
        options (``levels``, ``length_scale``, ``percentile``,
        ``noise_splits``, ``background``).

        Parameters
        ----------
        error : Simulation or sequence of Simulation
            The same receptor under a wind-error variant, or every
            realization of one.
        flux : xarray.DataArray
            Surface flux field, in µmol m⁻² s⁻¹.

        Examples
        --------
        >>> err = project.simulation(sim.receptor.id, "hrrr-err")
        >>> sim.transport_error(err, flux).variance
        >>> ensemble = [project.simulation(rid, "hrrr-err", k) for k in range(10)]
        >>> sim.transport_error(ensemble, flux)
        """
        sims = [error] if isinstance(error, Simulation) else list(error)
        return transport_error(
            self.particles,
            [e.particles for e in sims],
            flux,
            transforms=self._transforms,
            receptor=self.receptor,
            directory=self.directory,
            **options,
        )


__all__ = ["Simulation"]
