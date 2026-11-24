"""
:class:`stilt.Simulation`: a simulation, with the reads of what STILT makes from its particles.

It is the particle part (:class:`runtraj.simulation.Simulation`) and only
adds: the footprint, a footprint with other settings, the background, the
transport error, and the plots that draw them. A STILT project
(:class:`stilt.Project`) gives you these. The workers and the stages use the
particle part, and nothing on disk names this class.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from runtraj import simulation

from stilt.background import Background, background
from stilt.exceptions import EmptyFootprint
from stilt.footprint import gridding
from stilt.footprint.config import FootprintConfig
from stilt.footprint.grid import Grid
from stilt.footprint.io import read_footprint
from stilt.footprint.stage import simulation_particles, variant_footprint
from stilt.transport_error import TransportError, transport_error

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath

    from stilt.visualization import SimulationPlotAccessor


class Simulation(simulation.Simulation):
    """
    One release run under one variant, with its footprint, background, and transport error.

    It is :class:`runtraj.simulation.Simulation`, which knows where the
    particles and the file of each stage are and reads the particles, and it
    adds the reads of what STILT makes from them: :attr:`footprint`,
    :meth:`calc_footprint`, :meth:`background`, :meth:`transport_error`,
    and :attr:`plot`. It changes nothing it inherits.

    You rarely build one yourself. Get it from a project instead, as in
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

    # -- the footprint -----------------------------------------------------

    @property
    def footprint_path(self) -> Path | UPath | None:
        """Path of the footprint file in the output directory, or ``None`` before its folder exists."""
        return self.path("footprint")

    @property
    def has_footprint(self) -> bool:
        """
        Whether the footprint file exists.

        An empty footprint (no particles over the grid) is a finished result.
        """
        return self.has("footprint")

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
        if "footprint" not in self.variant.stages:
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
        tres: str | timedelta | None = None,
        hnf_plume: bool | None = None,
        transforms: Sequence[Any] | None = None,
    ) -> xr.DataArray | None:
        """
        Calculate a footprint from the stored particles, without writing it.

        This is :func:`stilt.calc_footprint` with this simulation's
        particles, release, and footprint settings filled in, as the
        footprint stage makes it: the particles get the ``foot`` those
        settings ask for (:func:`stilt.footprint.footprint_particles`).
        Each setting given replaces the variant's own. Use it to try other settings; the
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
        >>> smoother = sim.calc_footprint(smooth_factor=2.0)  # doctest: +SKIP
        >>> quarter_hours = sim.calc_footprint(tres="15min")  # doctest: +SKIP
        >>> fine = sim.calc_footprint(
        ...     grid=hexes.to_grid(cells_per_target=4)
        ... )  # doctest: +SKIP
        """
        own = self._footprint_config
        # The variant's geometry names its own grid, not another one.
        geometry_hash = own.geometry_hash if grid is None else None
        grid = grid if grid is not None else own.grid
        if grid is None:
            raise TypeError(f"{self} has no footprint settings; give a grid.")
        particles = simulation_particles(self, own)
        try:
            return gridding.calc_footprint(
                particles,
                self.release,
                grid,
                smooth_factor=own.smooth_factor
                if smooth_factor is None
                else smooth_factor,
                tres=own.tres if tres is None else tres,
                hnf_plume=own.hnf_plume if hnf_plume is None else hnf_plume,
                footprint_depth=own.footprint_depth,
                transforms=own.transforms if transforms is None else transforms,
                name=self.variant.name,
                directory=self.directory,
                geometry_hash=geometry_hash,
            )
        except EmptyFootprint:
            return None

    @property
    def _footprint_config(self) -> FootprintConfig:
        """The footprint settings of this simulation: the variant's, or the defaults with the model's depth (:func:`stilt.footprint.stage.variant_footprint`)."""
        return variant_footprint(self.variant)

    # -- what the particles give beyond the footprint ----------------------

    def background(self, field: xr.DataArray | pd.Series) -> Background:
        """
        Return the background mole fraction at the release.

        The background is the field where each particle ends, averaged over
        the particles with the footprint's weights (its transforms), so it
        adds to the enhancement. See :func:`stilt.background`.

        Parameters
        ----------
        field : xarray.DataArray or pandas.Series
            A mole-fraction field such as CarbonTracker, whose vertical
            dimension is named after a particle column (``pres`` or
            ``zagl``), or one value per particle, indexed by ``particle``.

        Examples
        --------
        >>> ct = xr.open_dataarray("ct_ch4.nc").rename(level="pres")  # doctest: +SKIP
        >>> sim.background(ct).value  # doctest: +SKIP
        """
        return background(
            self.particles,
            field,
            transforms=self._footprint_config.transforms,
            release=self.release,
            directory=self.directory,
        )

    def transport_error(
        self,
        error: simulation.Simulation | Sequence[simulation.Simulation],
        flux: xr.DataArray,
        **options: Any,
    ) -> TransportError:
        """
        Return the transport error of this simulation's modeled enhancement.

        It is the extra spread in the per-particle enhancement that a
        wind-error variant of the same release adds (Lin and Gerbig, 2005).
        See :func:`stilt.transport_error`, which documents the
        options (``levels``, ``length_scale``, ``percentile``,
        ``background``).

        Parameters
        ----------
        error : Simulation or sequence of Simulation
            The same release under a wind-error variant, or every
            realization of one.
        flux : xarray.DataArray
            Surface flux field, in µmol m⁻² s⁻¹.

        Examples
        --------
        >>> err = project.simulation(sim.release.id, "hrrr-err")  # doctest: +SKIP
        >>> sim.transport_error(err, flux).variance  # doctest: +SKIP
        >>> ensemble = [
        ...     project.simulation(rid, "hrrr-err", k) for k in range(10)
        ... ]  # doctest: +SKIP
        >>> sim.transport_error(ensemble, flux)  # doctest: +SKIP
        """
        sims = [error] if isinstance(error, simulation.Simulation) else list(error)
        config = self._footprint_config
        near_field: dict[str, Any] = {
            "hnf_plume": config.hnf_plume,
            "footprint_depth": config.footprint_depth,
        }
        return transport_error(
            simulation_particles(self, config),
            [simulation_particles(e, config) for e in sims],
            flux,
            transforms=config.transforms,
            release=self.release,
            directory=self.directory,
            **{**near_field, **options},
        )


__all__ = ["Simulation"]
