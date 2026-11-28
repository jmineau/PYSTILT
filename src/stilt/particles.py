"""
The ``.stilt`` accessor on a particle table: the particle part's methods, and the enhancement.

Importing :mod:`stilt` registers :class:`ParticlesAccessor` as
``particles.stilt`` on every pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from runtraj.particles import accessor
from runtraj.sampling import sample_field


def _need_foot(particles: pd.DataFrame) -> None:
    """
    Raise unless *particles* have a ``foot`` column, saying how to get one.

    A particle file holds what the transport model wrote, and a model that
    writes only positions writes no ``foot``: the footprint stage computes
    one when it needs it and stores it nowhere.

    Raises
    ------
    ValueError
        If there is no ``foot`` column.
    """
    if "foot" in particles.columns:
        return
    raise ValueError(
        "The particle table has no 'foot' column: the transport model wrote "
        "none, and the footprint stage stores the foot it computes nowhere. "
        "stilt.footprint.footprint_particles adds the one the footprint uses, "
        "and sim.enhancement(flux) gives the enhancement from it."
    )


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor(accessor.ParticlesAccessor):
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    The particle part's methods (:class:`runtraj.particles.ParticlesAccessor`)
    and :meth:`enhancement`, which samples a surface flux at the particles.

    Examples
    --------
    >>> particles = sim.particles  # doctest: +SKIP
    >>> particles.stilt.endpoints()  # doctest: +SKIP
    >>> particles.stilt.grid_exits(grid)  # doctest: +SKIP
    >>> particles.stilt.enhancement(flux).mean()  # doctest: +SKIP
    >>> particles.stilt.to_cf_trajectories()  # doctest: +SKIP
    >>> particles.stilt.plot.map()  # doctest: +SKIP
    """

    def enhancement(self, flux: xr.DataArray) -> pd.Series:
        """
        Return each particle's enhancement, ``foot`` times flux summed along its trajectory.

        The mean over particles, after any weighting, is the modeled
        enhancement at the release. Unlike ``foot.stilt.enhancement``, the
        flux is taken at each particle position, with no gridding or
        smoothing.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid, in µmol m⁻² s⁻¹ for an
            enhancement in ppm. Its time steps are read as
            ``foot.stilt.enhancement`` reads them, with each particle row
            taking the step its ``time`` falls in (see
            :ref:`flux-step-labels`). Points outside the flux grid, and
            missing cells, count as zero flux.

        Returns
        -------
        pandas.Series
            Enhancement indexed by ``particle``. A particle that never crosses
            the flux field gets 0.

        Raises
        ------
        ValueError
            If the flux varies in time and the particles have no
            ``time`` column, or the particles have no ``foot``.

        Notes
        -----
        ``foot`` is the table's own. ``sim.particles`` holds the transport
        model's ``foot`` (HYSPLIT's, made with ``veght``), or none from a
        model that writes only positions. With ``pbl_height`` set the footprint
        stage computes its own ``foot``, so this enhancement differs from
        ``sim.footprint``'s. :meth:`stilt.Simulation.enhancement` gives the
        footprint's: it uses the ``foot``, near-field correction and
        transforms the footprint does.
        """
        p = self._particles
        _need_foot(p)
        times = p["time"].to_numpy() if "time" in p.columns else None
        if "time" in flux.dims and times is None:
            raise ValueError(
                "flux varies in time but the particles have no 'time' column."
            )
        sampled = sample_field(
            flux,
            p["lon"].to_numpy(),
            p["lat"].to_numpy(),
            times=times,
            fill_value=0.0,
            step_starts=True,
        )
        contribution = p["foot"].to_numpy(dtype=float) * sampled
        ids = p["particle"].to_numpy()
        unique, inverse = np.unique(ids, return_inverse=True)
        sums = np.bincount(inverse, weights=contribution, minlength=unique.size)
        return pd.Series(
            sums, index=pd.Index(unique, name="particle"), name="enhancement"
        )


__all__ = ["ParticlesAccessor"]
