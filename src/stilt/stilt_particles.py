"""
The ``.stilt`` accessor on a particle table: the particle part's methods, and the enhancement.

Importing :mod:`stilt` registers :class:`ParticlesAccessor` as
``particles.stilt`` on every pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from stilt.particles import accessor
from stilt.particles.table import _need_foot
from stilt.sampling import sample_field


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor(accessor.ParticlesAccessor):
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    The particle part's methods (:class:`stilt.particles.ParticlesAccessor`)
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
            enhancement in ppm, read the same way ``foot.stilt.enhancement``
            reads it: by its CF time bounds when it has them (a row before
            the first bound, after the last, or in a gap between bounds
            gets zero flux), or otherwise as steps that start at their
            labels (each particle row takes the last flux step that starts
            at or before the row's ``time``; a row before the first label
            gets zero flux, and a row after the last label takes the last
            step). A flux not labeled at the start of its steps and with no
            bounds must be relabeled first (see ``foot.stilt.enhancement``
            and :ref:`flux-step-labels`). Points outside the flux grid, and
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
        ``sim.footprint``'s. For the same one, pass the particles through
        :func:`stilt.footprint.footprint_particles` first, with the met files
        the run read; they must still be where the run read them.
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
