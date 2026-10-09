"""
The ``.stilt`` accessor on a particle table: endpoints, enhancement, plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from stilt.sampling import sample_field

if TYPE_CHECKING:
    from stilt.visualization import ParticlesPlotAccessor


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor:
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    Examples
    --------
    >>> particles = sim.particles
    >>> particles.stilt.endpoints()
    >>> particles.stilt.enhancement(flux).mean()
    >>> particles.stilt.plot.map()
    """

    def __init__(self, particles: pd.DataFrame) -> None:
        self._particles = particles

    def endpoints(self) -> pd.DataFrame:
        """
        Return the last row of each particle, the one farthest in time from release.

        For a backward run this is where the air came from, which is where
        to sample a background concentration field. For a forward run it is
        where the air went. A particle that left the meteorology domain
        early ends where it left.

        Returns
        -------
        pandas.DataFrame
            One row per particle, with every column of the particle table.
            ``age`` is minutes since release and ``time`` the UTC time.
        """
        p = self._particles.reset_index(drop=True)
        if p.empty:
            return p
        reach = p["age"].abs()
        last = reach.groupby(p["particle"], sort=False).idxmax().to_numpy(dtype=int)
        return p.iloc[last]

    def enhancement(self, flux: xr.DataArray) -> pd.Series:
        """
        Return each particle's enhancement, ``foot`` times flux summed along its trajectory.

        The mean over particles, after any weighting, is the modeled
        enhancement at the receptor. Unlike ``foot.stilt.enhancement``, the
        flux is taken at each particle position, with no gridding or
        smoothing.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid, in µmol m⁻² s⁻¹ for an
            enhancement in ppm. A flux with a ``time`` dimension is taken at
            each particle's ``time``. Points outside it, and missing
            cells, count as zero flux.

        Returns
        -------
        pandas.Series
            Enhancement indexed by ``particle``. A particle that never crosses
            the flux field gets 0.

        Raises
        ------
        ValueError
            If the flux varies in time and the particles have no
            ``time`` column.
        """
        p = self._particles
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
        )
        contribution = p["foot"].to_numpy(dtype=float) * sampled
        ids = p["particle"].to_numpy()
        unique, inverse = np.unique(ids, return_inverse=True)
        sums = np.bincount(inverse, weights=contribution, minlength=unique.size)
        return pd.Series(
            sums, index=pd.Index(unique, name="particle"), name="enhancement"
        )

    @property
    def plot(self) -> ParticlesPlotAccessor:
        """Plotting methods, such as ``particles.stilt.plot.map()``."""
        from stilt.visualization import ParticlesPlotAccessor

        return ParticlesPlotAccessor(self._particles)
