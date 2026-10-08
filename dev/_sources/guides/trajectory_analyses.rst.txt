Trajectory-Style Analyses
=========================

Tools such as HyTraj and pysplit work with a few trajectories per
measurement: residence time, potential source contribution (PSCF),
concentration-weighted trajectories (CWT), a mean path. This page does
the same with PYSTILT's results. A run gives hundreds of particles per
receptor, and a footprint that already weights them by their time near
the ground, so most of these analyses get simpler.

The examples use a month of receptors at one site, and a table of your
measurements with a ``receptor`` column, such as ``project.simulations``
merged with your data:

.. code-block:: python

   import numpy as np
   import pandas as pd

   sims = project.simulations
   july = sims[(sims.variant == "hrrr") & sims.time.between("2023-07-01", "2023-07-31 23:00")]
   obs = july.merge(measurements, on="receptor")   # one row per receptor, with "ch4"

Residence time
--------------

The footprint is STILT's residence time. It counts the time each particle
spends in the lower half of the mixed layer, divides by the air density
and the depth the flux mixes through, and sums over the particles. Summed
over the hours of the run and averaged over the receptors, it says where
the air came from near the ground:

.. code-block:: python

   footprints = project.footprints(july)
   residence = footprints.foot.sum("hour").mean("receptor")
   np.log10(residence.where(residence > 0)).plot()

For the time spent anywhere in the column, as HyTraj counts it, count
particle positions per cell. ``columns=`` reads only what you need:

.. code-block:: python

   particles = project.particles(july, columns=["lon", "lat"])
   x_edges = np.arange(-114.0, -109.0 + 0.1, 0.1)
   y_edges = np.arange(39.0, 42.0 + 0.1, 0.1)
   counts, _, _ = np.histogram2d(
       particles["lon"], particles["lat"], bins=[x_edges, y_edges]
   )
   residence_all = counts.T / counts.sum()   # (lat, lon), a fraction of all positions

Potential source contribution (PSCF)
------------------------------------

PSCF is, for each cell, the fraction of the visits that came from
receptors with a high value. With particles, each particle position is
one visit:

.. code-block:: python

   high = obs.loc[obs["ch4"] > obs["ch4"].quantile(0.75), "receptor"]
   visits, _, _ = np.histogram2d(particles["lon"], particles["lat"], bins=[x_edges, y_edges])
   high_rows = particles["receptor"].isin(high)
   high_visits, _, _ = np.histogram2d(
       particles.loc[high_rows, "lon"], particles.loc[high_rows, "lat"], bins=[x_edges, y_edges]
   )
   with np.errstate(invalid="ignore"):
       pscf = (high_visits / visits).T   # NaN where no particle went

Cells that few particles reached give noisy ratios. PSCF studies usually
down-weight cells with fewer visits than a few times the mean.

Concentration-weighted trajectories (CWT)
-----------------------------------------

CWT gives each cell the mean of the measured values, weighted by the
time the air spent over the cell. With footprints the weight is each
receptor's footprint, which counts only the time near the ground:

.. code-block:: python

   weight = footprints.foot.sum("hour")   # (receptor, lat, lon)
   value = obs.set_index("receptor")["ch4"].to_xarray().sel(receptor=weight.receptor)
   cwt = (weight * value).sum("receptor") / weight.sum("receptor")   # NaN where no footprint reached
   cwt.plot()

A receptor whose footprint is empty has no row in ``footprints``
(:doc:`checking`), so it does not count.

A mean path
-----------

The mean path of a receptor's particles is their mean position at each
output step:

.. code-block:: python

   particles = project.particles(july, columns=["time", "lon", "lat", "zagl"])
   path = particles.groupby(["receptor", "time"])[["lon", "lat", "zagl"]].mean()
   one = path.loc[rid].sort_index(ascending=False)   # from the release backwards

A mean path hides how the particles spread, which is what the footprint
keeps. ``sim.particles.stilt.plot.map()`` draws them all.

Comparing variants
------------------

Variants with different meteorology or settings run the same receptors,
so their results line up by receptor. The difference of two variants'
mean footprints shows where they disagree:

.. code-block:: python

   hrrr = project.footprints(sims[sims.variant == "hrrr"]).foot.sum("hour")
   nam = project.footprints(sims[sims.variant == "nam"]).foot.sum("hour")
   both = np.intersect1d(hrrr.receptor.values, nam.receptor.values)
   diff = (hrrr.sel(receptor=both) - nam.sel(receptor=both)).mean("receptor")
   diff.plot(cmap="RdBu_r", center=0)

Both variants need the same grid for this. For particles, compare the
endpoints of each receptor: ``sim.particles.stilt.endpoints()`` gives
each particle's last position.
