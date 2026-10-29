Trajectory-Style Analyses
=========================

Tools such as HyTraj and pysplit work with a few trajectories per
measurement: residence time, potential source contribution (PSCF),
concentration-weighted trajectories (CWT), a mean path, the time the
air took from a source. This page does the same with PYSTILT's results.
A run gives hundreds of particles per receptor, and a footprint that
already weights them by their time near the ground, so most of these
analyses get simpler.

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

   particles = project.particles(july, columns=["age", "lon", "lat", "zagl"])
   path = particles.groupby(["receptor", "age"])[["lon", "lat", "zagl"]].mean()
   one = path.loc[rid].sort_index(ascending=False)   # from the release backwards

A mean path hides how the particles spread, which is what the footprint
keeps. ``sim.particles.stilt.plot.map()`` draws them all.

Transit time from a source
--------------------------

How long did the air take to get here from a source region, and how many
of those hours were in sunlight? For smoke at a mountain site, these say
more about the ozone the smoke made on the way than the time it arrived.
A backward run answers both for each particle. Its *transit time* is the
smallest ``|age|`` among its rows inside the region: the time since it
last left the region before it reached the receptor. A particle that
never enters the region did not come from it.

Draw the region as a shapely polygon in longitude and latitude, around a
fire, a city, or a basin. ``top`` limits it to the air below a height
above ground. Leave it ``None`` for smoke that may have been lofted, or
use each row's ``mlht`` (read it too) for air in the mixed layer over a
surface source:

.. code-block:: python

   import shapely
   from shapely.geometry import Polygon

   source = Polygon([(-113.1, 40.7), (-112.2, 40.7), (-112.2, 41.7), (-113.1, 41.7)])
   top = None   # meters above ground; None for any height

   particles = project.particles(july, columns=["particle", "age", "time", "lon", "lat", "zagl"])
   particles["hours"] = particles["age"].abs() / 60

   inside = shapely.contains_xy(source, particles["lon"], particles["lat"])
   if top is not None:
       inside &= particles["zagl"].to_numpy() <= top
   transit = (
       particles[inside].groupby(["receptor", "particle"])["hours"].min().rename("transit")
   )

The sun's elevation at each row's time and place comes from `pvlib
<https://pvlib-python.readthedocs.io>`_ (``pip install pvlib``), whose
``spa_python`` takes arrays of times and positions. Only the rows between
leaving the region and arriving count, and each counts for one output
step:

.. code-block:: python

   import pvlib

   route = particles.join(transit, on=["receptor", "particle"], how="inner")
   route = route[route["hours"] < route["transit"]]
   sun = pvlib.solarposition.spa_python(
       pd.DatetimeIndex(route["time"]).tz_localize("UTC"),
       route["lat"].to_numpy(),
       route["lon"].to_numpy(),
   )
   route["sunlit"] = sun["elevation"].to_numpy() > 0
   step = particles["age"].abs().drop_duplicates().sort_values().diff().median() / 60   # hours per row
   sunlit = route.groupby(["receptor", "particle"])["sunlit"].sum().mul(step).rename("sunlit")

Then, for each receptor, the fraction of particles that came from the
region, their mean transit time and its spread, and their mean hours in
sunlight:

.. code-block:: python

   per_particle = pd.concat([transit, sunlit], axis=1).fillna({"sunlit": 0.0})
   numpar = particles.groupby("receptor")["particle"].nunique()
   summary = per_particle.groupby("receptor").agg(
       transit_mean=("transit", "mean"),
       transit_sd=("transit", "std"),
       sunlit_mean=("sunlit", "mean"),
   ).reindex(numpar.index)
   summary["from_source"] = (
       per_particle.groupby("receptor").size().reindex(numpar.index).fillna(0) / numpar
   )
   summary["sunlit_fraction"] = summary["sunlit_mean"] / summary["transit_mean"]

   both = obs.merge(summary, left_on="receptor", right_index=True)

A receptor whose particles never reached the region has ``from_source``
0 and NaN for the rest. ``from_source`` is itself a marker of the
source's air that needs no inventory. With a ratio of enhancements in
``obs``, such as ozone to CO for smoke, plot it against
``transit_mean`` and color it by ``sunlit_fraction``. The transit times
of one receptor's particles, ``per_particle.loc[rid, "transit"]``, give
the whole distribution.

A few things to know:

- **The run must be longer than the transit.** A particle that would
  reach the region only after ``n_hours`` is not counted, so
  ``from_source`` comes out low and ``transit_mean`` short. If the
  distribution runs up to ``n_hours``, run longer.
- **Memory.** A 24-hour run of 1,000 particles is 1.4 million rows, so a
  day of hourly receptors is about 35 million, and this recipe peaked near
  10 GB on one. For more, run it a day at a time and concatenate the
  summaries.
- **The horizon.** ``elevation > 0`` counts the hours the sun is up.
  Photolysis falls off quickly once it sets; ``> -6`` adds civil
  twilight.
- **Transport, not chemistry.** The transit time is how long the air
  traveled, not a chemical age from ratios of species.

Some variations:

- **A source that moves or switches on and off,** such as a fire's daily
  hotspots: make ``inside`` from a 0/1 field with a ``time`` dimension,
  ``sample_field(mask, particles["lon"], particles["lat"], times=particles["time"], fill_value=0) > 0``
  (:func:`~stilt.sampling.sample_field`).
- **Weighting by emissions.** With an emission field, ``foot`` times the
  emission at each row is what that row picked up, and the mean of
  ``hours`` weighted by it is the age of what was emitted into the air.
  ``foot`` counts only the lower half of the mixed layer, though, so
  smoke injected above the boundary layer drops out. The region version
  above does not depend on ``foot``.
- **Forward runs from the source.** When the release is the source, a
  row's ``age`` is already its transit time (:doc:`plume_background`).

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
