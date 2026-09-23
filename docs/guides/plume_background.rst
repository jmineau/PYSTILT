Plume Background
================

:doc:`background` takes the background from a model field at the
trajectory endpoints. When the observation is a satellite swath, the swath
itself holds the background: soundings the city's plume did not reach saw
clean air at the same time, from the same instrument, with the same biases.
The question is which soundings those are. Forward runs answer it: release
particles from the city over the hours before the overpass, look at where
they are when the satellite passes, and draw the plume around them. X-STILT
does this in ``fit.kde.plume`` and ``calc.bg.upwind`` (Wu et al. 2018,
method M3); PYSTILT does it with
:func:`~stilt.observations.plume_polygon` and
:func:`~stilt.observations.plume_background`.

Forward runs
------------

A forward run is a run with positive ``n_hours``. Everything else is the
same: the receptor is where and when the particles are released, and the
particle table records where they go. Since a plume is what a whole city
puts out, the release is spread over a box around it, at a height just
above the surface, and repeated every half hour over the hours before the
overpass, so that air of every age is represented. X-STILT's defaults are a
0.3° box, 10 m above ground, releases from ten hours before the overpass
to the overpass itself, each followed twelve hours forward with a thousand
particles. In PYSTILT each release is a
:class:`~stilt.receptors.MultiPointReceptor` and the box is spread with
:func:`~stilt.observations.jitter_points`:

.. code-block:: python

   import pandas as pd
   import stilt
   from stilt.observations import jitter_points
   from stilt.receptors import MultiPointReceptor

   site = (-111.9, 40.75)                       # Salt Lake City
   overpass = pd.Timestamp("2023-10-19 19:42")  # from the sounding times
   half = 0.15
   box = [(site[0] - half, site[1] - half), (site[0] + half, site[1] - half),
          (site[0] + half, site[1] + half), (site[0] - half, site[1] + half)]
   points = jitter_points(box, 100)

   receptors = [
       MultiPointReceptor(t, [p[0] for p in points], [p[1] for p in points],
                          [10.0] * len(points))
       for t in pd.date_range(overpass - pd.Timedelta(hours=10), overpass, freq="30min")
   ]

   forward = stilt.Model(
       project="./forward_12h",     # a project is a transport setup; keep forward runs in their own
       receptors=receptors,
       mets={"hrrr": met},
       n_hours=12,                  # positive: forward in time
       numpar=1000,
   )
   forward.run()

No footprint is needed. Only the particle tables matter, and they hold each
particle's position at every output step with its absolute ``datetime``.

The plume
---------

Pool the particle rows that fall in the overpass window across all the
forward runs, and outline them:

.. code-block:: python

   from stilt.observations import plume_polygon

   window = (overpass - pd.Timedelta(minutes=3), overpass + pd.Timedelta(minutes=3))
   rows = []
   for sim_id in forward.simulations:
       p = forward.simulations[sim_id].trajectories.data
       rows.append(p[p["datetime"].between(*window)])
   particles = pd.concat(rows)

   plume = plume_polygon(particles["long"], particles["lati"])
   plume.polygon      # shapely Polygon in longitude and latitude
   plume.density      # the kernel density it was cut from, (lat, lon), max 1

The plume is where the 2-D kernel density of those positions is at least a
tenth of its maximum. ``threshold`` moves that cut: higher is a tighter
plume, lower a wider one. ``bandwidth`` smooths the density (0.1° in
longitude and 0.15° in latitude by default; the kernel's standard deviation
is a quarter of that, as in R's ``kde2d``). When the density above the
threshold breaks into separate pieces, the largest is the plume. The
outline follows the density grid's cells, 100 by 100 over the particles by
default, so it is a little blocky; ``n`` refines it.

Plot ``plume.density`` with the soundings and the outline before trusting
it. A plume that misses the swath, or one that covers it entirely, has no
background to offer.

The background
--------------

Give the soundings of the overpass, and the plume, to
:func:`~stilt.observations.plume_background`:

.. code-block:: python

   from stilt.observations import plume_background, read_tropomi_ch4

   obs = read_tropomi_ch4(path, lon_range=(-113.5, -110.5), lat_range=(39.5, 42.0))
   obs = obs[obs["good"]]

   bg = plume_background(
       obs["longitude"], obs["latitude"], obs["value"], plume,
       uncertainties=obs["uncertainty"],
   )
   bg.value          # the background, ppb here
   bg.uncertainty    # spread of the background soundings and their retrieval error, in quadrature
   bg.sides          # the same per side of the plume
   obs["in_plume"] = bg.in_plume
   obs["enhancement"] = obs["value"] - bg.value

The soundings inside the plume are the enhanced ones. Around their
bounding box, padded by ``pad`` (a tenth of its size), the out-of-plume
soundings within ``width`` (half a degree) on each side are the background
candidates, and the background is their median. ``bg.sides`` has the
count, mean, median, spread and retrieval error for the north, south, east
and west strips separately. If they disagree, one side is probably
downwind of something; pass ``side="west"`` (or whichever is upwind, which
the particles' drift tells you) to use that side alone.

Before any of that, ``trim`` drops the out-of-plume soundings above the
90th percentile of their values. That guards against enhanced air the
outline missed, at the cost of a slightly low background when the field
is clean; ``trim=None`` keeps everything.

:func:`~stilt.observations.plume_background` raises when no sounding lies
inside the plume, since then the overpass did not see the city. Pass only
good-quality soundings; missing values are ignored.

Choices made here
-----------------

- The background is a statistic of the swath, not of a model. It carries
  the instrument's bias, which cancels when the enhancement is
  ``value - background`` from the same swath.
- The plume outline is cut from the density on its grid, not traced as a
  contour, so it is exact on that grid and never self-intersects. X-STILT
  traces the contour and repairs broken pieces; the largest-area rule here
  does the same job.
- The plume's threshold is yours. X-STILT raises it when no low density
  falls in the release box; here you see the density and decide.
- Side strips are clipped to the plume box's extent along the strip, so
  a northern strip does not run across the whole swath.
- Soundings are points at their centres. A pixel that straddles the
  outline counts by where its centre falls.
