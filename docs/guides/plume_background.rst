Plume Background
================

:doc:`background` takes the background from a model field at the
trajectory endpoints. For a satellite swath, the swath itself holds a
background. Soundings that the city's plume did not reach saw clean air at
the same time, with the same instrument and the same biases. To find those
soundings, run particles forward from the city over the hours before the
overpass, see where they are when the satellite passes, and draw the plume
around them. :func:`~stilt.observations.plume_polygon` draws the plume and
:func:`~stilt.observations.plume_background` picks the background
soundings. This follows X-STILT's ``fit.kde.plume`` and ``calc.bg.upwind``
(method M3 in `Wu et al., 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_).

Forward runs
------------

A forward run is a run with a positive ``n_hours``. Everything else stays
the same. The receptor is where and when the particles are released, and
the particle table records where they go.

A plume is what the whole city puts out, so spread the release over a box
around the city, just above the surface. Repeat it every half hour over
the hours before the overpass, so that air of every age is included.
X-STILT's defaults are a 0.3° box, 10 m above ground, and releases from ten
hours before the overpass up to the overpass, each followed twelve hours
forward with 1000 particles. In PYSTILT each release is a
:class:`~stilt.receptors.MultiPointReceptor`, and
:func:`~stilt.observations.jitter_points` spreads its points over the box:

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
       MultiPointReceptor(
           time=t,
           longitudes=[p[0] for p in points],
           latitudes=[p[1] for p in points],
           altitudes=[10.0] * len(points),
       )
       for t in pd.date_range(overpass - pd.Timedelta(hours=10), overpass, freq="30min")
   ]

   forward = stilt.Project.init(
       "./forward_12h",             # keep forward runs in a project of their own
       receptors=receptors,
       mets={"hrrr": met},          # as in the quickstart
       n_hours=12,                  # positive: forward in time
       numpar=1000,
   )
   forward.run()

No footprint is needed. The plume only needs the particle tables, which
hold each particle's position at every output step along with its
``datetime``.

The plume
---------

Pool the particle rows that fall within a few minutes of the overpass,
across all the forward runs, and outline them:

.. code-block:: python

   from stilt.observations import plume_polygon

   window = (overpass - pd.Timedelta(minutes=3), overpass + pd.Timedelta(minutes=3))
   rows = []
   for p in forward.load_particles().values():
       rows.append(p[p["datetime"].between(*window)])
   particles = pd.concat(rows)

   plume = plume_polygon(particles["long"], particles["lati"])
   plume.polygon      # shapely Polygon in longitude and latitude
   plume.density      # the kernel density it was cut from, (lat, lon), max 1

The plume is where the 2-D kernel density of these positions is at least a
tenth of its maximum. ``threshold`` sets that fraction. A higher value
gives a tighter plume, and a lower value a wider one. ``bandwidth`` sets
the smoothing, 0.1° in longitude and 0.15° in latitude by default. As in
R's ``kde2d``, the kernel's standard deviation is a quarter of the
bandwidth. If the area above the threshold falls into separate pieces, the
largest piece is the plume. The outline follows the cells of the density
grid, 100 by 100 over the particles by default, so it looks a little
blocky. Raise ``n`` for a finer outline.

Plot ``plume.density`` with the soundings and the outline before you trust
it. A plume that misses the swath, or one that covers all of it, leaves no
background to take.

The background
--------------

Pass the soundings of the overpass and the plume to
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
   bg.sides          # the same for each side of the plume
   obs["in_plume"] = bg.in_plume
   obs["enhancement"] = obs["value"] - bg.value

The soundings inside the plume are the enhanced ones. PYSTILT draws the
bounding box of those soundings and pads it on each side by ``pad`` times
its size (0.1 by default). The background candidates are the soundings
outside the plume within ``width`` degrees (0.5 by default) of that box, to
the north, south, east, or west. The background is their median.

``bg.sides`` has the count, mean, median, spread, and retrieval error of
the soundings on each side. If the sides disagree, one of them is probably
downwind of another source. Pass the upwind side, for example
``side="west"``, to use that side alone. The drift of the forward particles
tells you which side is upwind.

Before any of that, ``trim`` drops the out-of-plume soundings above the
90th percentile of their values. This guards against enhanced air that the
outline missed, at the cost of a slightly low background when the air is
clean. ``trim=None`` keeps every sounding.

:func:`~stilt.observations.plume_background` raises an error when no
sounding lies inside the plume, because then the overpass did not see the
city. Pass only good-quality soundings. Missing values are ignored.

Choices made here
-----------------

- The background comes from the swath, not from a model. It carries the
  instrument's bias, which cancels when the enhancement is
  ``value - background`` from the same swath.
- The outline is built from the density grid cells above the threshold
  rather than traced as a contour. It matches the grid exactly and never
  crosses itself. X-STILT traces the contour and repairs broken pieces.
  Keeping the largest piece does the same job here.
- You choose the threshold. X-STILT raises it automatically when no low
  density falls in the release box. Here you look at the density and
  decide.
- The side strips cover only the length of the padded box, so a northern
  strip does not run across the whole swath.
- Each sounding is a point at its centre. A pixel that straddles the
  outline counts as inside or outside by where its centre falls.
