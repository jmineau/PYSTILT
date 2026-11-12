Background
==========

A footprint gives the enhancement, the amount the fluxes inside the domain
add to a measurement. The measurement is that enhancement on top of the
background, the mole fraction of the air before it entered the domain. To
compare a modeled value with an observed one, or to subtract the
background from observations for an inversion, you need both.

Each back-trajectory ends where its air came from. Sample a mole-fraction
field at every particle's endpoint, average over the particles, and you
have the background the receptor saw. ``sim.background(field)`` does this
with any field you give it (:func:`stilt.particles.background` without a
project). X-STILT does the same with
CarbonTracker in ``endpts.trajfoot``. For a satellite swath you can also
take the background from the soundings that a city's plume did not reach
(:ref:`plume-background`).

The field
---------

The field is an :class:`xarray.DataArray` with ``lat`` and ``lon``
dimensions, and optionally ``time`` and one vertical dimension. Name the
vertical dimension after the particle column it should be matched against.
Use ``pres`` for pressure levels in hPa or ``zagl`` for height above ground
in meters. Both are in the particle table by default. You can also add a
column of your own, such as height above sea level from ``zagl + zsfc``
(add ``zsfc`` to ``varsiwant`` to get it).

.. code-block:: python

   import xarray as xr

   cams = xr.open_dataset("cams_ch4_2023-07.nc")["ch4"]     # (time, level, lat, lon), ppb
   field = cams.rename(level="pres")                        # levels are in hPa

For each endpoint PYSTILT uses the nearest level, the nearest time, and the
nearest grid cell. An endpoint above the top level takes the top level. An
endpoint outside the field's grid gets no value and is left out of the
average. Longitudes from 0 to 360, as in many global models, are handled.

PYSTILT does not read a global model's files. If its levels vary in space,
as CarbonTracker's do, sample it yourself and pass one value per particle
instead of a field, as a Series indexed by particle number (``particle``).
``sim.particles.stilt.endpoints()`` gives the endpoints as a table for this,
and lair's ``CarbonTracker.sample`` takes that table directly.

The background at a receptor
----------------------------

.. code-block:: python

   rows = []
   for rid in project.receptors.receptor:
       sim = project.simulation(rid, "hrrr")
       bg = sim.background(field)
       enhancement = float(sim.footprint.stilt.enhancement(flux).sum())
       rows.append({"receptor": sim.receptor.id, "background": bg.value,
                    "enhancement": enhancement, "modeled": bg.value + enhancement})

The background is weighted the way the footprint is, with the variant's
transforms: the averaging kernel (including one from a per-receptor
table), pressure weighting, and any lifetime decay. So the background and
the enhancement add. For a tower receptor there are no transforms, and the
value is the plain mean over the particles.

``bg.per_particle`` is the field at each endpoint, and ``bg.weights`` is
each particle's share of the average. Both are indexed by particle. The
spread of ``per_particle`` shows how uniform the background was over the
air the receptor sampled.

What a column's weights cover
-----------------------------

With pressure weighting, the weights sum to the fraction of the column's
air mass that the particles cover, ``(p_sfc - p_top) / p_sfc``. The
enhancement covers the same fraction, which is why the two add. It also
means ``bg.value`` is the background over the receptor's levels only. The
air above the receptor top saw no fluxes, so its share is the field alone.
Take it from the same model with the same pressure weights and averaging
kernel, and add it in your own code.

At the edge of the footprint grid
---------------------------------

A footprint is zero outside its grid, so the air that entered the grid
carried its background with it. ``particles.stilt.grid_exits(grid)`` gives
where each particle first crossed the grid's edge, to sample a field there
instead of at the endpoint, or to count the particles that came in by each
side. The example is 200 particles that drift out of a 2 by 2 degree grid
in a random walk:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import stilt

   rng = np.random.default_rng(0)
   n_particles, n_steps = 200, 20
   steps = rng.normal([-0.05, 0.02], 0.15, size=(n_steps, n_particles, 2))
   path = np.cumsum(steps, axis=0)
   particles = pd.DataFrame(
       {
           "particle": np.tile(np.arange(1, n_particles + 1), n_steps),
           "age": np.repeat(-60.0 * np.arange(1, n_steps + 1), n_particles),
           "lon": -111.0 + path[:, :, 0].ravel(),
           "lat": 41.0 + path[:, :, 1].ravel(),
           "zagl": 100.0,
       }
   )
   grid = stilt.Grid(xmin=-112, xmax=-110, ymin=40, ymax=42, xres=0.1, yres=0.1)
   exits = particles.stilt.grid_exits(grid)
   exits.head()
   exits.edge.value_counts()

There is one row per particle that left, and none for a particle that
stayed inside. The ``edge`` is ``n``, ``e``, ``s`` or ``w``. The position,
height and age are on the edge, interpolated in a straight line between the
particle's last row inside the grid and its first row outside. ``time`` is
there too when the table has it.

- The exit is the first time the particle crossed the edge. A particle that
  came back in, or ended long after at the edge of the meteorology, has the
  same exit; :meth:`~stilt.particles.ParticlesAccessor.endpoints` gives
  where it ended.
- The edge is the outer edge of the grid's cells. A projected grid is a
  rectangle in its projection, and a grid across the dateline is read
  modulo 360.
- A particle whose first row is already outside the grid has no row inside
  to interpolate from. It is reported at that row, on the edge it is
  farthest beyond. The HYSPLIT build bundled with PYSTILT writes no row at
  the release, so a receptor next to the edge can give such a particle.

With transport error
--------------------

Wind errors move the endpoints as well as the path near the surface. Where
the background field has gradients, this adds to the transport error. Pass
the field to ``sim.transport_error``, and it works with each particle's
modeled mole fraction, enhancement plus background:

.. code-block:: python

   err = project.simulation(sim.receptor.id, "hrrr-err")
   result = sim.transport_error(err, flux, background=field)
   result.enhancement - result.background   # the enhancement alone

Choices made here
-----------------

- The endpoint is each particle's row farthest in time from its release.
  A particle that left the domain early ends where it left, which is where
  the background should be sampled.
- Values come from the nearest cell in every dimension, without
  interpolation. Before the first or after the last time or level, the end
  value is used. Outside the grid there is no value.
- A lifetime transform decays the background by the endpoint's age, as it
  decays the enhancement along the trajectory.
- Particles without a value are left out, and the others carry their
  weight. This assumes the missing particles have the same mean as the
  rest. Check ``bg.per_particle`` when a regional field might not cover
  every endpoint.

.. _plume-background:

A background from a satellite swath
-----------------------------------

The sections above take the background from a model field at the
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
~~~~~~~~~~~~

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
       variants={"hrrr": {}},
       n_hours=12,                  # positive: forward in time
       numpar=1000,
   )
   forward.run()

No footprint is needed. The plume only needs the particle tables, which
hold each particle's position at every output step along with its
``time``.

The plume
~~~~~~~~~

Pool the particle rows that fall within a few minutes of the overpass,
across all the forward runs, and outline them:

.. code-block:: python

   from stilt.observations import plume_polygon

   window = (overpass - pd.Timedelta(minutes=3), overpass + pd.Timedelta(minutes=3))
   particles = forward.particles()
   particles = particles[particles["time"].between(*window)]

   plume = plume_polygon(particles["lon"], particles["lat"])
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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
- Each sounding is a point at its center. A pixel that straddles the
  outline counts as inside or outside by where its center falls.
