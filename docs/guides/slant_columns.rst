Slant Columns
=============

A column instrument that does not look straight up measures air along a
tilted path: a ground-based solar-tracking spectrometer (EM27/SUN, TCCON)
looks at the sun, and an off-nadir satellite sounding looks toward the
satellite. PYSTILT represents that path as a
:class:`~stilt.MultiPointReceptor` whose points step up the line of sight,
runs all of them in **one** simulation, and weights the particles afterwards.
This page walks through the geometry conventions, an EM27/SUN example, and
the satellite variant.

Geometry
--------

:func:`~stilt.observations.slant_points` turns a location, a list of
altitudes, and two angles into ``(longitude, latitude, altitude)`` points.
Each altitude sits at

.. math::

   d = (z - z_\text{anchor}) \tan\theta

metres from the location along the azimuth bearing, where :math:`\theta` is
the zenith angle and :math:`z_\text{anchor}` is the altitude at which the
path passes through the location (the first altitude unless you pass
``anchor=``). The conventions to get right:

``zenith``
   Degrees from the local vertical. For a solar tracker this is the
   **solar zenith angle** at the time of the measurement.

``azimuth``
   Degrees clockwise from north, giving the bearing **from the ground point
   toward the instrument or the sun**. For a solar tracker this is the
   **solar azimuth angle**. Higher points of the receptor are displaced in
   this direction. Satellite products usually report the bearing to the
   satellite under a name like ``sensor_azimuth_angle`` or
   ``viewing_azimuth_angle``; confirm the product's definition, since a few
   report the reverse bearing.

Altitudes
   Use ``altitude_ref="msl"`` for a slant. The path is a straight line in
   geometric height, and terrain-following (AGL) altitudes would bend it.
   Start at the station or surface altitude and end at the top of the air
   you want to resolve, no higher than the meteorology's top. Air above the
   last point is not sampled and belongs to the background (see *Weighting*
   below).

The construction uses a flat tangent plane. At a 3 km column and an 80°
zenith angle the path reaches 17 km horizontally, where the curvature
error is still under a few percent, so this is fine for the release
heights STILT uses.

EM27/SUN example
----------------

An EM27 retrieval (for example PROFFAST output) gives one column value per
spectrum with the solar zenith and azimuth angles at that time. The sun
moves about 15° per hour, so build one receptor per averaging window rather
than one per day. The points come from :func:`~stilt.observations.slant_points`
and the receptor from :meth:`stilt.Receptor.from_points`; no
``Observation`` is needed.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import stilt
   from stilt.observations import slant_points

   LON, LAT = -111.848, 40.766
   STATION_ALT = 1300.0  # m MSL
   ALTITUDES = np.linspace(STATION_ALT, STATION_ALT + 3000.0, 20)

   df = pd.read_csv("em27_2023-07-15.csv", parse_dates=["time"])
   windows = df.set_index("time").resample("10min").mean().dropna()

   receptors = [
       stilt.Receptor.from_points(
           t,
           slant_points(LON, LAT, ALTITUDES, zenith=row.sza, azimuth=row.saa),
           altitude_ref="msl",
       )
       for t, row in windows.iterrows()
   ]
   model.register(receptors=receptors)
   model.run()

Twenty points over 3 km is a typical spacing; HYSPLIT distributes the
particles across the points, so raise ``numpar`` with the point count. Check
the geometry before running by plotting ``receptor.longitudes`` and
``receptor.latitudes`` against ``receptor.altitudes`` for a morning and an
afternoon window: the path should lean east in the morning and west in the
afternoon.

Weighting
---------

A slant receptor's particles start uniformly in height. A column instrument
averages over air mass and applies its own averaging kernel, so weight the
particles before they become a footprint (:doc:`../advanced/transforms`):

.. code-block:: yaml

   footprints:
     column:
       grid: slv
       transforms:
         - kind: pressure_weighting

``pressure_weighting`` is derived from the particles and needs nothing
supplied. The weights sum to the fraction of the atmosphere's mass the
receptor covers, so the footprint is complete for the surface fluxes; the
column above the receptor top is background.

The averaging kernel is different. An EM27 kernel changes with the solar
zenith angle, so each window has its own. Keep it alongside the window and
apply it when generating that window's footprint:

.. code-block:: python

   from stilt.transforms import AveragingKernel

   kernels = [AveragingKernel(levels=row.ak_levels, values=row.ak) for _, row in windows.iterrows()]

   config = model.config.footprints["column"]
   for kernel, receptor in zip(kernels, receptors):
       sim = model.simulations[f"hrrr_{receptor.id}"]
       sim.generate_footprint("column", config, transforms=[kernel], write=True)

If one kernel is a good enough approximation for a campaign, list it under
``transforms`` in ``config.yaml`` instead and every receptor gets it inside
``stilt run``.

Satellite soundings
-------------------

For a satellite product you have many soundings per overpass, each with its
own location, surface altitude, viewing angles, and averaging kernel. That
is what :class:`~stilt.observations.Observation` is for: the reader fills
one per sounding, with the angles in a
:class:`~stilt.observations.ViewingGeometry` and the surface altitude in
``altitude`` (MSL), and :func:`~stilt.observations.build_slant_receptor`
anchors the path at that altitude:

.. code-block:: python

   import numpy as np
   from stilt.observations import build_slant_receptor, group_by_overpass

   def slant(obs):
       return build_slant_receptor(obs, np.linspace(obs.altitude, obs.altitude + 3000, 20))

   for scene in group_by_overpass(observations):
       model.register(receptors=scene.receptors(slant))

The *Adding your own instrument* section of :doc:`../advanced/observations`
has the full reader-builder-transform pattern. Many satellite workflows
neglect the slant and use a :class:`~stilt.ColumnReceptor` instead; at a 20°
viewing angle the top of a 3 km column is only 1 km off the nadir point,
which is within a footprint grid cell for coarse grids.

How the release heights come back
---------------------------------

HYSPLIT does not record which point a particle started from, and vertical
weighting needs each particle's release height. PYSTILT recovers it from
the first row HYSPLIT writes for the particle by matching on height, which
holds to about 20 m for a slant because the points differ in altitude. See
*Release heights for multipoint and slant receptors* in :doc:`receptors`,
including how to run a HYSPLIT build that writes exact release-time rows.
