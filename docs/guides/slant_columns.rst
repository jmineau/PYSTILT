Slant Columns
=============

A column instrument that does not look straight up measures the air along
a tilted path. A ground-based solar tracker (EM27/SUN, TCCON) looks at the
sun. An off-nadir satellite sounding looks toward the satellite. PYSTILT
models the path as a :class:`~stilt.MultiPointReceptor` whose points step up
the line of sight. All the points run in one simulation, and the particles
are weighted afterwards.

This page covers the geometry, an EM27/SUN example, and satellite
soundings.

Geometry
--------

:func:`~stilt.observations.slant_points` turns a location, a list of
altitudes, and two angles into ``(longitude, latitude, altitude)`` points.
Each altitude :math:`z` is placed a horizontal distance

.. math::

   d = (z - z_\text{anchor}) \tan\theta

from the location, along the azimuth bearing. :math:`\theta` is the zenith
angle. :math:`z_\text{anchor}` is the altitude where the path passes through
the location. It is the first altitude unless you pass ``anchor=``.

``zenith``
   Degrees from the local vertical. For a solar tracker, use the solar
   zenith angle at the time of the measurement.

``azimuth``
   Degrees clockwise from north, pointing from the ground toward the
   instrument or the sun. Higher points are moved in this direction. For a
   solar tracker, use the solar azimuth angle. Satellite products usually
   give the bearing to the satellite as ``sensor_azimuth_angle`` or
   ``viewing_azimuth_angle``. A few products give the reverse bearing, so
   check the product's definition.

Altitudes
   Give the altitudes above mean sea level and build the receptor with
   ``altitude_ref="msl"``. The path is a straight line in height above sea
   level, and heights above ground would bend it with the terrain. Start at
   the station or surface altitude. End at the top of the air you want to
   resolve, and no higher than the top of the meteorology. Air above the
   last point is not sampled. It belongs to the background (see
   `Weighting`_ below).

The points are laid out on a flat plane tangent to the Earth. A 3 km column
at an 80° zenith angle reaches 17 km sideways. Even there, ignoring the
Earth's curvature moves the top point by less than one percent of the
column height.

EM27/SUN example
----------------

An EM27/SUN retrieval gives one column value per spectrum, with the solar
zenith and azimuth angles at that time. :func:`~stilt.observations.read_ggg_oof`
reads a day of these (see :doc:`readers`). The sun moves about 15° per
hour, so build one receptor per averaging window, not one per day:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import stilt
   from stilt.observations import read_ggg_oof, slant_points

   project = stilt.Project("./em27")      # made with stilt init or Project.init

   df = read_ggg_oof("ha20230715.vav.ada.aia.oof", "xch4")
   df = df[df.good]
   windows = (
       df.set_index("time")[["longitude", "latitude", "surface_altitude", "zenith", "azimuth"]]
       .resample("10min").mean().dropna()
   )

   receptors = [
       stilt.Receptor.from_points(
           t,
           slant_points(
               w.longitude, w.latitude,
               np.linspace(w.surface_altitude, w.surface_altitude + 3000.0, 20),
               zenith=w.zenith, azimuth=w.azimuth,
           ),
           altitude_ref="msl",
       )
       for t, w in windows.iterrows()
   ]
   project.add_receptors(receptors)
   project.run()

Twenty points over 3 km is a typical spacing. HYSPLIT splits ``numpar``
among the points, so raise ``numpar`` when you add points.

Also add ``zsfc`` to ``varsiwant`` in ``config.yaml``. PYSTILT needs it to
work out which point each particle started from (see `How the release
heights come back`_).

Before running, check the geometry. Plot ``receptor.longitudes`` and
``receptor.latitudes`` against ``receptor.altitudes`` for a morning window
and an afternoon window. The path should lean east in the morning and west
in the afternoon.

Weighting
---------

A column instrument averages over air mass and applies its own averaging
kernel. The particles of a slant receptor start evenly spaced in height, so
weight them before they become a footprint (see
:doc:`../advanced/transforms`):

.. code-block:: yaml

   transforms:
     - kind: pressure_weighting

``pressure_weighting`` works everything out from the particles, so you give
it no inputs. Each point of the slant stands for its share of the air
mass, split evenly among the particles released there. The weights add up
to the fraction of the atmosphere's mass that the receptor covers. The air
above the receptor top is part of the background. For an MSL receptor the
transform needs ``zsfc`` in ``varsiwant``, as the release-height matching
does.

Each window needs its own averaging kernel, because an EM27/SUN kernel
changes with the solar zenith angle. A ``.oof`` file does not have the
kernels. Take them from the run's ``*.private.nc`` with
:func:`~stilt.observations.read_ggg_netcdf`, which expands GGG's kernel
table for each spectrum. You can also use a site table keyed by solar
zenith angle, as PROFFAST provides. Write the kernels into the project with
:func:`~stilt.transforms.averaging_kernel_table`. The ``averaging_kernel``
transform then looks up each receptor's kernel:

.. code-block:: python

   from stilt.observations import read_ggg_netcdf
   from stilt.transforms import averaging_kernel_table

   ak = read_ggg_netcdf("ha20230715_20230715.private.nc", "xch4").set_index("time")
   # one kernel per window: the spectrum nearest the window's midpoint
   nearest = ak.index.get_indexer(windows.index + pd.Timedelta("5min"), method="nearest")
   kernels = [ak.ak.iloc[i] for i in nearest]
   table = averaging_kernel_table(receptors, levels=ak.ak_pressure.iloc[0], values=kernels)
   table.to_parquet(project.directory / "kernels.parquet")

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       table: kernels.parquet
       coordinate: pres
     - kind: pressure_weighting

By default the kernel's ``levels`` are release heights in the receptor's
vertical reference, which is metres above sea level for a slant. The GGG
kernel is given on pressures, so set ``coordinate: pres``. If one kernel is
close enough for a whole campaign, give it inline as ``levels`` and
``values`` instead of a table.

Satellite soundings
-------------------

A satellite overpass gives many soundings. Each has its own location,
surface altitude, viewing angles, and averaging kernel. Build one slant
receptor per row of the product table, starting at each sounding's surface
altitude:

.. code-block:: python

   import numpy as np
   import stilt
   from stilt.observations import slant_points

   receptors = [
       stilt.Receptor.from_points(
           row.time,
           slant_points(
               row.longitude, row.latitude,
               np.linspace(row.surface_altitude, row.surface_altitude + 3000, 20),
               zenith=row.zenith, azimuth=row.azimuth,
           ),
           altitude_ref="msl",
       )
       for row in df.itertuples()
   ]
   project.add_receptors(receptors)

Many satellite workflows ignore the slant and use a
:class:`~stilt.ColumnReceptor` instead. At a 20° viewing angle the top of a
3 km column is only about 1 km from the nadir point, which is within one
cell of a coarse footprint grid.

Altitudes from the retrieval's pressure levels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retrievals define their vertical grid in pressure. OCO-2 gives
``pressure_levels`` for each sounding. TROPOMI's layer edges are
``surface_pressure`` minus multiples of ``pressure_interval``. To put the
slant points on those levels instead of an even spacing in height, convert
them with :func:`~stilt.observations.pressure_altitudes`. It uses the
hypsometric equation, starting from the sounding's surface pressure and
surface altitude:

.. code-block:: python

   from stilt.observations import pressure_altitudes, slant_points

   levels = pressure_altitudes(
       row.pressure_levels,                   # hPa, in the product's order
       surface_pressure=row.surface_pressure,  # hPa
       surface_altitude=row.surface_altitude,  # m MSL
       top=row.surface_altitude + 3000.0,      # drop levels above this
   )
   points = slant_points(
       row.longitude, row.latitude, levels, zenith=row.zenith, azimuth=row.azimuth
   )

The altitudes come back sorted from the surface upward, whatever order the
product lists its levels in. The first one anchors the path at the
sounding's location. Levels below the surface are dropped, and so are
levels above ``top``. The top of the meteorology is a sensible value for
``top``.

Without a temperature, the conversion uses the standard-atmosphere lapse
rate (6.5 K/km) from the surface. This matches the U.S. Standard Atmosphere
below 11 km. If the retrieval or its prior gives a temperature profile,
pass it as ``temperature=`` with one value per level, in kelvin.

HYSPLIT still sees these points as heights. The weighting in `Weighting`_
applies unchanged. :doc:`../advanced/observations` shows how to choose
which soundings to run.

How the release heights come back
---------------------------------

The weighting needs each particle's release height, but HYSPLIT does not
record which point a particle started from. PYSTILT works it out from the
first row HYSPLIT writes for the particle, by matching its height to the
nearest point. This works because the points of a slant are at different
altitudes.

For an MSL receptor the match uses ``zagl + zsfc``, so ``zsfc`` must be in
``varsiwant``. It is not there by default. Without it, PYSTILT falls back to
matching on horizontal position. That is unreliable for points closer than
1 km, and PYSTILT warns when it happens.

The section on release heights in :doc:`receptors` has the details. It
also shows how to run a HYSPLIT build that writes rows at the release time,
which makes the match exact.
