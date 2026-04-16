.. _column_receptors:

Column and Profile Receptors
============================

Standard STILT simulations release particles from a single fixed point. When
working with satellite retrievals or vertical profiles, the measurement
integrates through a column of the atmosphere. PYSTILT supports two receptor
types for this:

* **Column receptor** (``Receptor.kind == "column"``) — a fixed lat/lon with
  lower and upper release bounds at one lat/lon. PYSTILT later reconstructs a
  release-height column for those particles by spreading particle indices
  evenly between the bottom and top heights.
* **Multipoint receptor** (``Receptor.kind == "multipoint"``) — particles
  divided evenly among a discrete set of explicit (lon, lat, altitude) points.
  This is the basis for slant-column geometries (e.g., OCO-2 line of sight).

See `Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_ (X-STILT) for
the scientific motivation for column-integrated footprints.

.. note::

   **Vertical coordinate uncertainty.** When you specify receptor heights as
   zagl (height above ground level), it is unclear exactly how HYSPLIT maps
   those values to its internal vertical coordinate. HYSPLIT uses its own DEM
   and sigma-z grid, so the specified zagl may not correspond precisely to the
   geometric height in the model. This is an unresolved upstream issue.


Column Receptors
-----------------

:meth:`~stilt.Receptor.from_column` creates a receptor at a fixed lat/lon with
column bounds ``bottom`` and ``top``. When trajectory output is normalized into
``Trajectories.data``, PYSTILT reconstructs an ``xhgt`` column by spreading
particle indices evenly between those bounds.

This is the simplest way to represent a vertical atmospheric column:

.. code-block:: python

   import stilt
   import pandas as pd

   column_receptor = stilt.Receptor.from_column(
       time=pd.Timestamp("2023-07-15 21:00", tz="UTC"),
       longitude=-111.85,
       latitude=40.77,
       bottom=0,       # height AGL (m)
       top=3000,       # height AGL (m)
   )

   print(column_receptor.kind)    # "column"
   print(column_receptor.bottom)  # 0.0
   print(column_receptor.top)     # 3000.0


Slant Receptors
----------------

For slant satellite observations, the preferred API is now the observation
layer rather than a direct ``Receptor`` constructor. A physical line of sight
is defined with:

- :class:`stilt.observations.ViewingGeometry`
- :class:`stilt.observations.LineOfSight`
- :func:`stilt.observations.build_slant_receptor`

This keeps LOS geometry in the observation layer while still producing the
same transport-level result: a ``multipoint`` :class:`~stilt.Receptor`.

.. code-block:: python

   import pandas as pd
   from stilt.observations import (
       LineOfSight,
       Observation,
       ViewingGeometry,
       build_slant_receptor,
   )

   observation = Observation(
       sensor="oco2",
       species="xco2",
       time=pd.Timestamp("2023-07-15 21:00", tz="UTC"),
       latitude=40.77,
       longitude=-111.85,
       viewing=ViewingGeometry(
           viewing_zenith_angle=25.0,
           viewing_azimuth_angle=90.0,
       ),
       line_of_sight=LineOfSight(
           start_altitude=1500.0,
           end_altitude=5000.0,
           count=20,
           altitude_ref="msl",
       ),
   )

   slant_receptor = build_slant_receptor(observation)

   print(slant_receptor.kind)         # "multipoint"
   print(len(slant_receptor))         # 20
   print(slant_receptor.altitude_ref) # "msl"

If ``anchor_altitude`` is omitted, PYSTILT uses ``Observation.altitude`` when
available. For MSL LOS definitions, omitted ``surface_altitude`` also falls
back to an MSL ``Observation.altitude``, so below-surface levels are clipped
automatically.

The low-level :meth:`~stilt.Receptor.from_points` constructor remains useful
for explicit custom release geometries, but slant-column LOS construction
should live in the observation layer.


Multi-Point Receptors from Explicit Points
-------------------------------------------

:meth:`~stilt.Receptor.from_points` creates a multipoint receptor from an
arbitrary list of (lon, lat, altitude) tuples. Particles are divided evenly among
the points, with any remainder assigned to the earliest points first:

.. code-block:: python

   import numpy as np

   altitudes = np.linspace(0, 5000, 20)   # 20 levels, 0–5000 m AGL
   points = [(-111.85, 40.77, z) for z in altitudes]

   receptor = stilt.Receptor.from_points(
       time=pd.Timestamp("2023-07-15 21:00", tz="UTC"),
       points=points,
   )


Reading Column/Multipoint Receptors from CSV
---------------------------------------------

The :func:`~stilt.read_receptors` function supports a ``group`` column that
assigns individual point rows to a named multipoint receptor:

.. code-block:: text

   time,lat,lon,altitude,altitude_ref,group
   2023-07-15 21:00:00+00:00,40.77,-111.85,0,agl,col_01
   2023-07-15 21:00:00+00:00,40.77,-111.85,500,agl,col_01
   2023-07-15 21:00:00+00:00,40.77,-111.85,1000,agl,col_01
   2023-07-15 21:00:00+00:00,40.77,-111.85,1500,agl,col_01

All rows with the same ``group`` are combined into one
:class:`~stilt.Receptor`. If all points share the same lon/lat, the receptor
is classified as ``"column"``; otherwise it is ``"multipoint"``.


Running Column Simulations
---------------------------

Column and multipoint receptors are passed to :class:`~stilt.Model` exactly
like point receptors:

.. code-block:: python

   model = stilt.Model(
       project="/path/to/column_project",
       receptors=[column_receptor],
       config=config,
   )
   model.run(mets=["hrrr"], footprints=["default"])


Applying Vertical Weighting
----------------------------

For column-integrated footprints weighted by an averaging kernel (AK) and
pressure weighting function (PWF), use a footprint transform to scale particle
sensitivity before the footprint is gridded. See :doc:`custom_hooks` for the
runtime transform API.

.. code-block:: python

   import numpy as np
   from stilt.config import VerticalOperatorTransformSpec

   ak  = np.array([...])   # averaging kernel, length = n_levels
   pwf = np.array([...])   # pressure weighting function, length = n_levels
   transform = VerticalOperatorTransformSpec(
       kind="vertical_operator",
       mode="ak_pwf",
       levels=np.linspace(receptor.bottom, receptor.top, len(ak)).tolist(),
       values=(ak * pwf).tolist(),
       coordinate="xhgt",
   )

   config = stilt.ModelConfig(
       ...,
       footprints={
           "column": stilt.FootprintConfig(
               grid=grid,
               transforms=[transform],
           )
       },
   )

.. seealso::

   :doc:`custom_hooks`: full particle-transform API reference.

   `X-STILT (Wu et al. 2018) <https://doi.org/10.5194/gmd-11-4843-2018>`_:
   column-receptor methodology for satellite retrievals.

.. seealso::

   :doc:`../tutorials/column_satellite`: complete tutorial using OCO-2/TROPOMI-style column receptors.
