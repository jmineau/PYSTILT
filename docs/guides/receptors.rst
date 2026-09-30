Receptors
=========

A :term:`receptor` is the place and time of a measurement. STILT releases
particles there and follows them backward in time. Pick the receptor type
that matches how your instrument samples the air.

.. currentmodule:: stilt

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Receptor type
     - Use it for
     - You give it
   * - :class:`PointReceptor`
     - An instrument sampling at one point, such as a tower inlet, a flask,
       or an aircraft or vehicle measurement
     - ``longitude``, ``latitude``, ``altitude``
   * - :class:`ColumnReceptor`
     - An instrument measuring the vertical column above one spot, such as a
       ground-based spectrometer (TCCON, EM27/SUN)
     - ``longitude``, ``latitude``, ``bottom``, ``top``
   * - :class:`MultiPointReceptor`
     - A slanted line of sight, such as a solar-tracking spectrometer or an
       off-nadir satellite sounding (see :doc:`slant_columns`), or any set of
       release points at one time
     - ``longitudes``, ``latitudes``, ``altitudes`` (arrays)

Every receptor also needs a ``time`` in UTC. A timezone-aware time is
converted to UTC. Altitudes are in metres above ground level. Pass
``altitude_ref="msl"`` to give them above mean sea level instead.

PointReceptor
-------------

This is the most common type. Use it for a fixed surface site or an airborne
sample at one height.

.. code-block:: python

   import stilt

   receptor = stilt.PointReceptor(
       time="2023-07-15 18:00:00",
       longitude=-111.848,
       latitude=40.766,
       altitude=10.0,          # metres above ground
   )


ColumnReceptor
--------------

Use it when the measurement covers a range of heights above one spot.
HYSPLIT releases the particles evenly in height between ``bottom`` and
``top``.

.. code-block:: python

   receptor = stilt.ColumnReceptor(
       time="2023-07-15 18:00:00",
       longitude=-111.848,
       latitude=40.766,
       bottom=50.0,
       top=6000.0,
       altitude_ref="msl",
   )

``bottom`` must be below ``top``. Above-ground altitudes cannot be negative.

A column footprint usually needs weighting by pressure, and often by the
instrument's averaging kernel. Add the ``pressure_weighting`` and
``averaging_kernel`` transforms for that (see :doc:`/advanced/transforms`).


MultiPointReceptor
------------------

Use it when the instrument looks along a slanted path, so the air it
samples is spread out both horizontally and vertically. OCO-2 soundings and
other satellite views with a full line-of-sight geometry are examples.

.. code-block:: python

   import numpy as np

   receptor = stilt.MultiPointReceptor(
       time="2023-07-15 18:00:00",
       longitudes=np.linspace(-111.85, -111.80, 10),
       latitudes=np.linspace(40.76,  40.80,  10),
       altitudes=np.linspace(0.0,    8000.0, 10),
       altitude_ref="msl",
   )

The three arrays must have the same length. The receptor's ID is built from
a hash of its points, so listing the same points in another order gives the
same ID. To build the points from viewing angles instead of by hand, see
:doc:`slant_columns`.

.. warning::

   Every point must have its own horizontal location. HYSPLIT treats
   consecutive release points at the same latitude and longitude as one
   vertical line source, and releases particles only between the last two
   heights. Stacking several heights at one location would silently drop all
   but the top segment, so :class:`MultiPointReceptor` raises
   ``ValueError`` instead.

   For a vertical column, use :class:`ColumnReceptor`. For several separate
   heights at one location, run one :class:`PointReceptor` per height (a
   distinct ``r_idx`` in the CSV) and combine the footprints afterward.


Loading receptors from a CSV file
---------------------------------

For more than a handful of receptors, keep them in a CSV file. The simplest
form has one row per point receptor:

.. code-block:: text

   time,longitude,latitude,altitude
   2023-07-15 18:00:00,-111.848,40.766,10
   2023-07-15 19:00:00,-111.848,40.766,10

The column names from STILT-R also work. Use ``long`` or ``lon`` for
longitude, ``lati`` or ``lat`` for latitude, and ``zagl`` or ``zmsl`` for
altitude. A ``zmsl`` column means the altitudes are above sea level. To mix
the two in one file, add an ``altitude_ref`` column with ``agl`` or ``msl``
on each row.

To build a column or multipoint receptor, give its rows the same ``r_idx``
value. Rows that share an ``r_idx`` become one receptor, so they must also
share the same ``time``:

.. code-block:: text

   time,longitude,latitude,altitude,r_idx
   2023-07-15 18:00:00,-111.848,40.766,0,1
   2023-07-15 18:00:00,-111.848,40.766,3000,1

Two rows at the same location make a :class:`ColumnReceptor`. Rows at
different locations make a :class:`MultiPointReceptor`.

Any other column is kept on the receptor in ``attrs``. A ``scene`` or
``site`` label you add to the file is then available when you select
results (see :doc:`outputs`):

.. code-block:: python

   model.simulations.sel(scene="A")
   model.receptors.sel(site=["WBB", "UOU"])

For a receptor built from several rows, the first row's labels are used.

``model.receptors.to_frame()`` gives the receptors back as one table with a
row per release point, in the same columns as the file.

A project's ``receptors.csv`` is read automatically. To load another CSV,
use :func:`read_receptors`, or pass the path to the model:

.. code-block:: python

   receptors = stilt.read_receptors("my_receptors.csv")
   model = stilt.Model(project="./my_project", receptors="my_receptors.csv")

Receptors you add to a project later are appended to its ``receptors.csv``
using the file's own columns (see :doc:`project_layout`). If a new receptor's
heights are above sea level and the file has no ``altitude_ref`` column, the
column is added.


How particles are released
--------------------------

``numpar`` is the number of particles released in each simulation.

- A :class:`PointReceptor` releases all of them from its one location.
- A :class:`ColumnReceptor` spreads them evenly from ``bottom`` to ``top``.
- A :class:`MultiPointReceptor` divides them among its locations. HYSPLIT
  rounds the count per location up and stops when ``numpar`` runs out, so
  the last location can get fewer. With 200 particles over 10 locations,
  the first nine get 21 each and the last gets 11.


Advanced: release heights for multipoint and slant receptors
------------------------------------------------------------

Vertical weighting needs each particle's release height (``xhgt``). HYSPLIT
does not record which release point a particle came from, so PYSTILT works
it out from the first row HYSPLIT writes for the particle. With the bundled
HYSPLIT build, that row comes one timestep after release. By then the wind
has moved the particle a few hundred metres, which can be more than the
spacing between the points of a slant column.

PYSTILT recovers release heights in this order:

1. If the HYSPLIT build writes rows at release time (``t = 0``), it uses
   them. The result is exact.
2. If the release altitudes are all different, as they are in a slant
   column, it matches particles by height. Height changes much less than
   horizontal position over one timestep, so this is accurate to about
   20 m.
3. Otherwise it matches by horizontal position. It warns when the release
   points are closer than 1 km, because the match cannot be trusted there.

A HYSPLIT change that writes ``t = 0`` rows has been sent to NOAA ARL. Until
a published build includes it, you can point PYSTILT at your own build:

.. code-block:: yaml

   # config.yaml
   exe_dir: /path/to/hysplit/exec    # directory containing hycs_std

The setting is saved with each trajectory, so you can tell which build
produced it.


Working with receptors in code
------------------------------

The rest of this page is for code that handles any type of receptor.

Shared interface
~~~~~~~~~~~~~~~~

All three classes have these members:

``receptor.id``
   A string of the form ``YYYYMMDDHHMM_{location}``. It names the receptor's
   result files (see :doc:`project_layout`), so two different receptors may
   not share one. ``receptor.location_id`` is the location part.

``receptor.time``
   The release time, as a :class:`datetime.datetime` in UTC with no
   timezone attached.

``receptor.altitude_ref``
   ``"agl"`` or ``"msl"``.

``receptor.attrs``
   A dictionary of extra labels, such as the extra columns of
   ``receptors.csv``. They are not part of the receptor's ID.

``receptor.coords()``
   The receptor's points as a list of ``(latitude, longitude, altitude)``.
   A :class:`PointReceptor` has one point, a :class:`ColumnReceptor` two
   (bottom and top), and a :class:`MultiPointReceptor` one per location.
   This lets code read coordinates without checking the type.

``receptor.geometry``
   A shapely geometry: a ``Point``, ``LineString``, or ``MultiPoint``.

``receptor.plot.map()``
   A quick map of the receptor.

``receptor.to_dict()`` and ``Receptor.from_dict(d)``
   Convert to and from a JSON-friendly dictionary. The dictionary has a
   ``"kind"`` key (``"point"``, ``"column"``, or ``"multipoint"``), which
   ``Receptor.from_dict`` uses to rebuild the right class.

Receptors are frozen. To change one, make a copy with
``receptor.model_copy(update={"altitude": 20})``. Every argument is given by
name, so a longitude cannot be passed as a latitude by mistake.


Checking the type
~~~~~~~~~~~~~~~~~

Use :func:`isinstance` to branch on the receptor type:

.. code-block:: python

   from stilt import ColumnReceptor, MultiPointReceptor, PointReceptor

   if isinstance(receptor, PointReceptor):
       print(f"Single point at {receptor.altitude} m")
   elif isinstance(receptor, ColumnReceptor):
       print(f"Column from {receptor.bottom} to {receptor.top} m")
   elif isinstance(receptor, MultiPointReceptor):
       print(f"Multi-point with {len(receptor.coords())} locations")

All three are subclasses of :class:`Receptor`.


Building a receptor from a list of points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`Receptor.from_points` takes a list of ``(longitude, latitude,
altitude)`` tuples and returns the matching type:

- one tuple gives a :class:`PointReceptor`;
- two tuples at the same location give a :class:`ColumnReceptor`, with
  ``bottom`` and ``top`` in the right order;
- anything else gives a :class:`MultiPointReceptor`.

.. code-block:: python

   r = stilt.Receptor.from_points(
       time="2023-07-15 18:00:00",
       points=[(-111.848, 40.766, 10.0)],
   )
   # PointReceptor
