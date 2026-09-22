Receptors
=========

A *receptor* is a point (or set of points) in space and time that defines
where and when STILT releases particles for a backward run.  PYSTILT provides
three concrete receptor classes that share a common base:

.. currentmodule:: stilt

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - When to use
     - Key attributes
   * - :class:`PointReceptor`
     - Single measurement location (tower, flask, aircraft point)
     - ``longitude``, ``latitude``, ``altitude``
   * - :class:`ColumnReceptor`
     - Vertically-integrated column at one horizontal location (e.g. TCCON)
     - ``longitude``, ``latitude``, ``bottom``, ``top``
   * - :class:`MultiPointReceptor`
     - Slant or multi-angle column (e.g. OCO-2, satellite soundings)
     - ``longitudes``, ``latitudes``, ``altitudes`` (arrays)


PointReceptor
-------------

The most common type.  Use it for any fixed surface site or airborne sample
at a single height.

.. code-block:: python

   import stilt

   receptor = stilt.PointReceptor(
       time="2023-07-15 18:00:00",
       longitude=-111.848,
       latitude=40.766,
       altitude=10.0,          # metres AGL (default) or MSL
   )

``altitude_ref`` defaults to ``"agl"`` (above ground level).  Pass
``altitude_ref="msl"`` for mean-sea-level altitudes.


ColumnReceptor
--------------

Use when the measurement integrates over a vertical range at a single
horizontal position.  HYSPLIT releases particles uniformly in height between
``bottom`` and ``top``.  To turn the result into a mass-weighted column
footprint (with an optional averaging kernel), add ``pressure_weighting``
and ``averaging_kernel`` transforms — see :doc:`/advanced/transforms`.

.. code-block:: python

   receptor = stilt.ColumnReceptor(
       time="2023-07-15 18:00:00",
       longitude=-111.848,
       latitude=40.766,
       bottom=50.0,            # lower altitude (must be < top)
       top=6000.0,
       altitude_ref="msl",
   )

``bottom`` must be strictly less than ``top``; both are validated against
``altitude_ref`` (AGL altitudes must be ≥ 0).


MultiPointReceptor
------------------

Use when the instrument views along a slant path, producing observations at
multiple distinct horizontal *and* vertical positions simultaneously
(e.g. OCO-2 soundings, satellite-retrieved columns with a full
line-of-sight geometry).

.. code-block:: python

   import numpy as np

   receptor = stilt.MultiPointReceptor(
       time="2023-07-15 18:00:00",
       longitudes=np.linspace(-111.85, -111.80, 10),
       latitudes=np.linspace(40.76,  40.80,  10),
       altitudes=np.linspace(0.0,    8000.0, 10),
       altitude_ref="msl",
   )

All three coordinate arrays must have the same length.  The location
identifier is an order-independent SHA-256 hash of the point set, so
reordering the points produces the same simulation ID.

.. warning::

   Every point must have a **distinct horizontal location**.  HYSPLIT
   interprets consecutive starting locations that share a latitude/longitude
   as a single vertical line source and releases particles only between the
   last two heights, so stacking several altitudes at one location would
   silently drop all but the top segment.  :class:`MultiPointReceptor`
   raises ``ValueError`` in that case.  For a vertical column use
   :class:`ColumnReceptor`; for discrete release heights at one location run
   one :class:`PointReceptor` per height (distinct ``r_idx`` in the CSV) and
   combine the footprints afterwards.


Shared interface
----------------

All three classes share the following interface, which lets generic code work
with any receptor type.

``receptor.id``
   A :class:`ReceptorID` string in ``YYYYMMDDHHMM_{location}`` format.
   Combined with a met name it forms the simulation id and output keys.

``receptor.time``
   A naive UTC :class:`datetime.datetime`.

``receptor.altitude_ref``
   ``"agl"`` or ``"msl"``.

``for lat, lon, alt in receptor:``
   Iterates over ``(latitude, longitude, altitude)`` tuples — one for
   :class:`PointReceptor`, two for :class:`ColumnReceptor`, *n* for
   :class:`MultiPointReceptor`.  This is the primary way generic code
   reads coordinates without branching on type.

``receptor.geometry``
   A shapely geometry (``Point``, ``LineString``, or ``MultiPoint``).

``receptor.plot.map()``
   Quick map visualisation.

``receptor.to_dict()`` / ``Receptor.from_dict(d)``
   JSON-round-trippable serialisation.  The dict always contains a ``"type"``
   key (``"PointReceptor"``, ``"ColumnReceptor"``, or ``"MultiPointReceptor"``)
   so ``Receptor.from_dict`` can reconstruct the correct subclass.


Particle distribution
---------------------

``numpar`` controls the total number of particles released per simulation.
How those particles are distributed depends on receptor type.

**PointReceptor** — all ``numpar`` particles are released from the single location.

**ColumnReceptor** — particles are evenly spread across the column from
``bottom`` to ``top``.

**MultiPointReceptor** — ``numpar`` particles are distributed across the
``n`` release locations.  HYSPLIT rounds the per-location count *up* and stops
when ``numpar`` runs out, so the last location can come up short (200
particles over 10 locations gives 21 to each of the first nine and 11 to the
last).  Release locations must be horizontally distinct (see the warning
above).


Release heights for multipoint and slant receptors
---------------------------------------------------

HYSPLIT does not record which release location a particle came from, so PYSTILT
recovers each particle's release height (``xhgt``, which vertical weighting
depends on) from the first row HYSPLIT writes for it.  With the bundled build
that row is one timestep *after* release, by which time the wind has carried
the particle a few hundred metres — further than the points of a slant column
are apart.  PYSTILT therefore:

1. uses release-time (``t = 0``) rows when the HYSPLIT build writes them, which
   makes the recovery exact;
2. otherwise matches on **height** when the release altitudes are all distinct,
   as they always are for a slant column.  Height drifts about thirty times less
   than horizontal position over one step, and this recovers release heights to
   within roughly 20 m;
3. otherwise matches on horizontal position, and **warns** when the locations
   are closer than 1 km, since the result cannot then be trusted.

A HYSPLIT modification that writes ``t = 0`` rows has been submitted to NOAA
ARL.  Until it is in a published build, you can run your own:

.. code-block:: yaml

   # config.yaml
   exe_dir: /path/to/hysplit/exec    # directory containing hycs_std

The setting is stored with the trajectory parameters, so outputs record which
build produced them.


Loading from CSV
----------------

:func:`read_receptors` loads a receptor CSV and returns a list of the
appropriate subclass objects.  The CSV format follows the R-STILT convention
(``time``, ``longitude``/``long``, ``latitude``/``lati``, ``zagl``/``zmsl``).
Rows grouped under the same ``r_idx`` are assembled into a
:class:`ColumnReceptor` or :class:`MultiPointReceptor` automatically.
All rows in a group must share the same ``time`` (a receptor releases at a
single time); observations with different times need distinct ``r_idx``
values so that each becomes its own simulation.

.. code-block:: python

   receptors = stilt.read_receptors("receptors.csv")


Type dispatch
-------------

Use :func:`isinstance` to branch on receptor type in generic code — never
inspect string attributes:

.. code-block:: python

   from stilt import ColumnReceptor, MultiPointReceptor, PointReceptor

   if isinstance(receptor, PointReceptor):
       print(f"Single point at {receptor.altitude} m")
   elif isinstance(receptor, ColumnReceptor):
       print(f"Column from {receptor.bottom} to {receptor.top} m")
   elif isinstance(receptor, MultiPointReceptor):
       print(f"Multi-point with {len(receptor)} locations")

All three are subclasses of :class:`Receptor`, so
``isinstance(receptor, stilt.Receptor)`` is always ``True`` for any receptor
object.


Smart constructor
-----------------

:meth:`Receptor.from_points` picks the right subclass from a list of
``(longitude, latitude, altitude)`` tuples:

- One tuple → :class:`PointReceptor`
- Two tuples at the same horizontal location → :class:`ColumnReceptor`
  (``bottom``/``top`` are sorted automatically)
- Anything else → :class:`MultiPointReceptor`

.. code-block:: python

   r = stilt.Receptor.from_points(
       time="2023-07-15 18:00:00",
       points=[(-111.848, 40.766, 10.0)],
   )
   # → PointReceptor
