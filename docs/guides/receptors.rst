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
horizontal position.  STILT releases particles at both ``bottom`` and ``top``
and the footprints are averaged with weighting supplied externally (e.g. an
averaging kernel).

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


Shared interface
----------------

All three classes share the following interface, which lets generic code work
with any receptor type.

``receptor.id``
   A :class:`ReceptorID` string in ``YYYYMMDDHHMM_{location}`` format.
   Used as the unique key throughout the storage and index layers.

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

**MultiPointReceptor** — ``numpar`` particles are distributed as evenly as
possible across the ``n`` release locations.  Choosing ``numpar`` as a
multiple of ``n`` ensures every location receives exactly equal counts.


Loading from CSV
----------------

:func:`read_receptors` loads a receptor CSV and returns a list of the
appropriate subclass objects.  The CSV format follows the R-STILT convention
(``time``, ``longitude``/``long``, ``latitude``/``lati``, ``zagl``/``zmsl``).
Rows grouped under the same ``r_idx`` are assembled into a
:class:`ColumnReceptor` or :class:`MultiPointReceptor` automatically.

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
