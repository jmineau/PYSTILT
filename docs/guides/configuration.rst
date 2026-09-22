Configuration
=============

A project's settings live in ``config.yaml`` in the project folder: which
meteorology to use, which footprints to make, and how to run HYSPLIT. This
page covers the settings most projects need. The
:doc:`../reference/configuration` lists every option.

A typical config.yaml
---------------------

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

   footprints:
     slv:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

   n_hours: -24
   numpar: 500

``stilt init`` writes a commented starter file. When you build a
:class:`~stilt.Model` in Python, the same settings are written to
``config.yaml`` the first time you run it.

The settings most people change
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Setting
     - What it does
     - Typical value
   * - ``mets``
     - Where the meteorology files are and how they are named. The name you
       give each one (``hrrr`` above) goes into every simulation ID. See
       :doc:`meteorology`.
     - one entry
   * - ``footprints``
     - The map grid(s) to calculate footprints on: longitude range
       (``xmin``/``xmax``), latitude range (``ymin``/``ymax``), and cell size
       in degrees (``xres``/``yres``). Each footprint gets a name you choose
       (``slv`` above).
     - 0.01° (about 1 km) for a city; 0.1° for a region
   * - ``n_hours``
     - How many hours to follow particles. Negative is backward in time,
       which footprints need; positive runs forward from the receptor
       (:doc:`plume_background`).
     - ``-24`` to ``-72``
   * - ``numpar``
     - Particles released per simulation. More is smoother and slower.
     - ``200`` to ``1000``
   * - ``execution``
     - Where to run. Leave it out to run on your own computer. See
       :doc:`execution/index`.
     - (omit)
   * - ``skip_existing``
     - Skip simulations whose outputs already exist.
     - ``true`` (default)

Footprint options
-----------------

Besides the grid, each footprint accepts:

``smooth_factor``
   Scales the Gaussian smoothing applied to each particle's influence.
   ``1.0`` (the default) is standard STILT; smaller values smooth less.

``time_integrate``
   ``true`` sums the footprint over time into a single map, making smaller
   files. The default ``false`` keeps an hourly time dimension.

``transforms``
   Particle weighting steps, mainly for column measurements. See
   :doc:`../advanced/transforms`.

Several footprints at once
--------------------------

You can make more than one footprint from the same particles, for example a
fine grid over the city and a coarse one over the region:

.. code-block:: yaml

   footprints:
     city:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

     regional:
       xmin: -125.0
       xmax: -100.0
       ymin: 30.0
       ymax: 50.0
       xres: 0.1
       yres: 0.1

Each footprint gets its own file. Adding a footprint to an existing project
and running again calculates just the new footprint for every simulation.

The grid can also be written under a ``grid:`` key. The two forms mean the
same thing:

.. code-block:: yaml

   footprints:
     city:
       grid:
         xmin: -114.0
         xmax: -111.0
         ymin: 39.0
         ymax: 42.0
         xres: 0.01
         yres: 0.01

Footprints for shapefiles, hexagons, or point sources
-----------------------------------------------------

If you will add footprints up over irregular areas, such as counties from a
shapefile, H3 hexagons, or small windows around point sources, you can name
those areas instead of a grid. PYSTILT then picks a grid fine enough to
resolve them (:meth:`stilt.Grid.from_geometry`). This needs the ``geometry``
extra:

.. code-block:: yaml

   footprints:
     counties:
       geometry:
         kind: file          # shapefile / GeoPackage / GeoJSON (needs geopandas)
         path: counties.shp
         ids: NAME           # attribute column used as cell ids
     hexes:
       geometry:
         kind: h3            # needs the h3 package
         resolution: 8
         bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}
       cells_per_target: 4   # native cells across the smallest hexagon (default)
     sources:
       geometry:
         kind: windows
         coords: [[-111.97, 40.515], [-112.015, 40.779]]
         size: 0.01
         ids: [landfill, wwtp]

The derived ``grid`` is written back into the config, so the geometry object
is never needed to read a stored footprint.  Give both ``grid`` and
``geometry`` to pin the raster explicitly; ``geometry`` is then kept as a
record and ``config.geometry.build()`` returns the :class:`stilt.Mesh` to
aggregate onto.  A content hash of the built geometry (``geometry_hash``) is
stored with the config and in each footprint file; ``Footprint.aggregate``
warns if the mesh it is handed no longer matches, which catches a shapefile
edited after the footprints were computed.

.. note::

   :class:`stilt.Zones` (super-cells) have no YAML form yet; build them in
   code with ``Zones.from_labels(base, labels)``.  A ``kind: zones`` spec
   would need a label source (an attribute column, a CSV keyed by cell id, or
   polygons assigned by cell centre) and will be added once a project needs
   its zoning to live in the config.

Running on a cluster
--------------------

To run on Slurm instead of your own computer, add an ``execution`` section:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 100          # number of Slurm array tasks
     account: my-account
     partition: my-partition
     time: "02:00:00"
     mem: 8G

Everything besides ``backend``, ``n_workers``, ``cpus_per_task``,
``array_parallelism``, and ``setup`` is passed to ``sbatch``. See
:doc:`execution/slurm`.

The same settings in Python
---------------------------

Every ``config.yaml`` key can be passed to :class:`~stilt.Model` directly,
as plain dictionaries:

.. code-block:: python

   import stilt

   model = stilt.Model(
       project="./my_project",
       mets={
           "hrrr": {
               "directory": "/data/arl/hrrr",
               "file_format": "%Y%m%d_%H",
               "file_tres": "6h",
           }
       },
       footprints={
           "slv": {
               "xmin": -114.0, "xmax": -111.0,
               "ymin": 39.0, "ymax": 42.0,
               "xres": 0.01, "yres": 0.01,
           }
       },
       n_hours=-24,
       numpar=500,
   )

or as typed objects, which your editor can autocomplete and check:

.. code-block:: python

   config = stilt.ModelConfig(
       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/arl/hrrr",
               file_format="%Y%m%d_%H",
               file_tres="6h",
           )
       },
       footprints={
           "slv": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-114.0, xmax=-111.0,
                   ymin=39.0, ymax=42.0,
                   xres=0.01, yres=0.01,
               )
           )
       },
       n_hours=-24,
       numpar=500,
   )
   model = stilt.Model(project="./my_project", config=config)

Advanced HYSPLIT settings
-------------------------

``config.yaml`` also accepts HYSPLIT and STILT's lower-level settings
(turbulence, time step, output variables, and so on) under the same names
STILT-R uses. Most projects never change them; the
:doc:`../reference/configuration` describes each one.

Typos are caught early: an unknown key in ``config.yaml`` is an error when the
file is loaded, before anything runs.
