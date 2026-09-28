Configuration
=============

A project's settings live in ``config.yaml`` in the project folder: which
meteorology to use, which footprint to make, and how to run HYSPLIT. This
page covers the settings most projects need, and `Variants`_ covers running
the same receptors under several settings. The
:doc:`../reference/configuration` lists every option.

A typical config.yaml
---------------------

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

   grid:
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
     - Where the meteorology files are and how they are named. With no
       ``variants``, each one (``hrrr`` above) is run for every receptor and
       names its simulations. See :doc:`meteorology`.
     - one entry
   * - ``grid``
     - The map grid to calculate the footprint on: longitude range
       (``xmin``/``xmax``), latitude range (``ymin``/``ymax``), and cell size
       in degrees (``xres``/``yres``). Leave it out, or set it to ``null``,
       to keep only the particle trajectories.
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

Footprint options
-----------------

Besides the grid, the footprint takes these settings, next to ``grid`` at
the top level of ``config.yaml``:

``smooth_factor``
   Scales the Gaussian smoothing applied to each particle's influence.
   ``1.0`` (the default) is standard STILT; smaller values smooth less.

``time_integrate``
   ``true`` sums the footprint over time into a single map, making smaller
   files. The default ``false`` keeps an hourly time dimension.

``transforms``
   Particle weighting steps, mainly for column measurements. See
   :doc:`../advanced/transforms`.

Variants
--------

Every receptor is run once per **variant**. A variant is a name, a met
stream, and any settings that differ from the ones written at the top level
of ``config.yaml``, which are the defaults. With no ``variants`` section,
there is one variant per met stream, named after it, so the typical config
above runs each receptor once as ``hrrr``.

Once you write a ``variants`` section, only the variants in it run. Keep an
entry with no overrides (``hrrr: {}`` below) for the run with the defaults
as they are; the name is yours to choose. Declare more to run the same
receptors under other settings:

.. code-block:: yaml

   variants:
     hrrr: {}                     # the defaults with the hrrr met
     hrrr-zi08: {ziscale: 0.8}    # a mixed-layer sensitivity
     hrrr-np3k: {numpar: 3000}    # a particle-count check
     hrrr-err:                    # a wind-error run (see transport_error)
       siguverr: 2.6
       tluverr: 260
       zcoruverr: 450
       horcoruverr: 14
       grid: null                 # particles only

Each variant of each receptor is one simulation, stored in
``simulations/by-id/<receptor>/<variant>/`` (:doc:`project_layout`), and
every variant uses the same list of receptors. A variant may set:

``met``
   Which met stream to use. Needed only when ``mets`` has more than one entry.

``realizations``
   Make the variant a numbered group: it runs ``N`` times as ``<name>-0``
   to ``<name>-(N-1)``, each with ``seed + k``. This is how a transport-error
   ensemble is declared (:doc:`transport_error`). More than one needs
   ``krand: 4``, or ``krand: 2`` with a ``seed``, so the runs differ. A
   group is always numbered, even at ``realizations: 1``, so raising the
   count later only adds runs.

any other setting
   Transport settings (``numpar``, ``ziscale``, turbulence, error
   statistics) and footprint settings (``grid``, ``smooth_factor``,
   ``time_integrate``, ``transforms``) override the defaults for that
   variant only.

Names use lowercase letters, digits, and hyphens, because they become
directory names.

A second footprint from the same particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A footprint is cheap to make from stored particles; the HYSPLIT run is the
expensive part. ``from:`` declares a variant that re-uses another variant's
trajectory and changes only the footprint:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-regional:
       from: hrrr
       grid: {xmin: -125.0, xmax: -100.0, ymin: 30.0, ymax: 50.0, xres: 0.1, yres: 0.1}
     hrrr-ak:
       from: hrrr
       transforms: [{kind: averaging_kernel, table: kernels.parquet}]

A ``from:`` variant runs no HYSPLIT and may set only footprint settings.
Adding one to a finished project and running again calculates just its
footprints from the stored particles. A ``grid`` given in a variant updates
the default grid field by field, so a coarser version of the same domain is
just ``grid: {xres: 0.1, yres: 0.1}``. You may not need a variant at all:
:meth:`stilt.Footprint.aggregate` sums a fine footprint onto coarser cells or
irregular areas after the fact.

Changing a variant that has run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first time a project runs, PYSTILT writes the full settings of every
variant to ``simulations/variants.yaml``. That file is the record of what
produced the outputs (``config.yaml`` is yours to edit, and PYSTILT never
rewrites it). Running again with a variant whose settings changed, including
a changed default that the variant inherits, stops with an error naming the
settings, because the finished outputs would no longer match their name:

.. code-block:: text

   ConfigChangedError: These settings already ran under their name in ./my_project
   (hrrr: ziscale). Declare a new variant for the new settings, or remove the
   old outputs first (Model.remove / stilt rm --variant).

You have two ways forward:

- **Give the new settings a new name.** Add ``hrrr-zi08: {ziscale: 0.8}``
  and put ``hrrr`` back as it was. Only the new variant runs.
- **Remove the old outputs.** ``stilt rm --variant hrrr`` (or
  ``model.remove("hrrr")``) deletes every simulation of that variant and
  forgets its settings, so it runs again as new on the next run. Variants
  declared with ``from: hrrr`` are removed with it, since their footprints
  came from its particles.

Settings that do not change a result can change freely: ``execution``, the
HYSPLIT ``timeout``, ``rm_dat``, and where the meteorology files are
(``directory``). Taking a variant out of ``config.yaml`` does not delete
anything; ``stilt status`` lists such variants until you remove them.

Footprints for shapefiles, hexagons, or point sources
-----------------------------------------------------

If you will add footprints up over irregular areas, such as counties from a
shapefile, H3 hexagons, or small windows around point sources, you can name
those areas instead of a grid. PYSTILT then picks a grid fine enough to
resolve them (:meth:`stilt.Grid.from_geometry`). This needs the ``geometry``
extra:

.. code-block:: yaml

   geometry:
     kind: file          # shapefile / GeoPackage / GeoJSON (needs geopandas)
     path: counties.shp
     ids: NAME           # attribute column used as cell ids

The other kinds are ``h3`` hexagons and ``windows`` around point sources;
different geometries for the same particles are variants with ``from:``:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-hexes:
       from: hrrr
       geometry:
         kind: h3            # needs the h3 package
         resolution: 8
         bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}
       cells_per_target: 4   # native cells across the smallest hexagon (default)
     hrrr-sources:
       from: hrrr
       geometry:
         kind: windows
         coords: [[-111.97, 40.515], [-112.015, 40.779]]
         size: 0.01
         ids: [landfill, wwtp]

The derived ``grid`` is written back into the config, so the geometry object
is never needed to read a stored footprint.  Give both ``grid`` and
``geometry`` to pin the raster explicitly; ``geometry`` is then kept as a
record and ``sim.config.geometry.build()`` returns the
:class:`stilt.Mesh` to aggregate onto.  A content hash of the built geometry (``geometry_hash``) is
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
       grid={
           "xmin": -114.0, "xmax": -111.0,
           "ymin": 39.0, "ymax": 42.0,
           "xres": 0.01, "yres": 0.01,
       },
       variants={"hrrr": {}, "hrrr-zi08": {"ziscale": 0.8}},
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
       grid=stilt.Grid(
           xmin=-114.0, xmax=-111.0,
           ymin=39.0, ymax=42.0,
           xres=0.01, yres=0.01,
       ),
       n_hours=-24,
       numpar=500,
   )
   model = stilt.Model(project="./my_project", config=config)
   model.variants          # {"hrrr": VariantConfig(...)}: every variant, resolved

Advanced HYSPLIT settings
-------------------------

``config.yaml`` also accepts HYSPLIT and STILT's lower-level settings
(turbulence, time step, output variables, and so on) under the same names
STILT-R uses. Most projects never change them; the
:doc:`../reference/configuration` describes each one.

Typos are caught early: an unknown key in ``config.yaml`` is an error when the
file is loaded, before anything runs.
