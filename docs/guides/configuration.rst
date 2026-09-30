Configuration
=============

A project's settings live in ``config.yaml`` in the project folder. They say
which meteorology to use, what footprint to make, and how to run HYSPLIT.
This page covers the settings most projects need. `Variants`_ explains how
to run the same receptors under several settings, and
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

``stilt init`` writes a starter file with comments. If you build a
:class:`~stilt.Model` in Python instead, the settings you pass are written
to ``config.yaml`` each time you run it. The written file includes a
``variants`` section that lists the variants that run (see `Variants`_), so
it reads the same either way.

The settings most people change
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Setting
     - What it does
     - Typical value
   * - ``mets``
     - Where the meteorology files are and how they are named. See
       :doc:`meteorology`.
     - one entry
   * - ``grid``
     - The map grid for the footprint: the longitude range
       (``xmin``/``xmax``), the latitude range (``ymin``/``ymax``), and the
       cell size in degrees (``xres``/``yres``). Leave it out, or set it to
       ``null``, to keep only the particle trajectories.
     - 0.01° (about 1 km) for a city, 0.1° for a region
   * - ``n_hours``
     - How many hours to follow the particles. Negative values run backward
       in time, which is what footprints need. Positive values run forward
       from the receptor (see :doc:`plume_background`).
     - ``-24`` to ``-72``
   * - ``numpar``
     - Particles released per simulation. More particles give a smoother
       footprint and take longer. The default is 200.
     - ``200`` to ``1000``
   * - ``execution``
     - Where to run. Leave it out to run on your own computer. See
       :doc:`execution/index`.
     - (omit)

Footprint options
-----------------

These settings sit next to ``grid`` at the top level of ``config.yaml``.

``smooth_factor``
   Scales the Gaussian smoothing applied to each particle's influence. The
   default, ``1.0``, is standard STILT. Smaller values smooth less.

``time_integrate``
   ``true`` sums the footprint over time into a single map, which makes
   smaller files. The default, ``false``, keeps one layer per hour.

``transforms``
   Steps that weight the particles before the footprint is made, mainly for
   column measurements. See :doc:`../advanced/transforms`.

Variants
--------

Every receptor is run once per variant. A variant has a name, a
meteorology source, and any settings that differ from the top level of
``config.yaml``. The top-level settings are the defaults.

With no ``variants`` section there is one variant per meteorology source,
named after it. The typical config above runs each receptor once, as
``hrrr``.

Once you write a ``variants`` section, only the variants in it run. Keep an
entry with no overrides (``hrrr: {}`` below) to run the defaults unchanged.
You can give it any name. Add more entries to run the same receptors under
other settings:

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

Each variant of each receptor is one simulation. Its results go to the
output directory, in folders named after the variant (see
:doc:`project_layout`). Every variant runs the same list of receptors. A
variant may set:

``met``
   Which meteorology source to use. A variant named after a source uses
   that source. Otherwise you need ``met`` when ``mets`` has more than one
   entry.

``realizations``
   Run the variant ``N`` times, as ``<name>-0`` to ``<name>-(N-1)``. This is
   how you declare a transport-error ensemble (see :doc:`transport_error`).
   The default ``krand: 4`` gives each run different turbulence. For
   ensembles you can reproduce, set ``krand: 2`` and a ``seed``. Run ``k``
   then uses ``seed + k``. The names are numbered even with
   ``realizations: 1``, so raising the count later only adds runs.

Any other setting
   Transport settings (``numpar``, ``ziscale``, turbulence, wind errors)
   and footprint settings (``grid``, ``smooth_factor``, ``time_integrate``,
   ``transforms``) override the defaults for that variant only.

Variant names may use lowercase letters, digits, and hyphens, because they
become folder names. The same rule applies to the names in ``mets``.

A second footprint from the same particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running HYSPLIT is the slow part of a simulation. Making a footprint from
stored particles is quick. A variant that changes only footprint settings
shares the particles of the variant it matches and makes its own footprint
from them:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-regional:
       grid: {xmin: -125.0, xmax: -100.0, ymin: 30.0, ymax: 50.0, xres: 0.1, yres: 0.1}
     hrrr-ak:
       transforms: [{kind: averaging_kernel, table: kernels.parquet}]

``hrrr-regional`` and ``hrrr-ak`` have the same transport settings as
``hrrr``, so HYSPLIT runs once per receptor for the three of them. There is
nothing to declare: PYSTILT sees that the settings that decide the particles
are equal. If you add such a variant to a finished project and run again,
PYSTILT makes just the new footprints from the stored particles.

A ``grid`` in a variant changes only the fields it names. A coarser version
of the default domain is just ``grid: {xres: 0.1, yres: 0.1}``.

You may not need a variant at all. :meth:`stilt.Footprint.aggregate` sums a
fine footprint onto coarser cells or irregular areas after the run.

Changing a variant that has run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A variant's results live in folders named by a hash of its settings (see
:doc:`project_layout`). Change a setting, including a default the variant
inherits or a setting of its meteorology source, and the variant now points
at a folder that does not exist yet. The next run fills it, for every
receptor. The old folder stays as it was, and ``stilt status`` lists it as a
folder no variant uses any more.

If you want to keep both, give the new settings a new name instead:
add ``hrrr-zi08: {ziscale: 0.8}`` and leave ``hrrr`` as it was. Then only
the new variant runs, and you can select both when you load results, for
example ``sel(variant=["hrrr", "hrrr-zi08"])``. Each folder's
``_settings.yaml`` shows what it ran with.

Some settings do not change a result, so changing them changes no folder:

- ``execution``, ``output``, and ``keep_scratch``
- ``timeout``, ``rm_dat``, and ``exe_dir``
- where the meteorology files are (``directory`` and ``subgrid_dir``)

Taking a variant out of ``config.yaml`` does not delete its results.
``stilt status`` lists its folders until you delete them by hand. PYSTILT
never deletes a folder itself, because another project may share the
output directory.

Footprints for shapefiles, hexagons, or point sources
-----------------------------------------------------

You may plan to add footprints up over irregular areas, such as counties
from a shapefile, H3 hexagons, or small windows around point sources. You
can then name those areas in ``geometry`` instead of giving a grid. PYSTILT
picks a grid fine enough to resolve them (see
:meth:`stilt.Grid.from_geometry`). This needs the ``geometry`` extra
(``pip install "pystilt[geometry]"``).

.. code-block:: yaml

   geometry:
     kind: file          # a shapefile, GeoPackage, or GeoJSON
     path: counties.shp
     ids: NAME           # attribute column used as cell ids

The other kinds are ``h3`` hexagons and ``windows`` around point sources.
To make footprints for several geometries from the same particles, declare
one variant per geometry; they share the particles:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-hexes:
       geometry:
         kind: h3
         resolution: 8
         bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}
       cells_per_target: 4   # grid cells across the smallest hexagon (default)
     hrrr-sources:
       geometry:
         kind: windows
         coords: [[-111.97, 40.515], [-112.015, 40.779]]
         size: 0.01
         ids: [landfill, wwtp]

The grid PYSTILT picks is saved in the footprint folder's ``_settings.yaml``
and with each footprint. You do not need the shapefile again to read a
footprint.

If you give both ``grid`` and ``geometry``, the ``grid`` is used as given.
The geometry is kept with the settings, and ``sim.config.geometry.build()``
returns the :class:`stilt.Mesh` to aggregate onto.

Each footprint file also stores a hash of the geometry it was made for.
:meth:`stilt.Footprint.aggregate` warns if the mesh you pass no longer
matches, for example because the shapefile was edited after the run.

.. note::

   :class:`stilt.Zones` (groups of grid cells) cannot be written in YAML.
   Build them in code with ``Zones.from_labels(base, labels)``.

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

PYSTILT uses ``backend``, ``n_workers``, ``cpus_per_task``,
``array_parallelism``, and ``setup`` itself. Every other key is passed to
``sbatch``. See :doc:`execution/slurm`.

The same settings in Python
---------------------------

You can pass any ``config.yaml`` key to :class:`~stilt.Model` directly, as
plain dictionaries:

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

You can also use the config classes, which your editor can autocomplete and
check:

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
   model.variants          # {"hrrr": VariantConfig(...)}, each with its full settings

Settings given in Python replace ``config.yaml`` when the model runs. To
keep editing the file by hand, open the project with
``stilt.Model(project="./my_project")`` and no settings.

Advanced HYSPLIT settings
-------------------------

``config.yaml`` also accepts the lower-level HYSPLIT and STILT settings,
such as turbulence, time step, and output variables. They use the same
names as in STILT-R. Most projects never change them.
:doc:`../reference/configuration` describes each one.

PYSTILT checks ``config.yaml`` when it loads the file, before anything runs.
A misspelled setting is an error. The one exception is inside a ``mets``
entry. Extra keys there are passed on as download options (like ``domain``
for ``nams``), so check their spelling yourself.
