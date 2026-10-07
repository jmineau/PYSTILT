Projects And Variants
=====================

A PYSTILT :term:`project` is a folder with your settings and your
receptors. Every receptor runs once under each :term:`variant`, a named set
of settings, and the results go to an output directory the settings name.
This page covers what is in the folder, the settings most projects need,
variants, and how reruns and changed settings work.
:doc:`../reference/configuration` lists every option.

The project folder
------------------

.. code-block:: text

   my_project/
     config.yaml        # your settings: meteorology, variants, footprint, run options
     receptors.csv      # your receptors: where and when to release particles
     tables/            # other inputs, such as averaging kernels (project.add_table)

``stilt init`` writes a starter ``config.yaml`` with comments, and so does
``Project.init(path, starter=True)`` in Python. If you pass settings to
:meth:`Project.init <stilt.Project.init>` instead, they are written to
``config.yaml`` once, so it reads the same either way. ``Project.init``
stops if the folder already has a ``config.yaml``.

Both files are yours to edit. PYSTILT never rewrites ``config.yaml``, and it
only appends to ``receptors.csv``. ``project.add_table("kernels", table)``
adds rows to ``tables/kernels.parquet`` the same way, and a transform names
it as ``table: kernels``. A Slurm run also makes a ``_slurm/`` folder with
its job scripts and logs (:doc:`slurm`).

A typical config.yaml
---------------------

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

   variants:
     hrrr: {}            # every receptor runs once, with these settings

   grid:
     xmin: -114.0
     xmax: -111.0
     ymin: 39.0
     ymax: 42.0
     xres: 0.01
     yres: 0.01

   n_hours: -24
   numpar: 500

A key PYSTILT does not know is an error that names the nearest setting, so
a typo such as ``smooth_factr`` is caught when the file loads, before
anything runs. The one exception is inside a ``mets`` entry: extra keys
there are passed on as download options (like ``domain`` for ``nams``), so
check their spelling yourself.

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
   * - ``variants``
     - The variants that run. ``hrrr: {}`` runs the defaults (see
       `Variants`_).
     - one entry
   * - ``grid``
     - The map grid for the footprint: the longitude range
       (``xmin``/``xmax``), the latitude range (``ymin``/``ymax``), and the
       cell size in degrees (``xres``/``yres``). Leave it out, or set it to
       ``null``, to make particles but no footprint.
     - 0.01° (about 1 km) for a city, 0.1° for a region
   * - ``n_hours``
     - How many hours to follow the particles. Negative values run backward
       in time, which is what footprints need. Positive values run forward
       from the receptor (see :doc:`plume_background`).
     - ``-24`` to ``-72``
   * - ``numpar``
     - Particles released per simulation. More particles give a smoother
       footprint and take longer. The default is 200; the config
       ``stilt init`` writes sets 1000.
     - ``200`` to ``1000``
   * - ``output``
     - Where results go (`The output directory`_).
     - ``./output``
   * - ``execution``
     - Where to run. Leave it out to run on your own computer. See
       :doc:`running`.
     - (omit)

These footprint settings sit next to ``grid`` at the top level:

``smooth_factor``
   Scales the Gaussian smoothing applied to each particle's influence. The
   default, ``1.0``, is standard STILT. Smaller values smooth less.

``time_integrate``
   ``true`` sums the footprint over time into a single map, which makes
   smaller files. The default, ``false``, keeps one layer per hour.

``transforms``
   Steps that weight the particles before the footprint is made, mainly for
   column measurements. See :doc:`transforms`.

``config.yaml`` also takes the lower-level HYSPLIT and STILT settings, such
as turbulence, time step, and output variables, by the names STILT-R uses.
Most projects never change them; :doc:`../reference/configuration`
describes each one.

Variants
--------

Every receptor runs once per variant. A variant has a name, a met, and any
settings that differ from the top level of ``config.yaml``, which are the
defaults. Every config lists its variants, and only those run. An entry with
no overrides (``hrrr: {}`` below) runs the defaults unchanged. Add more
entries to run the same receptors under other settings:

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

Each variant of each receptor is one simulation, so 100 receptors and
three variants make 300 simulations. A variant may set:

``met``
   Which met to use. A variant named after a met uses that met.
   Otherwise you need ``met`` when ``mets`` has more than one entry.

``realizations``
   Run the variant ``N`` times, as realizations ``0`` to ``N - 1`` of one
   ensemble. This is how you declare a transport-error ensemble (see
   :doc:`transport_error`). The default ``krand: 4`` gives each realization
   different turbulence. For ensembles you can reproduce, set ``krand: 2``
   and a ``seed``. Realization ``k`` then uses ``seed + k``. Raising the
   count later only adds realizations.

``model``
   The transport model, ``hysplit`` unless set. A model in a package of
   its own is named by its import path.

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
       transforms: [{kind: averaging_kernel, table: kernels}]

``hrrr-regional`` and ``hrrr-ak`` have the same transport settings as
``hrrr``, so HYSPLIT runs once per receptor for the three of them. There is
nothing to declare: PYSTILT sees that the settings that decide the particles
are equal. Add such a variant to a finished project and run again, and
PYSTILT makes only the new footprints, from the stored particles.

A ``grid`` in a variant changes only the fields it names. A coarser version
of the default domain is just ``grid: {xres: 0.1, yres: 0.1}``. You may not
need a variant at all: ``foot.stilt.aggregate`` sums a fine footprint onto
coarser cells or irregular areas after the run.

The output directory
--------------------

``output:`` in ``config.yaml`` says where results go, ``./output`` inside
the project by default. Any path works, and projects that name the same
directory share it.

.. code-block:: text

   output/
     particles/
       settings=hrrr-a3f9c2/                 # one folder per set of transport settings
         _settings.yaml                      # the settings, written out in full
         date=2023-07-15/<receptor id>.parquet
     footprints/
       settings=hrrr-93278c/                 # one folder per variant's footprint settings
         _settings.yaml                      # names the particles folder it was made from
         date=2023-07-15/<receptor id>.parquet
     logs/
       settings=hrrr-a3f9c2/date=2023-07-15/<receptor id>.log
     scratch/                                # workdirs of failed runs

Each :term:`settings folder` is named after the variant that first made it,
plus a short code computed from the settings it was made with. That is why
a changed setting never overwrites a result: edit ``ziscale`` and the next
run writes into a new folder beside the old one, and projects with the same
settings share a folder. The ``_settings.yaml`` in each folder lists the
settings in full, so a folder explains itself. The logs and the kept
workdirs are explained in :doc:`checking`.

The folder names are what pyarrow, DuckDB, polars, and R's ``arrow``
package read as columns, so ``output/footprints`` opens as one table with
``settings`` and ``date`` columns, no PYSTILT needed.

On an object store
~~~~~~~~~~~~~~~~~~

The output directory can be on an object store such as Amazon S3 or Google
Cloud Storage. Write it as a URL:

.. code-block:: yaml

   output: s3://my-bucket/slv-2023

PYSTILT reads and writes it through `fsspec
<https://filesystem-spec.readthedocs.io>`_, so install the package for your
store: ``pip install s3fs`` for ``s3://``, ``gcsfs`` for ``gs://``.
Credentials come from where those packages look for them, such as
``~/.aws/credentials`` or ``AWS_ACCESS_KEY_ID``. For another store that
speaks the S3 protocol, set ``AWS_ENDPOINT_URL``.

To use another output directory than ``config.yaml`` names, without
editing it, give ``--output`` to ``stilt run``, ``stilt submit``, or
``stilt status``, or open the project with ``stilt.Project(path,
output="s3://my-bucket/slv-2023")``.

Everything else is the same: the folder layout, reruns that skip finished
simulations, ``stilt status``, and loading results. Runs write each result
straight to the store. HYSPLIT itself still runs in a workdir on local disk
(``--compute-root`` or ``PYSTILT_COMPUTE_ROOT``). Listing an object store
takes longer than listing a disk, so ``stilt status`` on a large project is
slower.

Reruns skip finished work
-------------------------

Before running, PYSTILT checks which simulations are finished and runs only
the rest. A simulation is finished when its results exist in the output
directory: the particle file, and the footprint file if the variant has a
grid. If the particle file is missing, HYSPLIT runs again, and the
footprints of every variant that shares those particles are remade from
the new particles.

So after an interruption, a failed Slurm task, or adding a variant, run the
project again, and only what is missing runs. To see what is not finished
yet, use ``stilt status`` or ``project.status()`` (:doc:`checking`). To run
everything again, pass ``skip_existing=False`` to ``project.run()``, or
``--no-skip`` to ``stilt run``.

Changing a setting
------------------

Edit ``config.yaml`` and run again. Change a setting, including a default
the variant inherits or a setting of its met, and the variant's settings
now point at a folder that does not exist yet. The next run fills it, for
every receptor, and the old folder stays as it was.

``stilt status`` lists every settings folder in the output directory, how
many result files it holds, which variants use it, and how a folder no
variant uses differs from the variant of its name. Here ``numpar`` went
from 1000 to 2000, and ``hrrr-smooth`` differs from ``hrrr`` only in its
footprint, so the two share particles:

.. code-block:: text

   $ stilt status ./my_project
   Project: /data/my_project  total=8760  complete=8760  failed=0  interrupted=0  pending=0
     hrrr: total=4380  complete=4380  failed=0  interrupted=0  pending=0
     hrrr-smooth: total=4380  complete=4380  failed=0  interrupted=0  pending=0
   Output: /data/output
     particles   settings=hrrr-5d01e7         4,380 files  hrrr, hrrr-smooth
     particles   settings=hrrr-a3f9c2         4,380 files  (no variant)  numpar: 1000 (config: 2000)
     footprints  settings=hrrr-93278c         4,380 files  hrrr
     footprints  settings=hrrr-smooth-1b7e40  4,380 files  hrrr-smooth

The counts at the top are simulations; a folder's count is files, and one
particles folder can serve several variants. In Python the same table is
``project.folders()``.

To keep both results, give the new settings a new name instead: add
``hrrr-zi08: {ziscale: 0.8}`` and leave ``hrrr`` as it was. Only the new
variant runs, and you can select both when you load results, for example
``sims[sims.variant.isin(["hrrr", "hrrr-zi08"])]``.

Some settings do not change a result, so changing them changes no folder:

- ``execution`` (including ``timeout`` and ``keep_scratch``) and ``output``
- the ``exe_dir`` and ``data_dir`` paths (what they hold is recorded
  instead: the build's version, and a checksum of each data table that
  differs from the bundled one, so a different build or table is a new
  folder)
- where the meteorology files are (``directory`` and ``subgrid_dir``)

Taking a variant out of ``config.yaml`` does not delete its results.
PYSTILT never deletes a folder, because another project may share the
output directory. When you are sure, delete a folder by hand.

Footprints for shapefiles, hexagons, or point sources
-----------------------------------------------------

You may plan to add footprints up over irregular areas, such as counties
from a shapefile, H3 hexagons, or small windows around point sources. You
can then name those areas in ``geometry`` instead of giving a grid. PYSTILT
picks a grid fine enough to resolve them (see
:meth:`stilt.Mesh.to_grid`). This needs the ``geometry`` extra
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

Loading ``config.yaml`` does not read the shapefile. PYSTILT reads it
once, when it works out the variants' settings, and picks the grid then.
To make that grid finer or coarser, change ``cells_per_target``. A variant
cannot change only part of it (``grid: {xres: 0.1}``), because the grid is
not known until the shapefile is read.

The grid PYSTILT picks is saved in the footprint folder's ``_settings.yaml``
and with each footprint, so you do not need the shapefile again to read a
footprint. If you give both ``grid`` and ``geometry``, the ``grid`` is used
as given. The geometry is kept with the settings, and
``stilt.Mesh.from_spec(sim.variant.footprint.geometry)`` reads the
:class:`stilt.Mesh` to aggregate onto. Each footprint file also stores a
code for the geometry it was made for, and ``foot.stilt.aggregate`` warns
if the mesh you pass no longer matches, for example because the shapefile
was edited after the run.

.. note::

   :class:`stilt.Zones` (groups of grid cells) cannot be written in YAML.
   Build them in code with ``Zones.from_labels(base, labels)``.

Simulation IDs
--------------

A simulation is one receptor run under one variant. Its ID is the receptor
ID and the variant name joined by a slash:

.. code-block:: text

   {YYYYMMDDHHMM}_{location}/{variant}

   202307151800_-111.848_40.766_10/hrrr

For a point receptor, the location is the longitude, latitude, and
altitude. A column receptor has ``X`` and its bottom and top in place of
the altitude, as in ``202101150600_-112_40.5_X0-3000``. A multipoint
receptor uses ``multi_`` and a short hash of its points, which does not
change if you reorder them. Heights above mean sea level add ``msl`` at
the end, as in ``_100msl`` or ``_X0-3000msl``. Heights above ground have no
marker.

In the output directory the receptor ID is the file name, in a folder for
the day of the receptor time. Each particle file also has a ``receptor``
column, so a scan of the whole ``particles/`` tree with pyarrow, DuckDB, or
R can tell the receptors apart.

Opening a project again
-----------------------

After the project is made, the folder is all you need:

.. code-block:: python

   import stilt

   project = stilt.Project("./my_project")
   project.status()   # one row per simulation, with a "state" column

Opening a project only reads it. To change a setting, edit ``config.yaml``.
To add receptors, add them and run:

.. code-block:: python

   project.add_receptors(new_receptors)     # returns the ids of the new ones
   project.run()

New receptors are appended to ``receptors.csv`` in the file's own columns.
Receptors already in the file are left as they are.

The same settings in Python
---------------------------

You can pass any ``config.yaml`` key to
:meth:`Project.init <stilt.Project.init>` directly, as plain dictionaries:

.. code-block:: python

   import stilt

   project = stilt.Project.init(
       "./my_project",
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

   from stilt.transport.hysplit import MetConfig

   config = stilt.ProjectConfig(
       mets={
           "hrrr": MetConfig(
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
       variants={"hrrr": {}},
   )
   project = stilt.Project.init("./my_project", config=config)
   project.variants        # {"hrrr": Variant(...)}, each with its full settings
