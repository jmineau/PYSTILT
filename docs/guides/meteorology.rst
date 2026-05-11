Meteorology
===========

STILT requires gridded meteorological fields in NOAA ARL format. PYSTILT
supports two ways to provide them:

- **Archive mode** — point PYSTILT at a directory of ARL files you already
  have. PYSTILT globs for the right files at runtime.
- **Source mode** — let PYSTILT download from NOAA archives automatically
  via the `arl-met <https://github.com/jmineau/arl-met>`_ package.

Both modes stage the required files into each simulation's compute-local
directory. Both support optional spatial subsetting.

``MetConfig``
   Output configuration stored in ``config.yaml``.

``MetStream``
   The runtime object that resolves and stages required files for one named
   met stream.


Archive mode
------------

Point ``directory`` at your ARL file tree and tell PYSTILT how filenames
encode time:

.. code-block:: python

   import stilt

   hrrr = stilt.MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="1h",
       n_min=2,
   )

You can register multiple streams in one ``ModelConfig``:

.. code-block:: python

   config = stilt.ModelConfig(
       mets={
           "hrrr": hrrr,
           "gfs": stilt.MetConfig(
               directory="/data/met/gfs",
               file_format="gfs_%Y%m%d_%H",
               file_tres="3h",
           ),
       },
   )

How file selection works
~~~~~~~~~~~~~~~~~~~~~~~~

At runtime, ``MetStream.required_files()`` computes the simulation time window,
derives the required filename patterns from ``file_format`` and ``file_tres``,
and globs for matching ARL files recursively under ``directory``. Each expected
time step triggers one targeted search rather than a full directory scan, which
keeps I/O proportional to the number of time steps needed rather than the size
of the archive.

If too few files are found, the run fails clearly instead of silently starting
with incomplete meteorology.


Source mode (automatic download)
---------------------------------

Set ``source`` to an arlmet source name to have PYSTILT fetch ARL files from
NOAA archives automatically:

.. code-block:: python

   hrrr = stilt.MetConfig(
       source="hrrr",
       directory="/data/met/hrrr",  # local cache directory
   )

Available sources:

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 30

   * - Name
     - Product
     - Domain
     - Period
   * - ``hrrr``
     - HRRR 3 km analysis
     - CONUS
     - Jun 2019–present
   * - ``hrrr.v1``
     - HRRR 3 km analysis v1
     - CONUS
     - Jun 2015–2019
   * - ``nam12``
     - NAM 12 km analysis
     - North America
     - May 2007–present
   * - ``nams``
     - NAMS hybrid sigma-pressure
     - CONUS / AK / HI
     - Jan 2010–present
   * - ``gdas1``
     - GDAS 1-degree global
     - Global
     - Dec 2004–present
   * - ``gdas0p5``
     - GDAS 0.5-degree global
     - Global
     - Sep 2007–mid 2019
   * - ``gfs0p25``
     - GFS 0.25-degree global
     - Global
     - Jun 2019–present
   * - ``reanalysis``
     - NCEP/NCAR Reanalysis 2.5-degree
     - Global
     - 1948–present
   * - ``narr``
     - NCEP North American Regional Reanalysis 32 km
     - North America
     - 1979–2019

Source-specific options are passed as inline fields. For example, ``nams``
supports a ``domain`` parameter:

.. code-block:: python

   nams_ak = stilt.MetConfig(
       source="nams",
       domain="ak",            # passed to NAMSSource(domain="ak")
       directory="/data/met/nams_ak",
   )

The ``backend`` field selects the download source (default ``"s3"``):

.. code-block:: python

   MetConfig(source="gdas1", directory="/data/met/gdas1", backend="ftp")

Downloaded files are cached in ``directory``. Re-running with the same config
skips already-downloaded files.

.. note::

   Download mode requires ``fsspec`` and ``s3fs`` (for the default S3
   backend). Install with: ``pip install pystilt[cloud]``


Subsetting
----------

Setting ``subgrid_enable=True`` restricts the meteorology to a spatial
bounding box before staging. This is strongly recommended for global products
(GFS, GDAS, Reanalysis) and useful on HPC where compute nodes have limited
memory.

.. code-block:: python

   from stilt import Bounds, MetConfig

   hrrr = MetConfig(
       source="hrrr",
       directory="/data/met/hrrr",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
       subgrid_buffer=0.5,   # degrees added on each side (default 0.2)
   )

In **source mode**, subsetting is applied during download — the cropped file
is cached directly so subsequent runs reuse it.

In **archive mode**, subsetting is applied the first time a file is needed.
Cropped copies are cached in ``subgrid_dir`` (defaults to
``<directory>/subgrid``) and reused by all simulations that share the same
met stream. Set ``subgrid_dir`` explicitly to use a shared cache across
multiple projects:

.. code-block:: python

   MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="1h",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
       subgrid_dir="/scratch/met_subgrid/hrrr_slv",
   )

Use ``subgrid_levels`` to also reduce the number of vertical levels:

.. code-block:: python

   MetConfig(
       ...,
       subgrid_levels=20,   # keep the lowest 20 levels
   )


Staging into compute-local space
---------------------------------

``Simulation.met_files`` stages the selected meteorology files into the
simulation's compute-local ``met/`` directory using link-or-copy semantics.

This matters when:

- your archive is read-only
- workers use a slower shared filesystem for output
- HYSPLIT should read from a short local path in scratch space
