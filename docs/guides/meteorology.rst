Meteorology
===========

STILT moves particles with gridded weather-model fields: winds, temperature,
turbulence, and boundary-layer height. These must be in NOAA's :term:`ARL`
format, which NOAA publishes for HRRR, NAM, GDAS, GFS, and other models.

You can give PYSTILT meteorology in two ways:

- **Use files you already have**, such as a research group's archive.
- **Download them from NOAA.** PYSTILT fetches the files it needs and keeps
  them for later runs.

Each meteorology source gets a name in ``config.yaml`` (``hrrr`` below). The
name goes into every simulation ID, so you can run the same receptors with
several sources and compare.

Use files you already have
--------------------------

Tell PYSTILT the folder, the pattern of the filenames, and how many hours
each file covers:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/met/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

``directory``
   The folder holding the files. Subfolders are searched too.

``file_format``
   The filenames with the date replaced by `strftime codes
   <https://docs.python.org/3/library/datetime.html#format-codes>`_: ``%Y``
   year, ``%m`` month, ``%d`` day, ``%H`` hour. Files named
   ``20230715_18`` match ``"%Y%m%d_%H"``; files named
   ``hysplit.20230715.18z.hrrra`` match ``"hysplit.%Y%m%d.%Hz.hrrra"``.

``file_tres``
   How much time each file covers, such as ``1h``, ``3h``, or ``6h``.

For each simulation, PYSTILT works out which files cover the receptor time
and the ``n_hours`` before it, and looks for exactly those. If the files
aren't there, the simulation fails with a clear error instead of running
with incomplete meteorology. ``n_min`` sets the minimum
number of files a run needs (default 1).

In Python the same settings are a dictionary or a :class:`~stilt.MetConfig`:

.. code-block:: python

   hrrr = stilt.MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="6h",
   )

Several sources at once:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/met/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h
     gfs:
       directory: /data/met/gfs
       file_format: "gfs_%Y%m%d_%H"
       file_tres: 3h


Download from NOAA
------------------

Set ``source`` to one of NOAA's products and ``directory`` to where the
downloads should go. You don't need ``file_format`` or ``file_tres``:

.. code-block:: yaml

   mets:
     hrrr:
       source: hrrr
       directory: /data/met/hrrr     # downloads are kept here

Downloading needs the ``cloud`` extra: ``pip install "pystilt[cloud]"``.
The downloads are handled by the `arl-met <https://github.com/jmineau/arl-met>`_ package.

.. warning::

   NOAA's files cover a whole continent or the globe, so each one is large
   (often gigabytes), and the full file is downloaded before any cropping.
   Crop to your region (below) so what is kept is small, and download on a
   machine with a fast connection and plenty of disk space.

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

The ``backend`` field picks where to download from (default ``"s3"``,
NOAA's archive on AWS):

.. code-block:: python

   MetConfig(source="gdas1", directory="/data/met/gdas1", backend="ftp")

Files already in ``directory`` are not downloaded again.


Cropping to your region
-----------------------

Setting ``subgrid_enable=True`` crops the meteorology to a box around your
region before HYSPLIT reads it. Smaller files mean faster runs and less
memory. This is strongly recommended for global products (GFS, GDAS,
Reanalysis) and helps on clusters where nodes have limited memory.

.. code-block:: python

   from stilt import Bounds, MetConfig

   hrrr = MetConfig(
       source="hrrr",
       directory="/data/met/hrrr",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
       subgrid_buffer=0.5,   # degrees added on each side (default 0.2)
   )

When **downloading**, each file is cropped right after download and only the
cropped copy is kept.

With **your own files**, each file is cropped the first time it is needed.
Cropped copies are cached in ``subgrid_dir`` (defaults to
``<directory>/subgrid``) and reused by all simulations that share the same
meteorology source. Set ``subgrid_dir`` explicitly to use a shared cache across
multiple projects:

.. code-block:: python

   MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="6h",
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


Where HYSPLIT reads the files
-----------------------------

Before each simulation, PYSTILT links (or, if linking isn't possible,
copies) the meteorology files it needs into that simulation's working
folder, and HYSPLIT reads them from there. Your archive is never modified, so
it can be read-only, and on a cluster HYSPLIT can read from fast local
scratch space (see ``compute_root`` in :doc:`project_layout`).
