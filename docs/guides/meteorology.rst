Meteorology
===========

STILT moves particles with gridded fields from a weather model: winds,
temperature, turbulence, and boundary-layer height. The files must be in
NOAA's :term:`ARL` format. NOAA publishes ARL files for HRRR, NAM, GDAS,
GFS, and other models.

You can give PYSTILT meteorology in two ways:

- Point it at files you already have, such as your group's archive.
- Let it download the files from NOAA. PYSTILT fetches the files each
  simulation needs and keeps them for later runs.

Each set of meteorology files is a *met* with a name in ``config.yaml``
(``hrrr`` in the examples below). Variants refer to a met by this name.
With no ``variants`` section, each met runs as a variant of the same name
(see :doc:`configuration`), so you can run the same receptors with several
mets and compare them.

Use files you already have
--------------------------

Give the folder, the pattern of the file names, and how many hours each
file covers:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/met/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

``directory``
   The folder that holds the files. Subfolders are searched too.

``file_format``
   The file names, with the date written as `strftime codes
   <https://docs.python.org/3/library/datetime.html#format-codes>`_:
   ``%Y`` for the year, ``%m`` month, ``%d`` day, and ``%H`` hour. Files
   named ``20230715_18`` match ``"%Y%m%d_%H"``. Files named
   ``hysplit.20230715.18z.hrrra`` match ``"hysplit.%Y%m%d.%Hz.hrrra"``. A
   file matches when its name starts with the pattern.

``file_tres``
   How much time each file covers, such as ``1h``, ``3h``, or ``6h``.

``n_min``
   The fewest files a simulation needs (default 1).

For each simulation, PYSTILT works out which files cover the hours the run
spans (``n_hours`` from the receptor time) and looks for those. If it finds fewer than
``n_min``, the simulation fails with an error. If some are missing but at
least ``n_min`` are found, PYSTILT logs a warning and runs with the files it
has. Raise ``n_min`` to turn gaps into errors. A 24-hour backward run with
6-hour HRRR files needs 5 or 6 files, depending on the receptor hour, so
``n_min: 5`` is a good choice there.

In Python, the same settings are a dictionary or a :class:`~stilt.MetConfig`:

.. code-block:: python

   hrrr = stilt.MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="6h",
   )

You can list several mets:

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

Set ``download`` to the name of one of NOAA's ARL archives and
``directory`` to where the downloads should go. You don't need
``file_format`` or ``file_tres``.

.. code-block:: yaml

   mets:
     hrrr:
       download: hrrr
       directory: /data/met/hrrr     # downloads are kept here

Downloading needs the ``cloud`` extra (``pip install "pystilt[cloud]"``).
The `arl-met <https://github.com/jmineau/arl-met>`_ package does the
downloading.

.. warning::

   NOAA's files cover a continent or the whole globe, so each one is large,
   often several gigabytes. The whole file is downloaded before it is
   cropped. Crop to your region (see below) so that what is kept is small,
   and download on a machine with a fast connection and plenty of disk
   space.

These archives are available:

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

Some archives take extra options, which you write next to the other fields.
For example, ``nams`` takes a ``domain``:

.. code-block:: python

   nams_ak = stilt.MetConfig(
       download="nams",
       domain="ak",            # "conus" (default), "ak", or "hi"
       directory="/data/met/nams_ak",
   )

``download_from`` picks the server. The default, ``"s3"``, is NOAA's
archive on AWS. The others are ``"ftp"`` and ``"http"``.

.. code-block:: python

   stilt.MetConfig(download="gdas1", directory="/data/met/gdas1", download_from="ftp")

Files already in ``directory`` are not downloaded again.


Cropping to your region
-----------------------

Set ``subgrid_enable`` and ``subgrid_bounds`` to crop the meteorology to a
box around your region before HYSPLIT reads it. Smaller files make runs
faster and use less memory. Cropping matters most for global products
(GFS, GDAS, Reanalysis) and on cluster nodes with limited memory.

.. code-block:: python

   from stilt import Bounds, MetConfig

   hrrr = MetConfig(
       download="hrrr",
       directory="/data/met/hrrr",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
       subgrid_buffer=0.5,   # degrees added on each side (default 0.2)
   )

When downloading, each file is cropped right after it arrives and only the
cropped copy is kept.

With your own files, each file is cropped the first time a simulation needs
it, and every later simulation reuses the crop. Set ``subgrid_dir`` to the
directory for the cropped copies. It is required, so that PYSTILT never
writes into your met archive. Scratch space suits it well, and several
projects can share one:

.. code-block:: python

   MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="6h",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
       subgrid_dir="/scratch/met_subgrid/hrrr_slv",
   )

Inside ``subgrid_dir``, each crop gets its own folder, named by a short
hash of the crop box and ``subgrid_levels``. Changing ``subgrid_bounds``,
``subgrid_buffer``, or ``subgrid_levels`` starts a new folder, and projects
with the same crop share one. ``Met(name, config).crop_dir`` gives
the folder. Old folders are not deleted.

``subgrid_levels`` also drops the upper vertical levels, for downloaded
files and your own.

.. code-block:: python

   MetConfig(
       ...,
       subgrid_levels=20,   # keep the lowest 20 levels
   )


Where HYSPLIT reads the files
-----------------------------

HYSPLIT reads the files where they are. PYSTILT does not change or copy
your files, apart from the cropped copies it makes when you ask for a
crop.
