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
(``hrrr`` in the examples below). Variants refer to a met by this name, and
a variant named after a met runs with it (see :doc:`projects`). Give
each met a variant to run the same receptors with several mets and compare
them.

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

For each simulation, PYSTILT works out which files cover the hours the run
spans (``n_hours`` from the receptor time) and looks for those. A 24-hour
backward run with 6-hour HRRR files needs 5 or 6 files, depending on the
receptor hour. Every one is needed. If one is missing, the simulation fails
before HYSPLIT runs, as ``MET_COVERAGE``, naming the hours with no file.

A met file cut short fails the run too, as ``MET_COVERAGE``, when HYSPLIT
says the meteorology ran out. Particles that all leave the meteorology's
domain, or its crop, before the end of the run are not a failure. The run
is complete, and its footprint holds the hours the particles were inside
(:doc:`checking`).

In Python, the same settings are a dictionary or a
:class:`~stilt.transport.hysplit.MetConfig`, HYSPLIT's met config:

.. code-block:: python

   from stilt.transport.hysplit import MetConfig

   hrrr = MetConfig(
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

Downloading needs the ``download`` extra (``pip install "pystilt[download]"``).
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

   nams_ak = MetConfig(
       download="nams",
       domain="ak",            # "conus" (default), "ak", or "hi"
       directory="/data/met/nams_ak",
   )

``download_from`` picks the server. The default, ``"s3"``, is NOAA's
archive on AWS. The others are ``"ftp"`` and ``"http"``.

.. code-block:: python

   MetConfig(download="gdas1", directory="/data/met/gdas1", download_from="ftp")

Files already in ``directory`` are not downloaded again.

ERA5
----

ERA5, ECMWF's global reanalysis, is hourly on a 0.25 degree grid. NOAA
does not publish it in ARL format, and PYSTILT does not convert it, so
convert it to ARL first. Then it is a met like any other:

.. code-block:: yaml

   mets:
     era5:
       directory: /data/met/era5
       file_format: "ERA5_%Y%m%d"
       file_tres: 24h
   variants:
     era5:
       met: era5
       kbls: 2
       kmixd: 0

Two settings differ from the defaults:

``kbls: 2``
   Derive the boundary layer's stability from the wind and temperature
   profiles. ERA5's friction velocity is too small (ECMWF's ERA5
   documentation says so), and stability derived from it, the default
   ``kbls: 1``, can leave too little mixing.

``kmixd: 0``
   Use ERA5's own boundary-layer height, which ECMWF computes on its model
   levels. HYSPLIT's own estimate, the default ``kmixd: 3``, would come
   from ERA5's pressure levels, about 200 m apart near the ground. The
   converted files need the boundary-layer height (PBLH) for this.

Cropping to your region
-----------------------

Set ``subgrid_enable`` and ``subgrid_bounds`` to crop the meteorology to a
box around your region before HYSPLIT reads it. Smaller files make runs
faster and use less memory. Cropping matters most for global products
(GFS, GDAS, Reanalysis) and on cluster nodes with limited memory.

.. code-block:: python

   from stilt import Bounds
   from stilt.transport.hysplit import MetConfig

   hrrr = MetConfig(
       download="hrrr",
       directory="/data/met/hrrr",
       subgrid_enable=True,
       subgrid_bounds=Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
   )

A particle that leaves the box stops, as it would at the edge of the
weather model's grid. Make the box wide enough for where the particles go
over the whole run, not only the footprint grid. A box too tight cuts off
the far part of a footprint without any warning.

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
code computed from the crop box and ``subgrid_levels``. Changing ``subgrid_bounds``
or ``subgrid_levels`` starts a new folder, and projects with the same crop
share one. ``Met(name, config).crop_dir`` gives
the folder. Old folders are not deleted.

``subgrid_levels`` also drops the upper vertical levels, for downloaded
files and your own.

.. code-block:: python

   MetConfig(
       ...,
       subgrid_levels=20,   # keep the lowest 20 levels
   )


Which weather a run records
---------------------------

Each ARL file names the weather model it comes from in its header, such
as ``HRRR``, ``NAM``, or ``GDAS``. A run's settings folder records that
name, read from the header of the first file in ``directory``, and the
crop. A downloaded met's name comes from its archive, so no file is read.

Where the files are kept and how they are named are not recorded. Moving
the met files, or switching to a copy named differently, keeps every run
already made. Two mets whose files are named alike but come from
different weather models, such as an HRRR and a NAM archive, give
different settings folders.

The header is read when a project's settings folders are first needed,
to run it or to read its results. A met's ``directory`` must hold its
files on the machine that opens the project.

Where HYSPLIT reads the files
-----------------------------

HYSPLIT reads the files where they are. PYSTILT does not change or copy
your files, apart from the cropped copies it makes when you ask for a
crop.
