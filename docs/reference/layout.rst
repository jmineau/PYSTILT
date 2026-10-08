Output Layout
=============

This page describes the files in an output directory, so that a reader in
R, Julia, DuckDB, or plain pyarrow can read them without PYSTILT. In Python,
:func:`stilt.read_particles`, :func:`stilt.read_footprint`, and
:meth:`stilt.Project.footprints` do all of this for you.

The folders
-----------

Each kind of result has its own tree. Inside it, each settings folder holds
the results of one set of settings, split by the receptor's date:

.. code-block:: text

   <output>/
     particles/
       settings=hrrr-7d36c8/
         _settings.yaml
         date=2024-07-15/<receptor id>.parquet
       settings=hrrr-err-b2399e/                 an ensemble
         _settings.yaml
         realization=0/date=2024-07-15/<receptor id>.parquet
         realization=1/date=2024-07-15/<receptor id>.parquet
     footprints/
       settings=hrrr-f82d40/
         _settings.yaml
         date=2024-07-15/<receptor id>.parquet
     logs/
       settings=hrrr-7d36c8/date=2024-07-15/<receptor id>.log
       settings=hrrr-7d36c8/date=2024-07-15/<receptor id>.failure.yaml
       settings=hrrr-f82d40/date=2024-07-15/<receptor id>.failure.yaml
     scratch/
       settings=hrrr-7d36c8/date=2024-07-15/<receptor id>/

A settings folder is named ``<variant>-<six characters>``. The variant is
the name in ``config.yaml`` that first made the folder, and the six
characters are the start of its settings hash (below). The date is the
receptor's release date in UTC. An ensemble (``realizations: N``) has a
``realization=k`` folder between the settings folder and the dates.

The ``logs/`` tree follows the folders of the results:

- ``<receptor id>.log`` is the transport model's log, under the particles'
  settings folder.
- ``<receptor id>.failure.yaml`` says why a result failed. A failed run
  writes it under the particles' settings folder. A failed footprint
  writes it under the footprint's own settings folder, since the
  particles were fine. It is removed when the result is written.

``scratch/`` holds kept workdirs, one folder per simulation, under the
particles' settings folder. A failed run's workdir is kept, and with
``keep_workdir: true`` under ``execution:`` every run's is.

The results trees hold only Parquet files and ``_settings.yaml``, so each
reads as one dataset.

``_settings.yaml``
------------------

Every settings folder holds a ``_settings.yaml``. A particles folder's:

.. code-block:: yaml

   name: hrrr
   hash: 7d36c80b3b5a1a33cc355998f5755de469f61c53151a9d1b953fa698c81cd51f
   pystilt: 0.1.0a23
   settings:
     capemin: -1
     # ... every transport parameter, in name order
     met:
       crop: null
       source: HRRR
     model:
       data_files: null
       name: hysplit
       version: v5.1.0
     n_hours: -24
     # ...

A footprints folder's adds the particles folder it was made from:

.. code-block:: yaml

   name: hrrr
   hash: f82d40aea8c29b2a30b99b1c5a74166937ee4eef5df20aa80407bfb69a3ab7a0
   particles: hrrr-7d36c8
   particles_hash: 7d36c80b3b5a1a33cc355998f5755de469f61c53151a9d1b953fa698c81cd51f
   pystilt: 0.1.0a23
   settings:
     cells_per_target: 4
     geometry: null
     geometry_hash: null
     grid:
       crs: +proj=longlat
       xmin: -112.5
       xmax: -111.5
       xres: 0.01
       ymin: 40.25
       ymax: 41.25
       yres: 0.01
     smooth_factor: 1
     time_integrate: false
     transforms: []

``name``
   The variant that made the folder.
``hash``
   The folder's settings hash, a SHA-256 in hex. The folder name ends in
   its first six characters.
``pystilt``
   The PYSTILT version that wrote the folder.
``settings``
   What the results were made with. In a particles folder, the transport
   model's parameters, ``met`` (the weather product and its crop), and
   ``model`` (the transport model and its build). An ensemble adds
   ``ensemble: true``, and realization ``k`` ran with ``seed + k``. In a
   footprints folder, the footprint settings, with the grid.
``particles``, ``particles_hash``
   In a footprints folder only: the name and hash of the particles folder.

A particles folder's hash covers its ``settings`` block. A footprints
folder's hash covers its ``settings`` block together with
``particles_hash``, so the same footprint settings on other particles give
another folder. Every result file records its folder's full hash as
``stilt:hash``.

Particle files
--------------

A particle file is ``particles/settings=.../date=.../<receptor id>.parquet``,
one row per particle per output step. :doc:`particles` describes what each
column means.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Column
     - Stored as
     - Notes
   * - ``receptor``
     - dictionary of string
     - The receptor id, the same on every row.
   * - ``particle``
     - int32
     - :func:`stilt.read_particles` gives it back as float64.
   * - ``time``
     - int32
     - Minutes since release, negative for a backward run. Read back as
       float64.
   * - ``lon``, ``lat``, ``zagl``, and the rest
     - float64
     - The columns ``varsiwant`` asked HYSPLIT for, plus ``xhgt`` for a
       column or multipoint receptor.

``datetime`` is not stored: it is the receptor time plus ``time``.

The file's metadata (Parquet key-value metadata):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Value
   * - ``stilt:receptor``
     - The receptor as JSON (below).
   * - ``stilt:settings``
     - The run's settings as JSON, the ``settings`` block of the folder's
       ``_settings.yaml``.
   * - ``stilt:met_files``
     - The met files the run read, a JSON list of paths.
   * - ``stilt:hash``
     - The folder's settings hash.
   * - ``stilt:pystilt``
     - The PYSTILT version that wrote the file.
   * - ``stilt:realization``
     - The realization number, in an ensemble only.

Footprint files
---------------

A footprint file holds only the cells that are not zero:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Column
     - Stored as
     - Notes
   * - ``receptor``
     - dictionary of string
     - The receptor id.
   * - ``hour``
     - int16
     - The layer, in hours after the receptor time (below).
   * - ``y``
     - int16
     - The cell's row in the grid, 0 at ``ymin``.
   * - ``x``
     - int16
     - The cell's column in the grid, 0 at ``xmin``.
   * - ``foot``
     - float32
     - The footprint value, in ppm per (µmol m⁻² s⁻¹).

The file's metadata:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Value
   * - ``stilt:receptor``
     - The receptor as JSON.
   * - ``stilt:footprint``
     - The footprint settings as JSON, with the grid: the ``settings``
       block of the folder's ``_settings.yaml``.
   * - ``stilt:hours``
     - Every layer of the footprint, a JSON list such as ``[-24, ..., -1]``.
       It includes layers with no cell left, so a dense array has
       ``len(stilt:hours)`` layers.
   * - ``stilt:name``
     - The variant that made the footprint.
   * - ``stilt:empty``
     - ``true`` when no particle reached the grid, else ``false``.
   * - ``stilt:hash``
     - The folder's settings hash.
   * - ``stilt:particles_hash``
     - The settings hash of the particles it was made from.
   * - ``stilt:pystilt``
     - The PYSTILT version that wrote the file.
   * - ``stilt:realization``
     - The realization number, in an ensemble only.

An empty footprint is a file with no rows and ``stilt:empty`` set to
``true``. A file without the key is not empty. An empty footprint is
complete: the run worked, and nothing reached the grid. Treat it as
missing, not as a footprint of zeros.

The hour
~~~~~~~~

Layer ``k`` holds the particles' influence from ``k`` hours to ``k + 1``
hours after the receptor time, and is stamped at its start. A particle
row at ``time`` minutes falls in layer ``floor(time / 60)``. For a receptor
at 18:30 on a backward run:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Particle ``time`` (minutes)
     - Layer
     - Covers
   * - -1 to -60
     - -1
     - 17:30 to 18:30, stamped 17:30
   * - -61 to -120
     - -2
     - 16:30 to 17:30, stamped 16:30

The layers keep the receptor's minute, so a receptor at 18:30 has layers
starting at half past. A footprint made with ``time_integrate: true`` has
one layer, hour 0, stamped at the receptor time, which holds the whole run.

The grid
~~~~~~~~

``x`` and ``y`` index the grid in ``stilt:footprint`` (or the folder's
``_settings.yaml``):

- The number of columns is ``floor((xmax - xmin) / xres)``, and likewise
  for rows. A partial cell at the edge is dropped, so ``xmax = -110.505``
  with ``xmin = -113`` and ``xres = 0.01`` gives 249 columns.
- Column ``i`` is centred on ``xmin + (i + 0.5) * xres``, and row ``j`` on
  ``ymin + (j + 0.5) * yres``. Row 0 is the southern edge.
- On a projected grid (``crs`` other than ``+proj=longlat``) the bounds are
  degrees and the resolution is in the projection's units. Only the
  corners ``(xmin, ymin)`` and ``(xmax, ymax)`` are transformed to the
  projection, and the cells are laid out between them.

For example, in pyarrow:

.. code-block:: python

   import json

   import numpy as np
   import pyarrow.parquet as pq

   table = pq.ParquetFile(path).read()
   meta = table.schema.metadata
   grid = json.loads(meta[b"stilt:footprint"])["grid"]
   hours = json.loads(meta[b"stilt:hours"])
   nx = int(np.floor((grid["xmax"] - grid["xmin"]) / grid["xres"] + 1e-9))
   ny = int(np.floor((grid["ymax"] - grid["ymin"]) / grid["yres"] + 1e-9))
   foot = np.zeros((len(hours), ny, nx), dtype="float32")
   layer = {h: i for i, h in enumerate(hours)}
   t = [layer[h] for h in table["hour"].to_pylist()]
   foot[t, table["y"].to_numpy(), table["x"].to_numpy()] = table["foot"].to_numpy()

The small tolerance keeps the last whole cell when the bounds are decimals
that binary floats store a little short.

Units
~~~~~

Footprint values are in ppm per (µmol m⁻² s⁻¹), ``ppm m2 s umol-1``. The
files do not say so; :func:`stilt.read_footprint` adds the units to the
array it returns.

Receptor ids and the receptor
-----------------------------

A receptor id is the release time to the minute, then the location:

.. code-block:: text

   202407151830_-111.85_40.77_5            a point: lon_lat_altitude
   202407151830_-111.85_40.77_X0-3000msl   a column: lon_lat_X<bottom>-<top>
   202407151830_multi_7bb3ffdc53           a multipoint: a hash of its points

A location ends in ``msl`` when its heights are above sea level. Whole
numbers are written without a decimal point.

``stilt:receptor`` holds the whole receptor as JSON. Its ``time`` keeps
the seconds, and is UTC with no time zone:

.. code-block:: text

   {"kind": "point", "time": "2024-07-15T18:30:00", "altitude_ref": "agl",
    "longitude": -111.85, "latitude": 40.77, "altitude": 5.0}

   {"kind": "column", "time": "2024-07-15T18:30:00", "altitude_ref": "msl",
    "longitude": -111.85, "latitude": 40.77, "bottom": 0.0, "top": 3000.0}

   {"kind": "multipoint", "time": "2024-07-15T18:30:00", "altitude_ref": "agl",
    "longitudes": [-111.85, -111.86], "latitudes": [40.77, 40.78],
    "altitudes": [5.0, 10.0]}

Reading many files
------------------

Read as a dataset with hive partitioning, a tree gives the folder names as
columns: ``settings``, ``date``, and in an ensemble ``realization``.
DuckDB (``hive_partitioning = true``), polars, R's arrow, and
``pyarrow.dataset`` all do this. A reader of one file gets only the stored
columns. :doc:`../guides/outputs` has DuckDB examples.

Keys you do not know
--------------------

A reader should skip metadata keys and ``_settings.yaml`` keys it does not
know. Later versions may add some, and files rewritten from development
builds may still carry old ones, such as an empty ``stilt:empty_reason``.

Which rules a file follows
--------------------------

A file follows the rules of the PYSTILT version that wrote it:
``stilt:pystilt`` in a result file, and ``pystilt`` in a
``_settings.yaml``. There is no separate format number. This page
describes the files of the current version, and the list below says when
a rule changed.

- **0.1.0a23**: the first release with this layout. Results moved from
  ``simulations/by-id/`` to the output directory, particles and footprints
  are Parquet files in settings folders, and an empty footprint is marked
  ``stilt:empty``. Files from development builds before it carry a
  version ending in ``.dev...``, and may differ.
