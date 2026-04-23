Meteorology
===========

STILT requires gridded meteorological fields in NOAA ARL format. PYSTILT does
not download meteorology for you; it discovers files you already have and
stages the required subset into each simulation's compute-local directory.

Meteorology in the current package has three layers:

``MetConfig``
   Durable configuration stored in ``config.yaml``.

``MetArchive``
   A runtime helper that resolves relative stream directories against a shared
   archive root and stages files into compute-local space.

``MetStream``
   The runtime object that discovers and validates required files for one named
   stream.

Obtaining ARL meteorology
-------------------------

Common sources include:

- **HRRR** for regional CONUS studies
- **NAM** for broader North American domains
- **GFS** for global coverage or longer back-trajectories
- converted **ERA5** or **WRF** products when you have your own ARL pipeline

PYSTILT expects you to point ``MetConfig.directory`` at a directory tree
containing ARL files and to tell it how filenames encode time.

The intended long-term meteorology preparation path is the companion
`arl-met <https://github.com/jmineau/arl-met>`_ package. PYSTILT is currently
waiting on ``arl-met`` before enabling subgrid generation and cropping in the
main package.

Configuring a stream
--------------------

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
------------------------

At runtime, ``MetStream.required_files()`` computes the simulation time window,
derives the required filename patterns from ``file_format`` and ``file_tres``,
scans the configured directory recursively, and returns the matching ARL files.

If too few files are found, the run fails clearly instead of silently starting
with incomplete meteorology.

.. code-block:: python

   source = stilt.MetStream(
       name="hrrr",
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="1h",
       n_min=2,
   )

   files = source.required_files(
       r_time="2023-07-15 18:00:00",
       n_hours=-24,
   )

Shared archives and relative paths
----------------------------------

When ``STILT_MET_ARCHIVE`` is set, relative meteorology directories are
resolved against that archive root. This is useful when the same project config
needs to run on multiple machines or inside workers that mount the same archive
at a common location.

.. code-block:: bash

   export STILT_MET_ARCHIVE=/data/stilt-meteorology

Then ``config.yaml`` can use a relative directory:

.. code-block:: yaml

   mets:
     hrrr:
       directory: hrrr/conus
       file_format: "%Y%m%d_%H"
       file_tres: 1h

Staging into compute-local space
--------------------------------

``Simulation.met_files`` stages the selected meteorology files into the
simulation's compute-local ``met/`` directory using link-or-copy semantics.

This matters when:

- your archive is read-only
- workers use object storage or a slower shared filesystem for durable output
- HYSPLIT should read from a short local path in scratch space

Current limitation
------------------

``subgrid_enable`` and related cropping fields remain intentionally visible in
the config model, but they are not implemented yet. That work is currently
blocked on ``arl-met``. Leave those fields unset unless you are actively
developing that pathway.
