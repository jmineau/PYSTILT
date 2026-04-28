Meteorology
===========

STILT requires gridded meteorological fields in NOAA ARL format. PYSTILT does
not download meteorology for you; it discovers files you already have and
stages the required subset into each simulation's compute-local directory.

``MetConfig``
   Output configuration stored in ``config.yaml``.

``MetSource``
   The runtime object that discovers and stages required files for one named
   met product.

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

Configuring a source
--------------------

.. code-block:: python

   import stilt

   hrrr = stilt.MetConfig(
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="1h",
       n_min=2,
   )

You can register multiple sources in one ``ModelConfig``:

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

At runtime, ``MetSource.required_files()`` computes the simulation time window,
derives the required filename patterns from ``file_format`` and ``file_tres``,
and globs for matching ARL files recursively under ``directory``. Each expected
time step triggers one targeted search rather than a full directory scan, which
keeps I/O proportional to the number of time steps needed rather than the size
of the archive.

If too few files are found, the run fails clearly instead of silently starting
with incomplete meteorology.

.. code-block:: python

   source = stilt.MetSource(
       met_id="hrrr",
       directory="/data/met/hrrr",
       file_format="%Y%m%d_%H",
       file_tres="1h",
       n_min=2,
   )

   files = source.required_files(
       r_time="2023-07-15 18:00:00",
       n_hours=-24,
   )

Staging into compute-local space
---------------------------------

``Simulation.met_files`` stages the selected meteorology files into the
simulation's compute-local ``met/`` directory using link-or-copy semantics.

This matters when:

- your archive is read-only
- workers use a slower shared filesystem for output
- HYSPLIT should read from a short local path in scratch space

Current limitation
------------------

``subgrid_enable`` and related cropping fields remain intentionally visible in
the config model, but they are not implemented yet. That work is currently
blocked on ``arl-met``. Leave those fields unset unless you are actively
developing that pathway.
