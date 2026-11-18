Your Own HYSPLIT Build And Data Tables
======================================

PYSTILT runs the HYSPLIT you give it (:doc:`../getting_started/installation`):
a newer HYSPLIT, a build with a fix STILT-R's lacks, or your own data
tables, such as a higher-resolution land use.

A HYSPLIT build
---------------

Get HYSPLIT from NOAA's download, a cluster's HYSPLIT module, or by
compiling NOAA's source. Then point ``directory`` at the folder, laid out
as HYSPLIT's download is: ``exec/hycs_std``, and the data tables in
``bdyfiles/``:

.. code-block:: yaml

   models:
     hysplit:
       directory: /path/to/hysplit

Each of NOAA's Linux downloads is built for one operating system release.
Choose yours (a Rocky Linux 8 cluster takes the RHEL 8 build), or the one
built for all Linux distributions: a download built for a newer system
stops on an older one with an error that names ``GLIBC``. Unpacked on
Windows and copied over, ``exec/hycs_std`` loses its permission to run;
``chmod +x exec/hycs_std`` gives it back.

``directory`` is a HYSPLIT setting, so a variant can set its own and
compare two builds on the same receptors:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-v532: {directory: /path/to/hysplit-5.3.2}

Without ``directory``, PYSTILT runs the first ``hycs_std`` on ``PATH``, with
the tables of a ``bdyfiles/`` folder beside the folder it is in.

Two builds can give different particles from the same settings, so a run
records the SHA-256 of the ``hycs_std`` it ran, and runs of different
builds go to different settings folders (:doc:`projects`). A rebuilt or
updated ``hycs_std`` starts new folders even when it behaves the same. The
folder ``hycs_std`` is in is not recorded: moving a build keeps its results.

A run also records the version ``hycs_std`` prints when it starts, such as
``hysplit.v5.1.0``, as ``model_version`` in the settings folder's
``_settings.yaml`` (:doc:`../reference/layout`). The version does not tell
builds apart: two builds of one release print the same, and some builds
print an older release than they are.

A machine without ``hycs_std``, such as one that only reports status,
still opens a project. ``stilt status`` finds each variant's folder by the
variant's name and its other settings, and exits with 2 when the variant
has folders from two builds.

A build that writes each particle's release as a row at ``age = 0`` gives
exact release heights for multipoint and slant receptors. HYSPLIT 5.1.0
writes its first row one time step after release, and PYSTILT matches
each particle to its release point from that first position
(:doc:`../reference/particles`).

What a build reads from PYSTILT
-------------------------------

PYSTILT writes HYSPLIT's ``CONTROL`` file with the met files as one grid,
listed in time. HYSPLIT reads at most 128 files of one grid (``mtim`` in
its ``DEFGRID.INC``), in STILT-R's build and in NOAA's, so a run can
read at most 128 met files: 32 days of 6-hour files, or about 5 days of
hourly ones. A run that needs more stops with ``Numb meteo times exceed
DEFGRID limit`` in its log.

STILT-R lists each file as its own grid instead, which NOAA's builds limit
to 12 files a run and STILT-R's build to 99 (``mgrd``). The particles
are the same either way.

To rerun a kept workdir by hand, you can edit its ``CONTROL`` file. A
``#`` starts a comment on the start time, the starting locations, and the
line with the number of met files. HYSPLIT reads the met directory and
file lines whole, so a ``#`` there becomes part of the path.

Data tables
-----------

HYSPLIT's data tables (``ASCDATA.CFG``, ``LANDUSE.ASC``, ``ROUGLEN.ASC``,
``TERRAIN.ASC``) set the land use, roughness, and terrain under the
particles. PYSTILT reads them from the ``bdyfiles/`` folder of
``directory``, or, without ``directory``, from a ``bdyfiles/`` beside the
folder of the ``hycs_std`` on ``PATH``. It includes none, and a run stops
before it starts when one of the four is missing. To use your own, put them
there.

HYSPLIT's own ``ASCDATA.CFG`` says the tables are in ``../bdyfiles/``.
PYSTILT links them into each run's workdir instead, so it writes
``ASCDATA.CFG`` there with that line set to ``./`` and the rest as yours.

The tables change the particles, so a run records the checksum of each,
and runs with other tables are kept apart from the others. The checksum of
``ASCDATA.CFG`` is of the file the run reads, so where a HYSPLIT folder says
its tables are changes nothing.
