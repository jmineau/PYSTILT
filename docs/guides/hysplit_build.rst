Your Own HYSPLIT Build And Data Tables
======================================

PYSTILT runs the copy of HYSPLIT that pip installs with it, and reads
HYSPLIT's data tables from the same place. You can replace either: a
newer HYSPLIT, a build with a fix the bundled one lacks, or a
higher-resolution land use.

A HYSPLIT build
---------------

Compile ``hycs_std`` from NOAA's HYSPLIT source. Beside it, put a file
named ``version`` that holds the build's version, such as ``v5.3.2`` or
``v5.3.2+t0-rows`` for a patched build. Then point ``exe_dir`` at that
folder:

.. code-block:: yaml

   exe_dir: /path/to/hysplit/exec

``exe_dir`` is a top-level setting, so a variant can set its own and
compare two builds on the same receptors:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-v532: {exe_dir: /path/to/hysplit-5.3.2/exec}

Two builds can give different particles from the same settings, so a run
records the build's version and runs of different builds go to different
settings folders (:doc:`projects`). The folder itself is not recorded:
moving a build keeps its results. A build without a ``version`` file is
refused, since its runs could not be told apart from another build's.

A build that writes each particle's release as a row at ``time = 0`` gives
exact release heights for multipoint and slant receptors. The bundled
build writes its first row one time step after release, and PYSTILT
matches each particle to its release point from that first position
(:doc:`../reference/particles`).

Data tables
-----------

HYSPLIT's data tables (``ASCDATA.CFG``, ``LANDUSE.ASC``, ``ROUGLEN.ASC``,
``TERRAIN.ASC``) set the land use, roughness, and terrain under the
particles. To use your own, put them in a folder and set ``data_dir``. A
table the folder does not hold comes from the bundled set:

.. code-block:: yaml

   data_dir: /path/to/hysplit/bdyfiles

The tables change the particles, so a run records the checksum of each
table that differs from the bundled one, and such runs are kept apart from
the others. A folder whose tables match the bundled ones changes nothing.

Platforms without a bundled build
---------------------------------

pip installs a HYSPLIT build on Linux (x86_64) and Intel macOS. On any
other platform, such as an Apple Silicon Mac with an arm64 Python, PYSTILT
reads results but a run stops with an error until ``exe_dir`` points at a
build (:doc:`../getting_started/installation`).
