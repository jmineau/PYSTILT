Installation
============

Install PYSTILT
---------------

.. code-block:: bash

   pip install "pystilt[visualization]"

This installs PYSTILT, the ``stilt`` command-line tool, and matplotlib and
cartopy for plotting maps. Check that it worked:

.. code-block:: bash

   stilt --help

Optional extras
---------------

Extras add optional features. List them in the brackets, for example
``pip install "pystilt[visualization,geometry]"``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Adds
   * - ``visualization``
     - Plotting (``.stilt.plot.map()`` on footprints and particles, ``.plot.map()`` on receptors), with coastlines and state borders from cartopy.
   * - ``geometry``
     - Adding footprints up over shapefiles, counties, or H3 hexagons, with exactextract for fast overlaps.
   * - ``download``
     - Downloading meteorology from NOAA.
   * - ``sparse``
     - The Jacobian as a sparse xarray array (``H.to_xarray()``).
   * - ``stilt-r``
     - Reading and importing STILT-R and X-STILT output (:func:`stilt.migration.read_stilt_r`, :func:`stilt.migration.import_stilt_r`), whose ``.rds`` files rdata reads without R.
   * - ``complete``
     - Everything above.

An output directory on an object store needs the fsspec package for it,
such as ``s3fs`` for ``s3://`` or ``gcsfs`` for ``gs://``
(:doc:`../guides/projects`).

Get HYSPLIT
-----------

PYSTILT runs NOAA's HYSPLIT program, ``hycs_std``, to move particles, but it
does not include it: you get ``hycs_std`` yourself and tell PYSTILT where it
is. Point ``directory`` at the HYSPLIT folder, in a project's
``config.yaml``:

.. code-block:: yaml

   models:
     hysplit:
       directory: /path/to/hysplit

PYSTILT runs ``exec/hycs_std`` under that folder and reads HYSPLIT's data
tables from its ``bdyfiles/``, as NOAA's download unpacks them. Without
``directory`` it runs the first ``hycs_std`` on ``PATH``, which a cluster's
HYSPLIT module sets when you load it.

We recommend STILT-R's copy, HYSPLIT 5.1.0, which is the build PYSTILT's
comparisons with STILT-R run. Download ``hycs_std`` and HYSPLIT's four data
tables from STILT-R's repository into a folder laid out as HYSPLIT's, and
make ``hycs_std`` executable:

.. code-block:: bash

   stilt_r=https://raw.githubusercontent.com/uataq/stilt/main
   mkdir -p hysplit/exec hysplit/bdyfiles
   curl -L -o hysplit/exec/hycs_std "$stilt_r/bin/linux_x64/hycs_std"
   for table in ASCDATA.CFG LANDUSE.ASC ROUGLEN.ASC TERRAIN.ASC; do
       curl -L -o "hysplit/bdyfiles/$table" "$stilt_r/exe/$table"
   done
   chmod +x hysplit/exec/hycs_std

That is the Linux (x86_64) build. On a Mac, download
``bin/macos_x64/hycs_std`` instead; Apple Silicon runs it through Rosetta.

NOAA's HYSPLIT download works too: unpack it on the machine that runs
PYSTILT (``tar -xzf``) and set ``directory`` to the folder it makes. Unpacked
on Windows and copied over, ``exec/hycs_std`` loses its permission to run;
``chmod +x exec/hycs_std`` gives it back. Each Linux download
is built for one operating system release: choose yours (a Rocky Linux 8
cluster takes the RHEL 8 build), or the one built for all Linux
distributions. One built for a newer system stops on an older one with an
error that names ``GLIBC``.

``stilt check`` lists a project that can find no ``hycs_std``, or one that
cannot run, and ``stilt run`` stops with the same message before it runs
anything. A project still opens and reads its results without one. Each run
records the checksum of the ``hycs_std`` it ran, so runs of different builds
are kept apart (:doc:`../guides/hysplit_build`).

HYSPLIT is NOAA Air Resources Laboratory software, under the
`HYSPLIT terms of use <https://www.ready.noaa.gov/HYSPLIT_agreement.php>`_.

Next: :doc:`quickstart`.
