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

PYSTILT runs NOAA's HYSPLIT program, ``hycs_std``, but does not include it.
Download HYSPLIT from
`NOAA's Air Resources Laboratory <https://www.ready.noaa.gov/HYSPLIT.php>`_,
which asks you to accept the
`HYSPLIT terms of use <https://www.ready.noaa.gov/HYSPLIT_agreement.php>`_
first. Unpack it, and point ``directory`` at the folder it makes, in a
project's ``config.yaml``:

.. code-block:: yaml

   models:
     hysplit:
       directory: /path/to/hysplit

PYSTILT runs ``exec/hycs_std`` under that folder and reads HYSPLIT's data
tables from its ``bdyfiles/``. Without ``directory`` it runs the first
``hycs_std`` on ``PATH``, which a cluster's HYSPLIT module sets when you
load it.

.. tip::

   Make sure ``hycs_std`` has permission to run:
   ``chmod +x exec/hycs_std``.

``stilt check`` says when a project can find no ``hycs_std``, or one that
cannot run. :doc:`../guides/hysplit_build` has more on choosing a download
and on using more than one build.

Next: :doc:`quickstart`.
