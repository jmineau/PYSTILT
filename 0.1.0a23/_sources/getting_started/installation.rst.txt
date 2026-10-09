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
   * - ``complete``
     - Everything above.

An output directory on an object store needs the fsspec package for it,
such as ``s3fs`` for ``s3://`` or ``gcsfs`` for ``gs://``
(:doc:`../guides/projects`).

HYSPLIT is included
-------------------

PYSTILT runs NOAA's HYSPLIT program to move particles. pip installs a copy
of HYSPLIT with PYSTILT on:

- Linux (x86_64)
- macOS 11 or newer (Intel, x86_64)

On any other platform pip installs PYSTILT without HYSPLIT. This includes
Apple Silicon Macs with an arm64 Python. PYSTILT still imports and reads
existing results, but a run stops with an error that no bundled HYSPLIT binary
exists for the machine.

There, or anywhere you want another HYSPLIT version or your own land-use
tables, compile ``hycs_std`` and set ``exe_dir`` in ``config.yaml``
(:doc:`../guides/hysplit_build`).

Next: :doc:`quickstart`.
