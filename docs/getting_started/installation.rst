Installation
============

Install PYSTILT
---------------

.. code-block:: bash

   pip install "pystilt[visualization]"

This installs PYSTILT, the ``stilt`` command-line tool, and matplotlib for
plotting. Check that it worked:

.. code-block:: bash

   stilt --help

Optional extras
---------------

Add extras in the brackets to turn on more features, for example
``pip install "pystilt[visualization,geometry]"``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Adds
   * - ``visualization``
     - Plotting (``.plot.map()`` on footprints, trajectories, and receptors).
   * - ``projection``
     - Footprint grids in projected coordinates (anything other than
       latitude/longitude).
   * - ``geometry``
     - Adding footprints up over shapefiles, counties, or H3 hexagons.
   * - ``cloud``
     - Downloading meteorology from NOAA, and projects stored in ``s3://``
       or ``gs://`` buckets.
   * - ``complete``
     - Everything above.

For maps with coastlines and state borders, also install
`cartopy <https://scitools.org.uk/cartopy/>`_ (easiest with
``conda install -c conda-forge cartopy``). PYSTILT uses it automatically when
it is available.

HYSPLIT is included
-------------------

PYSTILT runs NOAA's HYSPLIT program to move particles, and a copy is included
in the package for:

- Linux (x86_64)
- macOS (Intel, x86_64)

On other platforms, or to use your own HYSPLIT build, compile ``hycs_std``
yourself and point PYSTILT at the folder that contains it with ``exe_dir``
in ``config.yaml``.

Next: :doc:`quickstart`.
