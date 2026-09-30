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

Extras add optional features. List them in the brackets, for example
``pip install "pystilt[visualization,geometry]"``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Adds
   * - ``visualization``
     - Plotting (``.plot.map()`` on footprints, trajectories, and receptors).
   * - ``geometry``
     - Adding footprints up over shapefiles, counties, or H3 hexagons.
   * - ``cloud``
     - Downloading meteorology from NOAA, the PostgreSQL work queue, and
       Kubernetes workers.
   * - ``complete``
     - Everything above.

For maps with coastlines and state borders, also install
`cartopy <https://scitools.org.uk/cartopy/>`_. It is easiest to install with
``conda install -c conda-forge cartopy``. PYSTILT uses it when it is
available.

HYSPLIT is included
-------------------

PYSTILT runs NOAA's HYSPLIT program to move particles. The package includes a
copy for:

- Linux (x86_64)
- macOS (Intel, x86_64)

On other platforms, or to use your own HYSPLIT build, compile ``hycs_std``
yourself. Then set ``exe_dir`` in ``config.yaml`` to the folder that contains
it.

Next: :doc:`quickstart`.
