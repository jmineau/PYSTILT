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

PYSTILT runs NOAA's HYSPLIT program to move particles. pip installs a copy
of HYSPLIT with PYSTILT on:

- Linux (x86_64)
- macOS 11 or newer (Intel, x86_64)

On any other platform pip installs PYSTILT without HYSPLIT. This includes
Apple Silicon Macs with an arm64 Python. PYSTILT still imports and reads
existing results, but a run stops with an error that no bundled HYSPLIT binary
exists for the machine.

To use your own HYSPLIT build there or anywhere else, compile ``hycs_std``.
Put a file named ``version`` beside it that holds the build's version, such
as ``v5.3.2``. Then set ``exe_dir`` in ``config.yaml`` to that folder:

.. code-block:: yaml

   exe_dir: /path/to/hysplit/exec

Next: :doc:`quickstart`.
