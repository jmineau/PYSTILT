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
is. We recommend STILT-R's copy, HYSPLIT 5.1.0, which is the build PYSTILT's
comparisons with STILT-R run. Download it from STILT-R's repository and make
it executable:

.. code-block:: bash

   mkdir -p hysplit
   curl -L -o hysplit/hycs_std \
       https://raw.githubusercontent.com/uataq/stilt/main/bin/linux_x64/hycs_std
   chmod +x hysplit/hycs_std

That is the Linux (x86_64) build. On a Mac, download
``bin/macos_x64/hycs_std`` instead; Apple Silicon runs it through Rosetta.

Then tell PYSTILT where it is, in one of two ways:

- Put its folder on ``PATH``, for example in your shell's startup file:
  ``export PATH="/path/to/hysplit:$PATH"``. A cluster's HYSPLIT module
  does this when you load it.
- Set ``exe_dir`` to the folder in a project's ``config.yaml``, under
  ``models: hysplit:``. This wins over ``PATH``:

  .. code-block:: yaml

     models:
       hysplit:
         exe_dir: /path/to/hysplit

``stilt check`` lists a project that can find no ``hycs_std``, and
``stilt run`` stops with the same message before it runs anything. A
project still opens and reads its results without one.

Any other HYSPLIT build works too, such as NOAA's own download or one you
compile (:doc:`../guides/hysplit_build`). Each run records the checksum of
the ``hycs_std`` it ran, so runs of different builds are kept apart.

HYSPLIT is NOAA Air Resources Laboratory software, under the
`HYSPLIT terms of use <https://www.ready.noaa.gov/HYSPLIT_agreement.php>`_.

Next: :doc:`quickstart`.
