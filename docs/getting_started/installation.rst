Installation
============

Requirements
------------

- Python ≥ 3.10
- A UNIX-like operating system (Linux or macOS). Windows is not supported;
  use WSL if you are on Windows.
- Meteorological data in `ARL format
  <https://www.ready.noaa.gov/archives.php>`_ (e.g., HRRR, NAM, GFS).
  PYSTILT does not download meteorology for you; see
  :doc:`../user_guide/meteorology` for guidance on obtaining data.

From PyPI
---------

.. code-block:: bash

   pip install pystilt

This installs the core package. Optional extras:

.. code-block:: bash

   pip install "pystilt[projection]"    # pyproj support for projected grids
   pip install "pystilt[cloud]"         # S3/GCS/PostgreSQL/Kubernetes support
   pip install "pystilt[visualization]" # Matplotlib plotting helpers
   pip install "pystilt[complete]"      # All of the above

From Source
-----------

.. code-block:: bash

   git clone https://github.com/jmineau/PYSTILT.git
   cd PYSTILT
   pip install -e .

Development Installation
------------------------

Requires `uv <https://docs.astral.sh/uv/>`_ and
`just <https://just.systems/man/en/>`_:

.. code-block:: bash

   git clone https://github.com/jmineau/PYSTILT.git
   cd PYSTILT
   uv sync --group dev
   pre-commit install

Run the test suite:

.. code-block:: bash

   just test

Build the docs locally:

.. code-block:: bash

   just build-docs
