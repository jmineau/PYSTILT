Installation
============

Package install
---------------

Install the base transport package with:

.. code-block:: bash

   pip install pystilt

Install optional extras for projections, visualization, cloud storage, Slurm,
and Kubernetes-oriented workflows with:

.. code-block:: bash

   pip install "pystilt[complete]"

Developer install
-----------------

From a source checkout, the repository already includes a ``uv.lock`` and a
``justfile``. A typical developer setup is:

.. code-block:: bash

   uv sync --group dev

Bundled HYSPLIT binaries
------------------------

PYSTILT resolves bundled HYSPLIT binaries from ``stilt.hysplit.bin`` when it
can. The current source tree ships platform-specific bundles for:

- Linux ``x86_64``
- macOS ``x86_64``

If your platform is not bundled, the driver expects a compatible ``hycs_std``
binary available from a directory you provide.

Runtime dependencies you still need to supply
---------------------------------------------

PYSTILT does not manufacture your meteorology archive. You still need:

- ARL-formatted meteorology files for the met streams in ``config.yaml``
- a writable project or output root
- PostgreSQL when running claim-based shared workers
- Slurm or Kubernetes infrastructure when using those executors

Build the docs locally
----------------------

The repo includes a docs build target:

.. code-block:: bash

   just build-docs

or directly:

.. code-block:: bash

   uv run sphinx-build -M html docs docs/_build
