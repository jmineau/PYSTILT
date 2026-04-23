Durable State, Indexes, And Shared Workers
==========================================

PYSTILT does not treat simulation execution as a purely ephemeral process. The
current package centers durable state so that workers, CLI commands, and Python
code can all agree on what exists and what still needs work.

Index backends
--------------

Two index backends exist today:

``SQLite``
   The default for local output roots. Good for single-machine or shared-file
   workflows where one durable index file is enough.

``PostgreSQL``
   Used when ``PYSTILT_DB_URL`` is set or when cloud output roots require a
   shared claim-capable registry.

Why PostgreSQL matters
----------------------

Claim-based workers require a backend that can atomically lock one pending
simulation at a time. In the current alpha, that means PostgreSQL.

``pull_simulations()`` and ``stilt pull-worker`` will fail clearly if the model
is only backed by the local SQLite index.

Rebuild behavior
----------------

Push-style backends attach an ``index.rebuild`` callback to their job handle.
That means:

- local runs rebuild the index after the workers finish
- Slurm runs rebuild after the array job is observed complete

There is also an explicit escape hatch:

.. code-block:: bash

   stilt rebuild ./project

Use it after manual file movement or if durable outputs and the local index
have drifted apart.

Runtime environment
-------------------

The runtime-only settings are intentionally separate from ``config.yaml``. The
important environment variables are:

- ``PYSTILT_DB_URL``
- ``PYSTILT_CACHE_DIR``
- ``PYSTILT_COMPUTE_ROOT``
- ``PYSTILT_MAX_ROWS``
- ``STILT_MET_ARCHIVE``

This split keeps deployment concerns out of science configuration.
