Output State And Shared Workers
===============================

PYSTILT does not treat simulation execution as a purely ephemeral process. The
package centers output state so that workers, CLI commands, and Python code can
all agree on what exists and what still needs work. There is **no local
database** — state is split between three small pieces.

Completion is by key
--------------------

A simulation is *complete* iff every artifact it is configured to produce exists
in the store. Completion is computed by key from the outputs (see
:mod:`stilt.completion`); nothing tracks output presence separately, so the
store is always the source of truth. When wind-error params are set, the
expected set includes the error trajectory.

The manifest
------------

The registry of registered simulations lives in the project's ``.stilt/``
directory as ``manifest.parquet`` (see :mod:`stilt.manifest`). It holds only
what is *not* derivable from the outputs — identity, receptor, scene label, and
the configured footprint targets. It is read and written through a
:class:`~stilt.storage.Store`, so it works on local filesystems and cloud object
stores alike. Completion is never stored here.

The work queue
--------------

Claim-based workers need a backend that can atomically lock one pending
simulation at a time. That backend is a lean PostgreSQL work queue
(:class:`stilt.service.PostgresQueue`: enqueue → claim
``FOR UPDATE SKIP LOCKED`` → done/failed), present only when ``PYSTILT_DB_URL``
is set. It tracks *status* only — completion is still by key.

``pull_simulations()`` and ``stilt pull-worker`` fail clearly if the model has
no queue configured. Local projects have no database: the manifest is the
registry and completion is by key.

Rebuild behavior
----------------

.. code-block:: bash

   stilt rebuild ./project

Local projects have nothing to rebuild — completion is read directly from the
outputs by key — so this just reports status. With a configured queue
(``PYSTILT_DB_URL``) it rescans outputs back into the queue.

Runtime environment
-------------------

The runtime-only settings are intentionally separate from ``config.yaml``. The
important environment variables are:

- ``PYSTILT_DB_URL``
- ``PYSTILT_CACHE_DIR``
- ``PYSTILT_COMPUTE_ROOT``
- ``PYSTILT_MAX_ROWS``

This split keeps deployment concerns out of science configuration.
