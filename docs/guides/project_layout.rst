Project Layout And Output State
================================

PYSTILT separates three concepts that are often mixed together in older
workflows:

``project``
   The science-facing project root. This is where ``config.yaml`` and
   ``receptors.csv`` normally live.

``output_dir``
   The output root. This is where the simulation index and simulation
   artifacts live. By default it is the same as ``project``.

``compute_root``
   A compute-local parent directory for worker scratch directories. This is
   especially useful when outputs live on object storage or a slower
   shared filesystem.

Default local layout
--------------------

When project and output roots are the same, PYSTILT uses:

.. code-block:: text

   project/
     config.yaml
     receptors.csv
     simulations/
       index.sqlite
       by-id/
         <sim_id>/

Separate input and output roots
-------------------------------

You can keep human-edited inputs in one place and output results in another:

.. code-block:: python

   model = stilt.Model(
       project="./inputs/wbb_july_case",
       output_dir="gs://my-bucket/wbb_july_case",
       compute_root="/scratch/$USER/pystilt",
   )

This is the right model when:

- project inputs are version-controlled locally
- outputs belong in object storage
- workers need fast local scratch space for staging meteorology and HYSPLIT files

How configuration is loaded
---------------------------

``Model.config`` is loaded lazily:

- from the local project root when ``config.yaml`` is present
- otherwise from output storage if the project has already been bootstrapped

The same rule applies to receptors. ``Model.register_pending()`` is the output
boundary that publishes config and receptor inputs before simulations are
registered in the index.

Simulation identity
-------------------

Each simulation ID has the form:

.. code-block:: text

   {met}_{YYYYMMDDHHMM}_{location_id}

For point receptors, ``location_id`` is a coordinate triple. For column
receptors it ends in ``_X``. For multipoint receptors, PYSTILT uses a stable
hash-based location ID.

Status model
------------

The output index tracks aggregate counts and per-simulation output presence.
In the current alpha, the useful mental model is:

- trajectories move through ``pending``, ``running``, ``complete``, or ``failed``
- footprints are tracked per footprint name
- ``complete-empty`` is a successful terminal footprint outcome
- ``skip_existing=True`` avoids rerunning complete work

Use ``stilt status`` or ``Model.status()`` to inspect project-level counts.
