Project Layout And Output State
================================

PYSTILT has two locations, not three:

``project``
   The project root: a local directory or an object-store URI (``s3://``,
   ``gs://``). ``config.yaml``, ``receptors.csv``, and every simulation
   output live here.

``compute_root``
   A local parent directory for worker scratch. Defaults to the project's
   ``simulations/by-id`` for a local project (so nothing is copied) and to a
   temp directory for a cloud project (outputs are uploaded when a
   simulation finishes).

Layout
------

.. code-block:: text

   project/
     config.yaml
     receptors.csv
     simulations/
       by-id/
         <sim_id>/
           stilt.log
           <sim_id>_traj.parquet
           <sim_id>_error.parquet          # when wind-error params are set
           <sim_id>_<footprint>_foot.nc
           <sim_id>_<footprint>_foot.empty # legitimately empty footprint

Cloud projects
--------------

.. code-block:: python

   model = stilt.Model(
       project="gs://my-bucket/wbb_july_case",
       compute_root="/scratch/$USER/pystilt",
   )

Workers stage meteorology and run HYSPLIT under ``compute_root`` and publish
outputs to the bucket. ``PYSTILT_CACHE_DIR`` controls where remote outputs are
cached when read back.

How inputs are loaded
---------------------

``Model.config`` and ``Model.receptors`` are loaded lazily from the project
store unless given at construction. ``Model.register()`` writes them into the
store; ``Model.run()`` calls it first, so workers launched on other machines can
rebuild the model from the root alone. Registering an explicit receptor batch
merges it into ``receptors.csv`` (deduplicated by receptor id).

Simulation identity
-------------------

Each simulation ID has the form:

.. code-block:: text

   {met}_{YYYYMMDDHHMM}_{location_id}

For point receptors, ``location_id`` is a coordinate triple. For column
receptors it ends in ``_X``. For multipoint receptors, PYSTILT uses a stable
hash-based location ID. ``model.simulations`` is every receptor crossed with
every met stream.

Status model
------------

Completion is read from the outputs by key (:meth:`stilt.Simulation.is_complete`):

- a simulation is complete when its trajectory, its error trajectory (if
  configured), and every configured footprint exist
- an ``.empty`` marker is a successful terminal footprint outcome
- ``skip_existing=True`` (the default) re-dispatches only incomplete simulations

Use ``stilt status`` or ``Model.status()`` for project-level counts and
``model.simulations.incomplete()`` for the ids.
