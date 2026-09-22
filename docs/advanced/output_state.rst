How PYSTILT Tracks Finished Work
================================

This page explains what happens behind ``stilt status`` and reruns. You
don't need it to use PYSTILT, but it helps when running at scale or across
machines.

No database: the files are the record
-------------------------------------

PYSTILT doesn't keep a separate list of which simulations have run. To decide
whether a simulation is finished, it checks whether that simulation's output
files exist (:meth:`stilt.Simulation.is_complete`):

- the trajectory: ``simulations/by-id/<id>/<id>_traj.parquet``
- the error trajectory, when wind-error settings are used
- one footprint NetCDF, or ``.empty`` marker, for each footprint in
  ``config.yaml``

Each output has exactly one expected path, so each check is a quick "does
this file exist?", with no directory scanning. Because the files are the
only record, a notebook, a Slurm task, and a cloud worker all agree on what's
finished without talking to each other, and the record can't get out of
sync with the outputs. Deleting an output file marks that simulation
unfinished again.

What defines the simulations
----------------------------

A project's simulations are its receptors (``receptors.csv``) run with each
of its meteorology sources (``config.yaml``); ``model.simulations`` lists
exactly that set. ``Model.register()``, which ``Model.run()`` calls first,
writes both files into the project so that any worker, on any machine, can
rebuild the model from the project folder alone. New receptors are merged
into ``receptors.csv``.

Where HYSPLIT runs vs where outputs are stored
----------------------------------------------

HYSPLIT has to run in a local folder. Workers run each simulation under
``compute_root`` and then copy the outputs into the project
(:meth:`stilt.Simulation.publish`). For a project on a local or shared disk,
``compute_root`` defaults to the project's own ``simulations/by-id``, so
nothing is copied. For a project in a cloud bucket, it's a temporary folder
and the outputs are uploaded.

The work queue (cloud only)
---------------------------

Local and Slurm runs never need a database: each worker gets a fixed list of
simulations. Cloud workers instead take simulations one at a time from a
shared queue, and that needs a database that can hand each simulation to
exactly one worker. PYSTILT uses a small PostgreSQL queue
(:class:`stilt.service.PostgresQueue`), enabled only when ``PYSTILT_DB_URL``
is set. The queue tracks which simulations are being worked on; whether a
simulation is finished is still decided by its files.

Environment variables
---------------------

These control where and how PYSTILT runs, never what it calculates, so they
live outside ``config.yaml``:

- ``PYSTILT_COMPUTE_ROOT``: folder where workers run HYSPLIT
- ``PYSTILT_CACHE_DIR``: local cache for files downloaded from a cloud project
- ``PYSTILT_DB_URL``: the cloud work queue
