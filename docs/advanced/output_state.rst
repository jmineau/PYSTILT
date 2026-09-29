How PYSTILT Tracks Finished Work
================================

This page explains what happens behind ``stilt status`` and reruns. You
don't need it to use PYSTILT, but it helps when running at scale or across
machines.

The output files are the record
-------------------------------

PYSTILT keeps no separate list of which simulations have run. A simulation
is finished when its output files exist
(:meth:`stilt.Simulation.is_complete`). It needs:

- the trajectory,
  ``simulations/by-id/<receptor>/<variant>/<receptor>_traj.parquet``,
  unless the variant is declared with ``from:``
- the footprint NetCDF or its ``.empty`` marker, when the variant has a grid

Each output has one expected path, so each check is a single file lookup
with no directory scanning. A notebook, a Slurm task, and a cloud worker all
see the same files, so they agree on what's finished without talking to
each other. Deleting an output file makes that simulation unfinished again.

What defines the simulations
----------------------------

A project's simulations are its receptors (``receptors.csv``) run under each
of its variants (``config.yaml``). ``model.simulations`` is exactly that
set.

``Model.register()`` writes both files into the project, so any worker on
any machine can rebuild the model from the project folder alone.
``Model.run()`` calls it first. New receptors are appended to
``receptors.csv``. A ``config.yaml`` that was loaded from the project is
never rewritten. One given in Python is written out.

The settings record
-------------------

If a setting changed, existing outputs would no longer match their variant
name. So PYSTILT also records the settings that made them, in
``simulations/variants.yaml``. It holds the fully resolved settings of every
registered variant, plus the meteorology.

This record is kept apart from ``config.yaml``, which is the user's file.
PYSTILT compares against the record. That way the check works whether a
setting was changed in a notebook or in an editor.

``register()`` resolves every variant. If a variant that is already
recorded would now resolve differently, it stops with
:class:`stilt.errors.ConfigChangedError`. A new variant is always accepted,
because none of its simulations exist yet. ``Model.remove()`` (``stilt rm``)
deletes a variant's outputs and its record entry together.

The record lists settings only. Which simulations are finished is still read
from their files.

The unit of work
----------------

Workers are handed receptors. A worker runs every variant of its receptor.
Variants that run HYSPLIT go first, then the ``from:`` variants that reuse
their particles. The Slurm chunk files and the cloud queue both hold
receptor IDs.

Where HYSPLIT runs and where outputs are stored
-----------------------------------------------

HYSPLIT has to run in a local folder. Workers run each simulation under
``compute_root`` and then copy the outputs into the project
(:meth:`stilt.Simulation.publish`).

For a project on a local or shared disk, ``compute_root`` defaults to the
project's own ``simulations/by-id``, so nothing is copied. For a project in
a cloud bucket, it defaults to ``$TMPDIR/pystilt/<project name>``, and the
outputs are uploaded.

The work queue (cloud only)
---------------------------

Local and Slurm runs don't need a database, because each worker gets a
fixed list of receptors. Cloud workers take receptors one at a time from a
shared queue, which needs a database that hands each receptor to exactly
one worker. PYSTILT uses a small PostgreSQL queue
(:class:`stilt.service.PostgresQueue`), turned on only when
``PYSTILT_DB_URL`` is set.

``register()`` adds receptors to the queue as pending. A worker claims one,
marks it running, and marks it done or failed when it finishes. A worker
that is interrupted puts its receptor back as pending. The queue only tracks
this status. Whether a simulation is finished is still decided by its
files.

Environment variables
---------------------

These control where and how PYSTILT runs. They never change what it
calculates, so they live outside ``config.yaml``.

- ``PYSTILT_COMPUTE_ROOT``: the folder where workers run HYSPLIT
- ``PYSTILT_CACHE_DIR``: a local cache for files downloaded from a cloud
  project
- ``PYSTILT_DB_URL``: the PostgreSQL URL of the work queue
