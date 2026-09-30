How PYSTILT Tracks Finished Work
================================

This page explains what happens behind ``stilt status`` and reruns. You
don't need it to use PYSTILT, but it helps when running at scale, across
machines, or with several projects sharing one output directory.

The result files are the record
-------------------------------

PYSTILT keeps no separate list of which simulations have run. A simulation
is finished when its result files exist in the output directory
(:meth:`stilt.Simulation.is_complete`). It needs:

- the particle file,
  ``particles/settings=<variant>-<hash>/date=<day>/<receptor>.parquet``
- the footprint file in the variant's ``footprints/`` folder, when the
  variant has a grid. A footprint that no particle reached is a file with
  no rows and the reason in its metadata, and it counts.

Each result has one expected path, so each check is a single file lookup.
A notebook, a Slurm task, and another project sharing the directory all see
the same files, so they agree on what's finished without talking to each
other. Deleting a result file makes that simulation unfinished again.

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

Settings name the folders
-------------------------

Each variant resolves into two parts. Its transport settings (everything
that changes the particles: the HYSPLIT parameters, the content of the
meteorology, and the engine version; :class:`stilt.config.TransportSettings`)
are hashed, and the hash names the ``particles/`` folder. Its footprint
settings, hashed together with the transport hash, name the ``footprints/``
folder. Each folder holds a ``_settings.yaml`` with the settings written out
in full.

That is how PYSTILT knows a setting changed: the variant now hashes to a
folder that does not exist yet, so its receptors run again into it. The old
folder is untouched. It is also how two variants come to share particles:
if only their footprint fields differ, their transport settings hash the
same and they resolve to one ``particles/`` folder.

A folder is found by loading its ``_settings.yaml`` back through the current
settings model and hashing that, rather than by comparing stored digests.
So a setting added in a later version, with a default, still matches folders
written before it existed; a default whose meaning changed does not, which
is right, because the particles would differ.

``stilt status`` lists the folders no variant of the current config uses.
PYSTILT never deletes a folder, because another project may share the
directory; deletion is by hand.

The unit of work
----------------

Workers are handed receptors. A worker runs every variant of its receptor,
in config order. When a variant's particles are missing it runs HYSPLIT; the
variants that share those particles reuse them and make their own
footprints, remade if the particles were replaced in the same call. A
Slurm task is handed a batch of receptor IDs.

Where HYSPLIT runs
------------------

HYSPLIT has to run in a local folder that holds its input files. That
folder is scratch: ``compute_root`` if given, else ``PYSTILT_COMPUTE_ROOT``,
else ``$TMPDIR/pystilt/<project name>``. After a successful run the particle
file and the log are written to the output directory and the folder is
removed. After a failure the folder is copied to ``scratch/`` in the output
directory first, so CONTROL, SETUP.CFG, and MESSAGE survive the job.
``keep_scratch: true`` in ``config.yaml`` keeps every run's folder.

Files are written through a temporary name and renamed into place, so a
reader never sees a partial file, and two workers that run the same receptor
by accident produce equivalent files with the last rename winning. No lock
is needed.

Environment variables
---------------------

This controls where PYSTILT runs. It never changes what it calculates, so it
lives outside ``config.yaml``.

- ``PYSTILT_COMPUTE_ROOT``: the scratch folder where workers run HYSPLIT
