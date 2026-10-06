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

Each result has one expected path, so checking one simulation is a file
lookup. For many simulations (``stilt status``, or planning a run) PYSTILT
lists the date folders the selection falls in, once each, and reads the
answer for every receptor from that. That keeps a project of a hundred
thousand receptors quick to check, and a few simulations of it cheap.
A notebook, a Slurm task, and another project sharing the directory all see
the same files, so they agree on what's finished without talking to each
other. Deleting a result file makes that simulation unfinished again.

What defines the simulations
----------------------------

A project's simulations are its receptors (``receptors.csv``) run under each
of its variants (``config.yaml``). ``project.simulations`` is exactly that
set, one row per simulation.

Both files are in the project folder before anything runs, so any worker on
any machine can open the project from the folder alone.
``Project.init()`` writes ``config.yaml`` once, and PYSTILT never rewrites
it. ``project.add_receptors()`` appends new receptors to
``receptors.csv``.

Settings name the folders
-------------------------

Each variant resolves into two records of settings (:mod:`stilt.identity`).
Its run settings hold everything that changes the particles: the HYSPLIT
parameters, the content of the meteorology, and the transport model's
build. Their hash names the ``particles/`` folder. Its footprint settings
hold the footprint config, its grid, and the hash of the geometry the grid
was derived for. Hashed together with the run settings, they name the
``footprints/`` folder. Each folder holds a ``_settings.yaml`` with the
settings written out in full.

That is how PYSTILT knows a setting changed: the variant now hashes to a
folder that does not exist yet, so its receptors run again into it. The old
folder is untouched. It is also how two variants come to share particles:
if only their footprint fields differ, their transport settings hash the
same and they resolve to one ``particles/`` folder.

A folder is found by reading its ``_settings.yaml`` back through the current
config classes and hashing that, rather than by comparing stored digests.
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

HYSPLIT has to run in a local folder that holds its input files, the
workdir: ``workdir`` if given, else ``PYSTILT_WORKDIR``,
else ``$TMPDIR/pystilt/<project name>``. After a successful run the particle
file and the log are written to the output directory and the folder is
removed. After a failure the folder is copied to ``scratch/`` in the output
directory first, so CONTROL, SETUP.CFG, and MESSAGE survive the job.
``keep_scratch: true`` under ``execution:`` in ``config.yaml`` keeps every
run's folder.

Files are written through a temporary name and renamed into place, so a
reader never sees a partial file, and two workers that run the same receptor
by accident produce equivalent files with the last rename winning. No lock
is needed.

Environment variables
---------------------

This controls where PYSTILT runs. It never changes what it calculates, so it
lives outside ``config.yaml``.

- ``PYSTILT_WORKDIR``: the workdir, where workers run HYSPLIT
