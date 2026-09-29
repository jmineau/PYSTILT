Project Folders And Reruns
==========================

A PYSTILT :term:`project` is one folder. It holds your settings, your
receptors, and every output, so the folder alone is enough to reopen,
extend, or rerun the work.

What's in a project folder
--------------------------

.. code-block:: text

   my_project/
     config.yaml                     # your settings: meteorology, variants, run options
     receptors.csv                   # your receptors: where and when to release particles
     simulations/
       variants.yaml                 # PYSTILT's record of the settings each variant ran with
       by-id/
         <receptor id>/              # one folder per receptor
           <variant>/                # one folder per simulation
             stilt.log                     # run log: check here when a run fails
             <receptor id>_traj.parquet    # particle paths
             <receptor id>_foot.nc         # the footprint, when the variant has a grid
             <receptor id>_foot.empty      # instead of .nc when the footprint is empty
             met/, CONTROL, SETUP.CFG ...  # HYSPLIT inputs, kept for debugging

A variant declared with ``from:`` has only its footprint and the log of any
error; its particles are the other variant's.

The two files at the top are yours: PYSTILT reads them and never rewrites
them (if you build the project in Python instead, it writes them for you the
first time, and receptors added later are appended). ``variants.yaml`` is
PYSTILT's: the full settings of every variant that has run, which is how a
changed setting is caught (:doc:`configuration`). Don't edit it.

A Slurm run also creates ``chunks/`` and ``slurm/`` folders with the job
scripts and logs (:doc:`execution/slurm`).

Simulation IDs
--------------

A simulation is a receptor under a variant, and its id is the two joined by
a slash, which is also its folder below ``simulations/by-id``:

.. code-block:: text

   {YYYYMMDDHHMM}_{location}/{variant}

   202307151800_-111.848_40.766_10/hrrr

For point receptors the location is longitude, latitude, and altitude. Column
receptors end in ``_X``. Multipoint receptors use a short hash of their
points, which stays the same if you reorder them.

A project runs every receptor under every variant, so 100 receptors and
three variants make 300 simulations. With no ``variants`` in
``config.yaml`` there is one per met source (:doc:`configuration`).

Opening a project again
-----------------------

``config.yaml`` and ``receptors.csv`` are written the first time you run (or
register) a model. After that, the folder is all you need:

.. code-block:: python

   import stilt

   model = stilt.Model(project="./my_project")
   model.status()           # one row per simulation, with a "complete" column

From the command line:

.. code-block:: bash

   stilt status ./my_project

To add receptors to an existing project, pass them in. New ones are appended
to ``receptors.csv`` in its own columns, and receptors already in the file
are left alone:

.. code-block:: python

   model = stilt.Model(project="./my_project", receptors=new_receptors)
   model.run()

Reruns skip finished work
-------------------------

Before running, PYSTILT checks which simulations are finished and runs only
the rest. A simulation is finished when all of its outputs exist:

- the trajectory file, unless the variant is declared with ``from:``,
- the footprint file (or ``.empty`` marker), if the variant has a grid.

If the trajectory is missing, HYSPLIT runs again and the footprint is
remade from the new particles, along with the footprints of any ``from:``
variants, so a footprint never outlives the particles it came from.

So after an interruption, a failed Slurm task, or adding a variant to
``config.yaml``, just run again. Only what's missing will run: a new
variant runs for every receptor and nothing else is touched. Changing the
settings of a variant that already ran is refused; remove its outputs first
(:doc:`configuration`).

To list what is not finished yet:

.. code-block:: python

   model.simulations.incomplete().keys()   # (receptor, variant) ids
   model.simulations.status()              # a table of every simulation

To rerun one variant, delete its outputs with ``stilt rm --variant NAME``
or ``model.remove(NAME)``; to rerun one simulation, ``sim.delete()``. To
force everything to run again, pass ``skip_existing=False`` to
``model.run()``, or ``--no-skip`` to ``stilt run``.

Storing a project in the cloud
------------------------------

A project can also live in an ``s3://`` or ``gs://`` bucket (needs the
``cloud`` extra). HYSPLIT still has to run on a local disk, so give PYSTILT a
scratch folder with ``compute_root``; outputs are uploaded to the bucket when
each simulation finishes:

.. code-block:: python

   model = stilt.Model(
       project="gs://my-bucket/wbb_july_case",
       compute_root="/scratch/me/pystilt",
   )

Outputs read back from a bucket are cached locally in ``PYSTILT_CACHE_DIR``.
For how this works in more detail, see :doc:`../advanced/output_state`.
