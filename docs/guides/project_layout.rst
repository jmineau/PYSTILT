Project Folders And Reruns
==========================

A PYSTILT :term:`project` is one folder. It holds your settings, your
receptors, and every output. With the folder alone you can reopen, extend,
or rerun the work.

What's in a project folder
--------------------------

.. code-block:: text

   my_project/
     config.yaml                     # your settings: meteorology, variants, run options
     receptors.csv                   # your receptors: where and when to release particles
     simulations/
       variants.yaml                 # the settings each variant ran with (written by PYSTILT)
       by-id/
         <receptor id>/              # one folder per receptor
           <variant>/                # one folder per simulation
             stilt.log                     # run log, the first place to look when a run fails
             <receptor id>_traj.parquet    # particle paths
             <receptor id>_foot.nc         # the footprint, when the variant has a grid
             <receptor id>_foot.empty      # instead of .nc when no particle reaches the grid
             met/, CONTROL, SETUP.CFG ...  # HYSPLIT inputs, kept for debugging

A variant declared with ``from:`` has no particle file of its own. It uses
the particles of the variant it comes from, so its folder holds only the
footprint and a log.

``config.yaml`` and ``receptors.csv`` are yours to edit. PYSTILT never
rewrites a ``config.yaml`` it loaded from the folder, and it only appends
new receptors to ``receptors.csv``. If you pass settings to
:class:`~stilt.Model` in Python, they replace ``config.yaml`` when the model
runs. Receptors you pass are appended to ``receptors.csv``, or start it if
there is none.

``simulations/variants.yaml`` belongs to PYSTILT. It holds the full
settings of every variant that has run, which is how PYSTILT notices a
changed setting (see :doc:`configuration`). Don't edit it.

A Slurm run also creates ``chunks/`` and ``slurm/`` folders with the job
scripts and logs (see :doc:`execution/slurm`).

Simulation IDs
--------------

A simulation is one receptor run under one variant. Its ID is the receptor
ID and the variant name joined by a slash. This is also its folder under
``simulations/by-id``:

.. code-block:: text

   {YYYYMMDDHHMM}_{location}/{variant}

   202307151800_-111.848_40.766_10/hrrr

For a point receptor, the location is the longitude, latitude, and
altitude. A column receptor has ``X`` in place of the altitude. A
multipoint receptor uses ``multi_`` and a short hash of its points, which
does not change if you reorder them.

A project runs every receptor under every variant, so 100 receptors and
three variants make 300 simulations. With no ``variants`` in
``config.yaml``, there is one variant per meteorology source (see
:doc:`configuration`).

Opening a project again
-----------------------

``config.yaml`` and ``receptors.csv`` are in the folder once the project
has run (or been registered). After that, the folder is all you need:

.. code-block:: python

   import stilt

   model = stilt.Model(project="./my_project")
   model.status()           # one row per simulation, with a "complete" column

From the command line:

.. code-block:: bash

   stilt status ./my_project

To add receptors to an existing project, pass them in and run:

.. code-block:: python

   model = stilt.Model(project="./my_project", receptors=new_receptors)
   model.run()

New receptors are appended to ``receptors.csv`` in the file's own columns.
Receptors already in the file are left as they are.

Reruns skip finished work
-------------------------

Before running, PYSTILT checks which simulations are finished and runs only
the rest. A simulation is finished when all of its outputs exist:

- the particle file, unless the variant is declared with ``from:``;
- the footprint file or ``.empty`` marker, if the variant has a grid.

If the particle file is missing, HYSPLIT runs again. The footprint is then
remade from the new particles, and so are the footprints of any ``from:``
variants that use them.

So after an interruption, a failed Slurm task, or adding a variant to
``config.yaml``, run the project again. Only what is missing will run. A
new variant runs for every receptor, and nothing else is touched. PYSTILT
refuses to run a variant whose settings changed after it ran. Remove its
outputs first (see :doc:`configuration`).

To see what is not finished yet:

.. code-block:: python

   model.simulations.incomplete().keys()   # (receptor, variant) ids
   model.simulations.status()              # a table of every simulation

To rerun one variant, delete its outputs with ``stilt rm --variant NAME``
or ``model.remove(NAME)``. To rerun one simulation, call ``sim.delete()``
on it. To run everything again, pass ``skip_existing=False`` to
``model.run()``, or ``--no-skip`` to ``stilt run``.

Storing a project in the cloud
------------------------------

A project can also live in an ``s3://`` or ``gs://`` bucket. This needs the
``cloud`` extra. HYSPLIT still has to run on a local disk. By default
PYSTILT uses a temporary folder for this. Set ``compute_root`` to use your
own scratch space. Each simulation's outputs are uploaded to the bucket
when it finishes.

.. code-block:: python

   model = stilt.Model(
       project="gs://my-bucket/wbb_july_case",
       compute_root="/scratch/me/pystilt",
   )

Outputs read back from a bucket are cached on local disk. Set
``PYSTILT_CACHE_DIR`` to choose where. Otherwise a temporary folder is used.
For more on how this works, see :doc:`../advanced/output_state`.
