Project Folders And Reruns
==========================

A PYSTILT :term:`project` is one folder. It holds your settings, your
receptors, and every output, so the folder alone is enough to reopen,
extend, or rerun the work.

What's in a project folder
--------------------------

.. code-block:: text

   my_project/
     config.yaml                     # settings: meteorology, footprints, run options
     receptors.csv                   # where and when to release particles
     simulations/
       by-id/
         <simulation id>/            # one folder per simulation
           stilt.log                               # run log: check here when a run fails
           <simulation id>_traj.parquet            # particle paths
           <simulation id>_<footprint>_foot.nc     # one file per named footprint
           <simulation id>_<footprint>_foot.empty  # instead of .nc when the footprint is empty
           <simulation id>_error.parquet           # only when wind-error settings are used

A Slurm run also creates ``chunks/`` and ``slurm/`` folders with the job
scripts and logs (:doc:`execution/slurm`).

Simulation IDs
--------------

Each simulation is named after its meteorology and receptor:

.. code-block:: text

   {met name}_{YYYYMMDDHHMM}_{location}

   hrrr_202307151800_-111.848_40.766_10

For point receptors the location is longitude, latitude, and altitude. Column
receptors end in ``_X``. Multipoint receptors use a short hash of their
points, which stays the same if you reorder them.

A project runs every receptor with every met source, so 100 receptors and
two met sources make 200 simulations.

Opening a project again
-----------------------

``config.yaml`` and ``receptors.csv`` are written the first time you run (or
register) a model. After that, the folder is all you need:

.. code-block:: python

   import stilt

   model = stilt.Model(project="./my_project")
   model.status()           # how many simulations are finished

From the command line:

.. code-block:: bash

   stilt status ./my_project

To add receptors to an existing project, pass them in. They are merged into
``receptors.csv``, and receptors already in the file are not duplicated:

.. code-block:: python

   model = stilt.Model(project="./my_project", receptors=new_receptors)
   model.run()

Reruns skip finished work
-------------------------

Before running, PYSTILT checks which simulations are finished and runs only
the rest. A simulation is finished when all of its outputs exist:

- the trajectory file,
- the error trajectory file, if wind-error settings are used,
- a footprint file (or ``.empty`` marker) for every footprint in
  ``config.yaml``.

So after an interruption, a failed Slurm task, or adding a new footprint to
``config.yaml``, just run again. Only what's missing will run.

To list what is not finished yet:

.. code-block:: python

   model.simulations.incomplete()       # list of simulation IDs

To force everything to run again, pass ``skip_existing=False`` to
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
