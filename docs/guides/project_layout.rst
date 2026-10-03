Project Folders And Reruns
==========================

A PYSTILT :term:`project` is a folder with your settings and your
receptors. Its results go to an output directory that the settings name.
With the two together you can reopen, extend, or rerun the work, and
several projects can share one output directory.

What's in a project folder
--------------------------

.. code-block:: text

   my_project/
     config.yaml        # your settings: meteorology, variants, run options, and output:
     receptors.csv      # your receptors: where and when to release particles

Both files are yours to edit. PYSTILT never rewrites a ``config.yaml`` it
loaded from the folder, and it only appends new receptors to
``receptors.csv``. :meth:`Project.init <stilt.Project.init>` writes
``config.yaml`` once, from the settings you pass it, and stops if the folder
already has one. ``stilt init`` writes a commented starter ``config.yaml``
instead. Receptors you add are appended to ``receptors.csv``, or start it if
there is none.

A Slurm run also creates a ``slurm/`` folder with the job
scripts and logs (see :doc:`execution/slurm`).

What's in the output directory
------------------------------

``output:`` in ``config.yaml`` says where results go, ``./output`` inside
the project by default. Any path works, and two projects that name the same
directory share it.

.. code-block:: text

   output/
     particles/
       settings=hrrr-a3f9c2/                 # one folder per set of transport settings
         _settings.yaml                      # the settings, written out in full
         date=2023-07-15/<receptor id>.parquet
     footprints/
       settings=hrrr-93278c/                 # one folder per variant's footprint settings
         _settings.yaml                      # names the particles folder it was made from
         date=2023-07-15/<receptor id>.parquet
     logs/
       settings=hrrr-a3f9c2/date=2023-07-15/<receptor id>.log
       settings=hrrr-a3f9c2/date=2023-07-15/<receptor id>.failure.yaml   # why it failed, while it has
     scratch/                                # HYSPLIT working directories of failed runs

A folder is named after the variant that made it, plus a short hash of the
settings it was made with. The hash is why a changed setting never
overwrites a result: edit ``ziscale`` and the next run writes into a new
folder beside the old one. The ``_settings.yaml`` in each folder lists the
settings in full, so a folder explains itself.

Two variants that differ only in footprint settings (a coarser grid, another
smoothing) share one ``particles/`` folder and have one ``footprints/``
folder each. There is no need to declare that; PYSTILT sees that their
transport settings are equal.

The folder names are what tools like pyarrow, DuckDB, polars, and R's
``arrow`` package read as columns, so ``output/footprints`` opens as one
table with ``settings`` and ``date`` columns and no PYSTILT needed.

Simulation IDs
--------------

A simulation is one receptor run under one variant. Its ID is the receptor
ID and the variant name joined by a slash:

.. code-block:: text

   {YYYYMMDDHHMM}_{location}/{variant}

   202307151800_-111.848_40.766_10/hrrr

For a point receptor, the location is the longitude, latitude, and
altitude. A column receptor has ``X`` and its bottom and top in place of
the altitude:

.. code-block:: text

   202101150600_-112_40.5_X0-3000

A multipoint receptor uses ``multi_`` and a short hash of its points, which
does not change if you reorder them. Heights above mean sea level add
``msl`` at the end, as in ``_100msl`` or ``_X0-3000msl``. Heights above
ground have no marker.

In the output directory the receptor ID is the file name, in a folder for
the day of the receptor time. Each particle file also has a ``receptor``
column. That way a scan of the whole ``particles/`` tree with pyarrow,
DuckDB, or R can tell the receptors apart.

A project runs every receptor under every variant, so 100 receptors and
three variants make 300 simulations. With no ``variants`` in
``config.yaml``, there is one variant per met (see
:doc:`configuration`).

Opening a project again
-----------------------

``config.yaml`` and ``receptors.csv`` are in the folder from the moment the
project is made. After that, the folder is all you need:

.. code-block:: python

   import stilt

   project = stilt.Project("./my_project")
   project.simulations.status()   # one row per simulation, with a "complete" column

Opening a project only reads it. To change a setting, edit ``config.yaml``.

From the command line:

.. code-block:: bash

   stilt status ./my_project

To add receptors to an existing project, add them and run:

.. code-block:: python

   project = stilt.Project("./my_project")
   project.add_receptors(new_receptors)     # returns the ids of the new ones
   project.run()

New receptors are appended to ``receptors.csv`` in the file's own columns.
Receptors already in the file are left as they are.

Reruns skip finished work
-------------------------

Before running, PYSTILT checks which simulations are finished and runs only
the rest. A simulation is finished when its results exist in the output
directory:

- the particle file;
- the footprint file, if the variant has a grid. A footprint that no
  particle reached is still a file, with the reason inside, and counts.

If the particle file is missing, HYSPLIT runs again. The footprint is then
remade from the new particles, and so are the footprints of the other
variants that share them.

So after an interruption, a failed Slurm task, or adding a variant to
``config.yaml``, run the project again. Only what is missing will run. A
new variant runs for every receptor, and nothing else is touched.

To see what is not finished yet:

.. code-block:: python

   project.simulations.incomplete()   # the simulations that are not complete
   project.simulations.status()       # a table of every simulation

To run everything again, pass ``skip_existing=False`` to ``project.run()``,
or ``--no-skip`` to ``stilt run``.

Changing a setting
------------------

Edit ``config.yaml`` and run again. The variant's settings now hash to a
new folder, so every receptor runs again into it, and the old folder
stays. ``stilt status`` lists folders in the output directory that no
variant in the config uses any more:

.. code-block:: text

   particles folders in /data/output that no variant here uses: settings=hrrr-a3f9c2 ...

PYSTILT never deletes them, since another project may share the directory.
When you are sure, delete a folder by hand.

Failed runs
-----------

HYSPLIT runs in a scratch directory (``compute_root``,
``PYSTILT_COMPUTE_ROOT``, or ``$TMPDIR/pystilt/<project>``), which is
removed when a run succeeds. When a run fails, its working directory is
copied to ``scratch/`` in the output directory, with CONTROL, SETUP.CFG,
and MESSAGE. Beside the receptor's log, ``<receptor id>.failure.yaml``
records why its simulations on those particles failed, and is removed when
they succeed. ``sim.failure`` reads it and ``sim.log`` the log. Set
``keep_scratch: true`` under ``execution:`` to keep every run's working
directory.
