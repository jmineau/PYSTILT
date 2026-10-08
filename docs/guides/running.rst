Running
=======

The same project runs on your computer or on an HPC cluster. Only the
``execution`` section of ``config.yaml`` changes. Your receptors,
meteorology, footprints, and outputs stay the same.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Where
     - Use it when
     - Setup
   * - Your computer
     - You're working in a notebook or script, or have up to a few hundred
       simulations. This is the default.
     - Nothing
   * - :doc:`slurm`
     - You have thousands of simulations and access to an HPC cluster that
       uses Slurm.
     - An ``execution`` section with your account and partition
   * - :doc:`containers`
     - You run many containers at once, such as a Kubernetes Job.
     - A volume the containers share, and ``stilt run --task``

Run a project
-------------

From the command line:

.. code-block:: bash

   stilt run ./my_project

From Python or a notebook:

.. code-block:: python

   import stilt

   project = stilt.Project("./my_project")
   project.run()

Both run every simulation that isn't finished yet and return when they are
done. ``project.run()`` returns the status table of the simulations it ran,
one row per simulation with a ``state`` column. Both read and write the
same project folder, so you can mix them: run with ``stilt run`` and
analyze in a notebook with ``stilt.Project("./my_project")``.

A finished simulation is skipped, so running again only runs what is left.
``--no-skip`` (``project.run(skip_existing=False)``) runs every simulation
again.

Use several CPU cores
---------------------

To run several receptors at once, set ``cpus``:

.. code-block:: yaml

   execution:
     backend: local
     cpus: 4

or, for one run, ``stilt run ./my_project --cpus 4``. Each of the ``cpus``
processes runs one receptor at a time, with all of that receptor's
variants. Set it no higher than the number of CPU cores you have. ``cpus``
means the same on Slurm, per array task.

Save now, run later
-------------------

``Project.init`` and ``project.add_receptors()`` save settings and
receptors to the project folder without running anything.
``add_receptors`` returns the ids of the receptors it added:

.. code-block:: python

   project = stilt.Project("./my_project")
   receptor_ids = project.add_receptors(receptors)

Any machine that can see the folder can then run the project. To run only
the new receptors, every variant of each, give their ids:

.. code-block:: python

   project.run(receptors=receptor_ids)

``stilt run --receptors ids.txt`` does the same from a file with one id
per line.

Commands
--------

``stilt init <project>``
   Create a project folder with a starter ``config.yaml`` and
   ``receptors.csv``.

``stilt run <project>``
   Run every unfinished simulation and wait until they are done, here or
   as a Slurm job array. Its options:

   - ``--backend`` and ``--n-workers`` override ``config.yaml`` for this run.
   - ``--cpus N`` runs ``N`` receptors at once.
   - ``--no-skip`` reruns every simulation, finished or not.
   - ``--receptors FILE`` runs only the receptors listed in ``FILE``, one
     id per line.
   - ``--output DIR`` uses another output directory than ``config.yaml``
     names.
   - ``--compute-root DIR`` is the compute root: each run gets a workdir
     in ``DIR``, removed after it succeeds.
   - ``--task I/N`` runs task ``I`` of ``N`` here, as one task of a job
     array (:doc:`containers`).

``stilt submit <project>``
   Submit the unfinished simulations to Slurm as a job array, and return as
   soon as it is submitted (:doc:`slurm`).

``stilt status <project>``
   Count finished and remaining simulations, and the failed ones by reason.
   List the settings folders in the output directory and which variants
   use them. ``--json`` prints the same as JSON, for a program to read.

``stilt run`` exits with 0 when every simulation it ran is complete, 1 when
some failed, and 3 when some did not finish because the run was stopped. It
exits with 2 when its command line is wrong.

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. The worker records why it
failed and goes on, and the next ``stilt run`` tries it again.
``stilt status`` counts the failures by reason, and :doc:`checking` says
what each reason means and where to look.

.. toctree::
   :maxdepth: 1
   :hidden:

   slurm
   containers
