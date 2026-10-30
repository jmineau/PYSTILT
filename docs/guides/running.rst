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

A Python script that calls ``project.run()`` with ``cpus`` above 1 needs its
code under ``if __name__ == "__main__":``. Python 3.14, macOS, and Windows
start each worker by importing the script again, so anything at the top of the
script runs again in every worker. ``Project.init`` there stops with a
``FileExistsError`` because ``config.yaml`` exists. ``stilt run --cpus 4``
needs no guard.

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

``stilt check <project>``
   List what would stop the project's runs, before any run starts: a met
   directory that does not exist, met files a receptor's run needs that
   are not there, a receptor outside the met's crop or the footprint grid,
   and an averaging-kernel table that does not match the receptors. Nothing
   is run or downloaded. It exits with 0 when it finds nothing and 1 when
   it lists anything (:doc:`checking`).

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
   - ``--compute-root DIR`` is the compute root: each run gets a new
     workdir in ``DIR``, removed after it succeeds. Projects can share
     one.
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
exits with 2 when its command line is wrong, or when a met directory does
not exist or holds no ARL file; ``stilt submit`` and ``stilt status`` do
too. An error the command did not expect prints its traceback and exits
with 3 when it is a connection or I/O error, such as an output bucket that
cannot be reached, and with 4 for any other (:ref:`exit-codes`). Every
command does this, not only ``stilt run``.

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
