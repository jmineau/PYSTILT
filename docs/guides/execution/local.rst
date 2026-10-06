On Your Computer
================

Running locally is the default. With no ``execution`` section in
``config.yaml``, simulations run one after another on your machine. Use it
for notebooks, scripts, and anything that finishes in a reasonable time on
one computer.

Use several CPU cores
---------------------

To run several receptors at once, set ``cpus``:

.. code-block:: yaml

   execution:
     backend: local
     cpus: 4

Or set it for a single run from the command line:

.. code-block:: bash

   stilt run ./my_project --cpus 4

Each of the ``cpus`` processes runs one receptor at a time, with all of
that receptor's variants. Set ``cpus`` no higher than the number of CPU
cores you have. ``cpus`` means the same on Slurm, per array task.

From Python or a notebook
-------------------------

.. code-block:: python

   import stilt

   project = stilt.Project.init(
       "./my_project",
       receptors=receptors,
       mets={"hrrr": {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}},
       grid={"xmin": -114, "xmax": -111, "ymin": 39, "ymax": 42, "xres": 0.01, "yres": 0.01},
       variants={"hrrr": {}},
       execution={"backend": "local", "cpus": 4},
   )

   project.run()

``Project.init`` writes ``config.yaml`` and ``receptors.csv`` to the project
folder. Call it once. Later, open the project with
``stilt.Project("./my_project")``. ``project.run()`` does the same as
``stilt run``. It runs every unfinished simulation and returns when they
are done.

Then check on the results:

.. code-block:: python

   project.status()                  # one row per simulation, with a "state" column
   footprints = project.footprints()   # one dataset of every footprint

Python or the command line?
---------------------------

Use Python when you're exploring in a notebook, generating receptors in
code, or want to analyze results right after the run.

Use the command line when the project is already set up on disk, or when
running from a batch script.

Both read and write the same project folder, so you can mix them. For
example, run with ``stilt run`` and then analyze in a notebook with
``stilt.Project("./my_project")``.

Save now, run later
-------------------

``Project.init`` and ``project.add_receptors()`` save settings and
receptors to the project folder without running anything.
``add_receptors`` returns the IDs of the receptors it added:

.. code-block:: python

   project = stilt.Project("./my_project")
   receptor_ids = project.add_receptors(receptors)

Any machine that can see the folder can then run the project with
``stilt run``. From Python, :func:`stilt.execution.run_receptors` runs every
variant of the receptors you give it.
