On Your Computer
================

Running locally is the default. With no ``execution`` section in
``config.yaml``, simulations run one after another on your machine. Use it
for notebooks, scripts, and anything that finishes in a reasonable time on
one computer.

Use several CPU cores
---------------------

To run several receptors at once, set ``n_workers``:

.. code-block:: yaml

   execution:
     backend: local
     n_workers: 4

Or set it for a single run from the command line:

.. code-block:: bash

   stilt run ./my_project --n-workers 4

Each worker is a separate process. It runs one receptor at a time, with all
of that receptor's variants. Set ``n_workers`` no higher than the number of
CPU cores you have.

From Python or a notebook
-------------------------

.. code-block:: python

   import stilt

   model = stilt.Model(
       project="./my_project",
       receptors=receptors,
       mets={"hrrr": {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}},
       grid={"xmin": -114, "xmax": -111, "ymin": 39, "ymax": 42, "xres": 0.01, "yres": 0.01},
       execution={"backend": "local", "n_workers": 4},
   )

   model.run()

``model.run()`` does the same as ``stilt run``. It saves ``config.yaml`` and
``receptors.csv`` to the project folder, runs every unfinished simulation,
and returns when they are done.

Then check on the results:

.. code-block:: python

   model.status()                          # one row per simulation, with a "complete" column
   footprints = model.simulations.footprint.load()   # {simulation id: Footprint}

Python or the command line?
---------------------------

Use Python when you're exploring in a notebook, generating receptors in
code, or want to analyze results right after the run.

Use the command line when the project is already set up on disk, or when
running from a batch script.

Both read and write the same project folder, so you can mix them. For
example, run with ``stilt run`` and then analyze in a notebook with
``stilt.Model(project=...)``.

Save now, run later
-------------------

``model.register()`` saves the settings and receptors to the project folder
without running anything. It returns the receptor IDs:

.. code-block:: python

   receptor_ids = model.register()

Any machine that can see the folder can then run the project with
``stilt run``. From Python, :func:`stilt.execution.run_receptors` runs every
variant of the receptors you give it.
