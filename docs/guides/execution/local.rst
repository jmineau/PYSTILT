On Your Computer
================

Running locally is the default. With no ``execution`` section in
``config.yaml``, simulations run one after another in the current process.
This is the right choice for notebooks, scripts, and anything that finishes
in a reasonable time on one machine.

Use several CPU cores
---------------------

To run several simulations at once, set ``n_workers``:

.. code-block:: yaml

   execution:
     backend: local
     n_workers: 4

or for a single run from the command line:

.. code-block:: bash

   stilt run ./my_project --n-workers 4

Each worker is a separate process running one simulation at a time, so
``n_workers`` should be at most the number of CPU cores you have.

From Python or a notebook
-------------------------

.. code-block:: python

   import stilt

   model = stilt.Model(
       project="./my_project",
       receptors=receptors,
       mets={"hrrr": {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}},
       footprints={"slv": {"xmin": -114, "xmax": -111, "ymin": 39, "ymax": 42, "xres": 0.01, "yres": 0.01}},
       execution={"backend": "local", "n_workers": 4},
   )

   model.run()

``model.run()`` does the same as ``stilt run``: it saves ``config.yaml`` and
``receptors.csv`` to the project folder, runs every unfinished simulation,
and returns when they are done.

Then check on the results:

.. code-block:: python

   model.status()                          # finished vs remaining
   model.simulations.incomplete()          # IDs still to run
   footprints = model.footprints["slv"].load()

Python or the command line?
---------------------------

Use **Python** when you're exploring in a notebook, generating receptors in
code, or want to analyze results right after the run.

Use the **command line** when the project is already set up on disk, or when
running from a batch script.

Both read and write the same project folder, so you can mix them: set up and
run with ``stilt run``, then analyze in a notebook with
``stilt.Model(project=...)``.

Advanced: save now, run later
-----------------------------

``model.register()`` saves the settings and receptors to the project folder
without running anything, and returns the simulation IDs:

.. code-block:: python

   sim_ids = model.register()

Any machine that can see the folder can then run the project, with
``stilt run``, or directly with
:func:`stilt.execution.run_simulations`.
