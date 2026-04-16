.. PYSTILT documentation master file

PYSTILT
=======

.. image:: https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://badge.fury.io/py/pystilt.svg
   :target: https://badge.fury.io/py/pystilt
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pystilt.svg
   :target: https://pypi.org/project/pystilt/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

----

A Python implementation of the `STILT <https://uataq.github.io/stilt/>`_
Lagrangian atmospheric transport model.

PYSTILT runs HYSPLIT trajectories, computes footprints, and tracks simulation
state through a single repository-backed worker architecture that supports
one-off runs, queue-based batch runs, and long-lived streaming workers.

This project is in alpha and intentionally prioritizes clear execution
semantics, testable behavior, and a clean forward API.

Install
-------

.. code-block:: bash

   pip install pystilt

Minimal end-to-end run
----------------------

.. code-block:: python

   import stilt, pandas as pd

   receptor = stilt.Receptor(
       time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
       latitude=40.766, longitude=-111.848, altitude=10,
   )
   config = stilt.ModelConfig(
       n_hours=-24, numpar=100,
       mets={"hrrr": stilt.MetConfig(
           directory="/data/met/hrrr",
           file_format="hrrr_%Y%m%d.arl", file_tres="1h",
       )},
       footprints={"default": stilt.FootprintConfig(
           grid=stilt.Grid(xmin=-113, xmax=-110.5, ymin=40, ymax=42,
                           xres=0.01, yres=0.01),
       )},
   )
   model = stilt.Model(project="./my_project", receptors=[receptor], config=config)
   handle = model.run()
   handle.wait()

   foot = list(model.simulations.values())[0].get_footprint("default")
   if foot is not None:
       foot.integrate_over_time(*foot.time_range).data.plot()

Alpha execution semantics
-------------------------

- Delivery guarantee: at-least-once processing. Simulations may be retried
  after interruption or worker loss.
- Trajectory state: ``pending -> running -> complete`` or ``failed``.
- Footprint state: per-footprint terminal outcomes are ``complete``,
  ``complete-empty``, or ``failed``.
- Empty footprints: treated as successful terminal outputs and represented
  explicitly as ``complete-empty``.
- Reruns: ``skip_existing=True`` avoids rerunning complete outputs;
  ``skip_existing=False`` resets to pending.


----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :fas:`rocket` New to STILT?
      :link: getting_started/index
      :link-type: doc

      Start here if you have never used STILT before. Learn the science,
      install the package, and run your first simulation.

      :doc:`What is STILT? <getting_started/what_is_stilt>` ·
      :doc:`Installation <getting_started/installation>` ·
      :doc:`Quickstart <getting_started/quickstart>`

   .. grid-item-card:: :fas:`map` Coming from R-STILT / X-STILT?
      :link: migration
      :link-type: doc

      Concept-mapping tables: ``stilt_run()`` → ``ModelConfig``,
      ``receptors.csv`` → ``read_receptors()``, column receptors, and more.

   .. grid-item-card:: :fas:`book` User Guide & Advanced Topics
      :link: user_guide/index
      :link-type: doc

      Full documentation of project setup, meteorology, execution backends,
      run semantics, and advanced workflows.

      :doc:`User Guide <user_guide/index>` ·
      :doc:`Advanced <advanced/index>`

   .. grid-item-card:: :fas:`server` Queue / Service Runtime
      :link: user_guide/service
      :link-type: doc

      Queue-backed batch runs, long-lived workers, and Kubernetes helper
      manifests for always-on execution.

      :doc:`Running <user_guide/running>` ·
      :doc:`Service Runtime <user_guide/service>`

   .. grid-item-card:: :fas:`satellite-dish` Observation Layer
      :link: user_guide/observations
      :link-type: doc

      Observation records, scene grouping, receptor-building sensor families,
      and declarative transforms for X-STILT-style workflows.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step worked examples with real data.

      :doc:`WBB stationary <tutorials/wbb_stationary>` ·
      :doc:`Column satellite <tutorials/column_satellite>` ·
      :doc:`HPC/Slurm <tutorials/hpc_slurm>` ·
      :doc:`Flux inversion <tutorials/flux_inversion>`

   .. grid-item-card:: :fas:`code` API Reference
      :link: api/index
      :link-type: doc

      Auto-generated reference for all public classes and functions.

   .. grid-item-card:: :fas:`code-branch` Contributing
      :link: contributing
      :link-type: doc

      Bug reports, pull requests, and improvements are all welcome.


.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   advanced/index
   tutorials/index
   api/index
   migration
   changelog
   contributing
