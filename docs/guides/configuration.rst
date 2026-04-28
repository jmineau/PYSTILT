Configuration
=============

PYSTILT configuration is meant to describe one project: which meteorology files
to use, which footprints to create, and the common run controls for HYSPLIT.
The output file is ``config.yaml`` in the project root.

Use :doc:`../reference/configuration` when you need every field and default.
Use this guide when you are deciding what to write in ``config.yaml``.

Minimal project config
----------------------

Most projects start with one meteorology stream and one footprint:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 1h

   footprints:
     default:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

   n_hours: -24
   numpar: 500
   skip_existing: true


What each section means
-----------------------

``mets``
   Named meteorology streams. The name, such as ``hrrr``, becomes part of each
   simulation ID. Each stream points at ARL files and defines how timestamps map
   to filenames.

``footprints``
   Named footprint products. The simple form shown above is shorthand for a
   ``FootprintConfig`` with an inline grid. You only need a nested ``grid`` key
   when you want to be explicit.

``n_hours`` and ``numpar``
   The most common run controls. ``n_hours`` is negative for backward runs and
   positive for forward runs. ``numpar`` controls particle count.

``execution``
   Optional executor settings for Slurm or Kubernetes. Leave this out for local
   runs.

``skip_existing``
   Whether ``stilt run`` should avoid rerunning simulations whose required
   outputs are already complete.

Footprint grid shorthand
------------------------

The starter YAML uses the short form:

.. code-block:: yaml

   footprints:
     default:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

This is equivalent to:

.. code-block:: yaml

   footprints:
     default:
       grid:
         xmin: -114.0
         xmax: -111.0
         ymin: 39.0
         ymax: 42.0
         xres: 0.01
         yres: 0.01

Use the nested form if it reads better for your workflow or if you are
generating config files programmatically.

Multiple outputs
----------------

You can define more than one named footprint:

.. code-block:: yaml

   footprints:
     near_field:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

     regional:
       xmin: -125.0
       xmax: -100.0
       ymin: 30.0
       ymax: 50.0
       xres: 0.1
       yres: 0.1
       smooth_factor: 1.0

Each completed simulation will track each named footprint separately in the
output index.

Execution examples
------------------

Local execution does not need an ``execution`` section. For Slurm, add one:

.. code-block:: yaml

   execution:
     backend: slurm
     account: lin-group
     partition: lin
     time: "02:00:00"
     memory: 8G

Executor-specific fields are passed to the configured backend. Keep project
science controls, such as ``numpar`` and footprint grids, outside this section.

Python equivalent
-----------------

The same configuration can be built from Python:

.. code-block:: python

   import stilt

   config = stilt.ModelConfig(
       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/arl/hrrr",
               file_format="%Y%m%d_%H",
               file_tres="1h",
           )
       },
       footprints={
           "default": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-114.0,
                   xmax=-111.0,
                   ymin=39.0,
                   ymax=42.0,
                   xres=0.01,
                   yres=0.01,
               )
           )
       },
       n_hours=-24,
       numpar=500,
       skip_existing=True,
   )

Advanced parameters
-------------------

PYSTILT exposes lower-level HYSPLIT/STILT parameters for compatibility and
experimentation. Keep them out of starter configs unless you know why you are
changing them.

Unknown keys are rejected when YAML is loaded.
