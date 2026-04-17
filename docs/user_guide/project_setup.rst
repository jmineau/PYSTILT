.. _project_setup:

Project Setup
=============

A PYSTILT project is centred on a :class:`~stilt.Model`, the top-level object
that owns your receptors, configuration, and simulation results.


Configuration Classes
---------------------

All configuration is validated by Pydantic models before any simulations run.
The main entry point is :class:`~stilt.ModelConfig`, which inherits all STILT
transport parameters and adds project-level sections for meteorology, footprints,
and execution.

.. code-block:: python

   import stilt

   config = stilt.ModelConfig(
       # STILT run controls
       n_hours=-24,       # negative = backward trajectories
       numpar=100,        # number of particles per receptor
       hnf_plume=True,    # hyper-near-field plume correction

       # Meteorology sources (name → MetConfig)
       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/met/hrrr",
               file_format="hrrr_%Y%m%d.arl",
               file_tres="1h",
           ),
       },

       # Footprint output definitions (name → FootprintConfig)
       footprints={
           "default": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-113.0, xmax=-110.5,
                   ymin=40.0,   ymax=42.0,
                   xres=0.01,   yres=0.01,
               ),
               smooth_factor=1,
               time_integrate=False,  # keep time dimension
           ),
       },

       # Execution backend (optional — defaults to local)
       execution={"backend": "local", "n_workers": 4},
   )


Saving and Loading Config
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config.to_yaml("my_project/config.yaml")
   config = stilt.ModelConfig.from_yaml("my_project/config.yaml")


Grid and Bounds
---------------

:class:`~stilt.Grid` defines the spatial domain and resolution for a footprint.
:class:`~stilt.Bounds` is the base class for :class:`~stilt.Grid` and can be
used independently to define spatial subsets for meteorology subgridding.

.. code-block:: python

   grid = stilt.Grid(
       xmin=-113.0, xmax=-110.5,   # longitude bounds (°E)
       ymin=40.0,   ymax=42.0,     # latitude bounds (°N)
       xres=0.01,   yres=0.01,     # cell size (degrees)
   )

   print(grid.resolution)  # (0.01, 0.01)


Receptors
---------

See :doc:`../user_guide/running` for configuring receptors as part of a
:class:`~stilt.Model`, and :doc:`../advanced/column_receptors` for advanced
receptor types.

A simple list of point receptors:

.. code-block:: python

   import pandas as pd

   receptors = [
       stilt.Receptor(
           time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
           latitude=40.766,
           longitude=-111.848,
           altitude=10,
       ),
       stilt.Receptor(
           time=pd.Timestamp("2023-07-15 19:00", tz="UTC"),
           latitude=40.766,
           longitude=-111.848,
           altitude=10,
       ),
   ]

Or load from a CSV file:

.. code-block:: python

   receptors = stilt.read_receptors("receptors.csv")

The CSV must have at minimum columns ``time``, ``lat`` (or ``latitude``),
``lon`` (or ``longitude``), and ``zagl`` (or ``height``). An optional ``r_idx``
column enables multi-point (column) receptors; see
:doc:`../advanced/column_receptors`.


Creating a Model
----------------

.. code-block:: python

   model = stilt.Model(
       project="/path/to/my_project",   # project directory (created if absent)
       receptors=receptors,             # list[Receptor] or path to CSV
       config=config,                   # ModelConfig
   )

The project directory will be created automatically and will contain:

.. code-block:: text

   my_project/
     config.yaml          # serialised ModelConfig
     simulations.db       # SQLite simulation registry
     simulations/
       <sim_id>/          # one directory per simulation
         trajectories.parquet
         footprints/
           default.nc


STILT Transport Parameters
--------------------------

:class:`~stilt.ModelConfig` inherits :class:`~stilt.STILTParams` which
contains all HYSPLIT transport and turbulence settings. The most commonly
tuned parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``n_hours``
     - ``-24``
     - Duration of back-trajectories in hours. Negative = backward.
   * - ``numpar``
     - ``100``
     - Number of particles released per receptor.
   * - ``hnf_plume``
     - ``True``
     - Enable the hyper-near-field Gaussian plume dilution correction.
   * - ``rm_dat``
     - ``True``
     - Delete HYSPLIT's intermediate ``*.dat`` files after each run.
   * - ``varsiwant``
     - ``[...]``
     - List of trajectory output variables requested from HYSPLIT.

For the full parameter list see :class:`~stilt.STILTParams` in the API
reference.
