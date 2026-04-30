Quickstart
==========

The fastest path is:

1. create a project
2. point it at meteorology
3. add one or more receptors
4. run locally

CLI-first workflow
------------------

Initialize a starter project:

.. code-block:: bash

   stilt init ./my_project

This creates:

- ``config.yaml``
- ``receptors.csv``

Edit ``config.yaml`` so that ``mets.<name>.directory`` points at real ARL
meteorology. Then add receptors to ``receptors.csv``:

.. code-block:: text

   time,longitude,latitude,altitude
   2023-07-15 18:00:00,-111.848,40.766,10

Run the project locally:

.. code-block:: bash

   stilt run ./my_project

Python-first workflow
---------------------

The same first run can be driven entirely from Python:

.. code-block:: python

   import pandas as pd
   import stilt

   receptor = stilt.PointReceptor(
       time=pd.Timestamp("2023-07-15 18:00:00"),
       longitude=-111.848,
       latitude=40.766,
       altitude=10.0,
   )

   config = stilt.ModelConfig(
       n_hours=-24,
       numpar=200,
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
                   xmin=-113.0,
                   xmax=-110.5,
                   ymin=40.0,
                   ymax=42.0,
                   xres=0.01,
                   yres=0.01,
               )
           )
       },
   )

   model = stilt.Model(
       project="./my_project",
       receptors=[receptor],
       config=config,
   )

   handle = model.run()
   handle.wait()

   [simulation] = model.simulations.values()
   trajectories = simulation.trajectories
   footprint = simulation.get_footprint("default")

What gets written
-----------------

For each simulation ID, PYSTILT writes into the output project layout:

.. code-block:: text

   my_project/
     config.yaml
     receptors.csv
     simulations/
       index.sqlite
       by-id/
         hrrr_202307151800_-111.848_40.766_10/
           stilt.log
           hrrr_202307151800_-111.848_40.766_10_traj.parquet
           hrrr_202307151800_-111.848_40.766_10_default_foot.nc

What to do next
---------------

- See :doc:`../guides/project_layout` for the output project model.
- See :doc:`../guides/configuration` for editing ``config.yaml``.
- See :doc:`../guides/meteorology` for configuring ARL meteorology cleanly.
- See :doc:`../guides/outputs` for trajectory and footprint outputs.
- See :doc:`../guides/execution/index` for local, Slurm, and Kubernetes execution.
