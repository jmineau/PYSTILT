Tutorial: Stationary Station (WBB)
==================================

This tutorial reproduces the familiar WBB-style stationary tower workflow in
current PYSTILT terms: multiple hourly receptors, one met stream, one named
footprint, and model-level output loading.

What you'll learn
-----------------

- how to build a multi-receptor project
- how to run trajectories and footprints together
- how to load and summarize footprint outputs

Receptors
---------

.. code-block:: python

   import pandas as pd
   import stilt

   times = pd.date_range("2015-07-05 00:00", "2015-07-11 23:00", freq="1h")
   receptors = [
       stilt.PointReceptor(
           time=t,
           latitude=40.7665,
           longitude=-111.8472,
           altitude=21.0,
       )
       for t in times
   ]

You can also load a receptor CSV with :func:`stilt.read_receptors`.

Project configuration
---------------------

.. code-block:: python

   config = stilt.ModelConfig(
       n_hours=-24,
       numpar=100,
       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/met/hrrr",
               file_format="%Y%m%d_%H",
               file_tres="1h",
           ),
       },
       footprints={
           "wbb": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-114.0,
                   xmax=-109.0,
                   ymin=39.0,
                   ymax=42.5,
                   xres=0.01,
                   yres=0.01,
               ),
           ),
       },
       execution={"backend": "local", "n_workers": 4},
   )

   model = stilt.Model(
       project="./wbb_project",
       receptors=receptors,
       config=config,
   )

Run
---

.. code-block:: python

   model.run()

.. code-block:: bash

   stilt status ./wbb_project

Summarize footprints
--------------------

.. code-block:: python

   import matplotlib.colors as mcolors
   import matplotlib.pyplot as plt
   import xarray as xr

   footprints = model.footprints["wbb"].load(mets="hrrr")

   stacked = xr.concat(
       [foot.integrate_over_time().data for foot in footprints],
       dim="receptor",
   ).mean("receptor")

   fig, ax = plt.subplots(figsize=(10, 6))
   stacked.plot(
       ax=ax,
       norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-2),
       cmap="YlOrRd",
       cbar_kwargs={"label": "surface influence"},
   )
   ax.scatter(-111.8472, 40.7665, marker="*", s=160, c="k", zorder=5)
   ax.set_title("Mean integrated footprint for WBB receptors")
   plt.tight_layout()

Next
----

- :doc:`../guides/outputs`
- :doc:`flux_inversion`
- :doc:`hpc_slurm`
