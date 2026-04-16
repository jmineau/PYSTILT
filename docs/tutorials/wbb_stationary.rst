.. _tutorial_wbb:

Tutorial: Stationary Station (WBB)
====================================

This tutorial reproduces the classic `R-STILT WBB tutorial
<https://github.com/uataq/stilt-tutorials/tree/main/01-wbb>`_ in PYSTILT.
We compute 24-hour back-trajectories and footprints for the William Browning
Building (WBB) rooftop CO₂ / CH₄ monitor at the University of Utah.

**What you'll learn:**

* How to set up a multi-receptor PYSTILT project from a CSV file.
* How to run trajectories and footprints efficiently.
* How to visualise results with xarray and matplotlib.

**Prerequisites:**

* PYSTILT installed (``pip install pystilt``).
* HRRR ARL files for July 2015 covering Utah (available from
  `NOAA ARL <https://www.ready.noaa.gov/archives.php>`_).
* Tutorial data: ``tests/stilt-tutorials/01-wbb/`` (see
  :doc:`index` for the submodule command).


Receptor Setup
--------------

WBB is a fixed tower at 40.7665° N, 111.8472° W, with the inlet at 21 m AGL.
We simulate every hour over a one-week period:

.. code-block:: python

   import pandas as pd
   import stilt

   times = pd.date_range(
       "2015-07-05 00:00", "2015-07-11 23:00",
       freq="1h", tz="UTC",
   )
   receptors = [
       stilt.Receptor(
           time=t,
           latitude=40.7665,
           longitude=-111.8472,
           altitude=21,
       )
       for t in times
   ]

Or load from the tutorial CSV:

.. code-block:: python

   receptors = stilt.read_receptors(
       "tests/stilt-tutorials/01-wbb/receptors.csv"
   )


Project Configuration
---------------------

.. code-block:: python

   config = stilt.ModelConfig(
       n_hours=-24,
       numpar=100,
       hnf_plume=True,

       mets={
           "hrrr": stilt.MetConfig(
               directory="tests/stilt-tutorials/01-wbb/met",
               file_format="hrrr_%Y%m%d.arl",
               file_tres="1h",
           ),
       },

       footprints={
           "wbb": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-114.0, xmax=-109.0,
                   ymin=39.0,   ymax=42.5,
                   xres=0.01,   yres=0.01,
               ),
               smooth_factor=1,
           ),
       },

       execution={"backend": "local", "n_jobs": -1},
   )

   model = stilt.Model(
       project="./wbb_project",
       receptors=receptors,
       config=config,
   )


Running
-------

.. code-block:: python

   model.run(mets=["hrrr"], footprints=["wbb"])


Check progress:

.. code-block:: bash

   stilt status --project ./wbb_project


Visualising Footprints
-----------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.colors as mcolors

   footprints = model.get_footprints("wbb", mets=["hrrr"])

   # Stack all footprints and take the mean time-integrated influence
   import xarray as xr
   stacked = xr.concat(
       [f.integrate_over_time(*f.time_range).data for f in footprints],
       dim="receptor",
   ).mean("receptor")

   fig, ax = plt.subplots(figsize=(10, 6))
   stacked.plot(
       ax=ax,
       norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-2),
       cmap="YlOrRd",
       cbar_kwargs={"label": "ppm / (µmol m⁻² s⁻¹) · h"},
   )
   ax.set_title("Mean footprint — WBB, July 2015")
   ax.scatter(-111.8472, 40.7665, marker="*", s=200, c="k", zorder=5,
              label="WBB")
   ax.legend()
   plt.tight_layout()
   plt.show()


Next Steps
----------

* :doc:`../user_guide/results`: aggregate footprints with source inventories.
* :doc:`flux_inversion`: use these footprints to estimate CH₄ emissions.
* :doc:`hpc_slurm`: speed up the computation on a Slurm cluster.
