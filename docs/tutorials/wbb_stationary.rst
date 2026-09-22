Tutorial: A Week Of Tower Footprints
====================================

This tutorial runs hourly footprints for one week at a rooftop measurement
site, then averages them into a single map showing which areas usually
influence the site.

The site is WBB, a long-running greenhouse gas monitoring site on the roof of
the William Browning Building at the University of Utah in Salt Lake City.
The same steps work for any fixed site: change the coordinates.

What you'll learn
-----------------

- how to make many receptors at once
- how to run them in parallel on your computer
- how to load many footprints and combine them

You'll need ARL meteorology covering 4 to 11 July 2015 (the week, plus the day
before for the 24-hour back-trajectories). See
:doc:`../guides/meteorology`.

Step 1: One receptor per hour
-----------------------------

.. code-block:: python

   import pandas as pd
   import stilt

   times = pd.date_range("2015-07-05 00:00", "2015-07-11 23:00", freq="1h")
   receptors = [
       stilt.PointReceptor(
           time=t,
           latitude=40.7665,
           longitude=-111.8472,
           altitude=21.0,         # inlet height above ground, in metres
       )
       for t in times
   ]
   len(receptors)                 # 168: one per hour for 7 days

If your measurement times are in a spreadsheet, save them as a CSV and use
:func:`stilt.read_receptors` instead (see :doc:`../guides/receptors`).

Step 2: Set up the model
------------------------

.. code-block:: python

   model = stilt.Model(
       project="./wbb_project",
       receptors=receptors,
       n_hours=-24,
       numpar=100,
       mets={
           "hrrr": {
               "directory": "/data/met/hrrr",
               "file_format": "%Y%m%d_%H",
               "file_tres": "6h",
           }
       },
       footprints={
           "wbb": {
               "xmin": -114.0, "xmax": -109.0,
               "ymin": 39.0,   "ymax": 42.5,
               "xres": 0.01,   "yres": 0.01,
           }
       },
       execution={"backend": "local", "n_workers": 4},   # 4 simulations at a time
   )

``numpar=100`` keeps this tutorial quick. Use more particles for research
results (see :doc:`../guides/configuration`).

Step 3: Run
-----------

.. code-block:: python

   model.run()

With 4 workers this takes a while (168 simulations). You can stop it at any
time with Ctrl-C: running it again picks up where it left off. To check
progress from another terminal:

.. code-block:: bash

   stilt status ./wbb_project

Step 4: Average the footprints
------------------------------

Load all 168 footprints, sum each over time, and average them:

.. code-block:: python

   import matplotlib.colors as mcolors
   import matplotlib.pyplot as plt
   import xarray as xr

   footprints = model.footprints["wbb"].load()

   mean_foot = xr.concat(
       [foot.integrate_over_time().data for foot in footprints],
       dim="receptor",
   ).mean("receptor")

   fig, ax = plt.subplots(figsize=(10, 6))
   mean_foot.plot(
       ax=ax,
       norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-2),
       cmap="YlOrRd",
       cbar_kwargs={"label": "footprint (ppm per µmol m⁻² s⁻¹)"},
   )
   ax.scatter(-111.8472, 40.7665, marker="*", s=160, c="k", zorder=5)
   ax.set_title("Mean footprint at WBB, 5–11 July 2015")
   plt.tight_layout()

The color scale is logarithmic because footprints span several orders of
magnitude: they are strongest right around the site and fall off with
distance.

Next
----

- Turn these footprints into modeled concentrations:
  :doc:`flux_inversion`
- More ways to load and plot results: :doc:`../guides/outputs`
- Run a whole year on a cluster: :doc:`hpc_slurm`
