Tutorial: A Week Of Tower Footprints
====================================

This tutorial runs hourly footprints for one week at a rooftop measurement
site. Then it averages them into one map that shows which areas usually
influence the site.

The site is WBB, a long-running greenhouse gas site on the roof of the
William Browning Building at the University of Utah in Salt Lake City. The
same steps work for any fixed site. Just change the coordinates.

What you'll learn
-----------------

- how to make many receptors at once
- how to run them in parallel on your computer
- how to load many footprints and combine them

You'll need ARL meteorology for 4 to 11 July 2015. That is the week plus the
day before it, which the 24-hour back-trajectories reach into. See
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
   len(receptors)                 # 168, one per hour for 7 days

If your measurement times are in a spreadsheet, save them as a CSV and use
:func:`stilt.read_receptors` instead (see :doc:`../guides/receptors`).

Step 2: Make the project
------------------------

.. code-block:: python

   project = stilt.Project.init(
       "./wbb_project",
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
       grid={
           "xmin": -114.0, "xmax": -109.0,
           "ymin": 39.0,   "ymax": 42.5,
           "xres": 0.01,   "yres": 0.01,
       },
       variants={"hrrr": {}},
       execution={"backend": "local", "cpus": 4},   # 4 receptors at a time
   )

``numpar=100`` keeps this tutorial quick. Use more particles for research
results (see :doc:`../guides/projects`).

``Project.init`` writes ``config.yaml`` and ``receptors.csv`` to
``./wbb_project`` and stops if the folder already has a ``config.yaml``.
In a later session, open the project instead:

.. code-block:: python

   project = stilt.Project("./wbb_project")

Step 3: Run
-----------

.. code-block:: python

   project.run()

There are 168 simulations, so this takes a while with 4 workers. You can
stop it at any time with Ctrl-C. Running it again picks up where it left
off. To check progress from another terminal:

.. code-block:: bash

   stilt status ./wbb_project

Step 4: Average the footprints
------------------------------

Load all 168 footprints, sum each one over time, and average them:

.. code-block:: python

   import matplotlib.colors as mcolors
   import matplotlib.pyplot as plt

   footprints = project.footprints()   # one dataset: receptor, hour, lat, lon

   mean_foot = footprints.foot.sum("hour").mean("receptor")

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

Footprints are strongest right around the site and fall off quickly with
distance. They span several orders of magnitude, so the color scale is
logarithmic.

Next
----

- Turn these footprints into modeled concentrations in
  :doc:`flux_inversion`.
- See :doc:`../guides/outputs` for more ways to load and plot results.
- Run a whole year on a cluster: :doc:`../guides/slurm`.
