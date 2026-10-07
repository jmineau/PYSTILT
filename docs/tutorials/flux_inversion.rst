Tutorial: From Footprints To Concentrations
===========================================

A footprint says how much each grid cell influences a measurement. Multiply
it by how much each cell emits and add it all up. The result is how much the
emissions should raise the concentration at the receptor, called the
*enhancement* above background. Inversions are built on this link between
transport and emissions.

The units work out directly. Footprints are in ppm per (µmol m⁻² s⁻¹). Fluxes
in µmol m⁻² s⁻¹ times footprints, summed over cells and hours, give ppm.

What you'll learn
-----------------

- how to model enhancements from a few point sources
- how to model enhancements from a gridded emissions inventory
- how to compare modeled and observed concentrations

Starting point
--------------

This tutorial uses the ``project`` from :doc:`wbb_stationary`.
Any project with footprints works.

A few point sources
-------------------

For a handful of known sources, draw a small window around each one with
:meth:`stilt.Mesh.from_windows`. ``foot.stilt.aggregate`` adds up
the footprint inside each window. Multiply each sum by that source's flux and
add them up.

.. code-block:: python

   import pandas as pd

   import stilt

   project = stilt.Project("./wbb_project")
   sims = project.simulations
   hrrr = sims[sims.variant == "hrrr"]
   feet = [project.simulation(rid, "hrrr").footprint for rid in hrrr.receptor]
   feet = [foot for foot in feet if foot is not None]   # skip empty footprints

   # longitude, latitude, flux (µmol m⁻² s⁻¹, averaged over the window)
   sources = {
       "landfill": (-111.970, 40.515, 45.0),
       "wwtp": (-112.015, 40.779, 120.0),
       "refinery": (-111.890, 40.650, 30.0),
   }
   windows = stilt.Mesh.from_windows(
       [(lon, lat) for lon, lat, _ in sources.values()],
       size=0.05,  # window width in degrees
       ids=list(sources),
   )
   flux = pd.Series({name: f for name, (_, _, f) in sources.items()})

   rows = []
   for foot in feet:
       hours = foot.indexes["time"]
       whole_run = pd.IntervalIndex.from_tuples(
           [(hours.min(), hours.max() + pd.Timedelta("1h"))], closed="left"
       )
       in_window = foot.stilt.aggregate(windows, whole_run).iloc[:, 0]  # one value per source
       rows.append(
           {"time": foot.stilt.receptor.time, "enhancement_ppm": (in_window * flux).sum()}
       )

   modeled = pd.DataFrame(rows).set_index("time").sort_index()

Make each window at least two footprint cells wide. A smaller window gets a
warning, because the footprint is too coarse to say how much of a cell falls
inside it.

A gridded inventory
-------------------

For an emissions map, ``foot.stilt.enhancement`` does the
multiplication. The inventory is an :class:`xarray.DataArray` in
µmol m⁻² s⁻¹ with ``lat`` and ``lon`` dimensions. Each footprint cell takes
the flux of the inventory cell it falls in. If the inventory has a ``time``
dimension, each footprint hour uses the nearest inventory time.

.. code-block:: python

   import xarray as xr

   inventory = xr.open_dataarray("inventory.nc")

   rows = []
   for foot in feet:
       enhancement = float(foot.stilt.enhancement(inventory).sum())   # sum over hours
       rows.append({"time": foot.stilt.receptor.time, "enhancement_ppm": enhancement})

   modeled = pd.DataFrame(rows).set_index("time").sort_index()

Inventory and footprint grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each footprint cell takes the flux of the inventory cell nearest its
centre. That is right when the inventory's cells are as large as the
footprint's or larger. Choose the footprint grid with your inventories in
mind:

- Match the footprint grid to the finest inventory you will use. Every
  inventory is then as coarse as the footprint, or coarser.
- For a coarser inventory, or to solve for fluxes on coarse cells, sum the
  footprints onto its cells with ``project.jacobian`` or
  ``foot.stilt.aggregate`` (:doc:`../guides/aggregation`).
- For a finer inventory, put it on the footprint grid as an area-weighted
  mean, such as with `xESMF <https://xesmf.readthedocs.io>`_'s conservative
  regridding, or make the footprints on the inventory's grid with
  ``sim.calc_footprint(grid=...)``.

``foot.stilt.enhancement`` refuses an inventory with finer cells than the
footprint's, and warns when cells of one size are offset by more than a
tenth of a cell.

A flux is averaged onto larger cells, because it is per unit area. A
footprint is summed, because each value belongs to its whole cell. Never
put a footprint through a conservative (averaging) regridder: it would come
out too small by the ratio of the cell areas.

Comparing with observations
---------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # observed CH4 minus background, in ppm
   observed = pd.read_csv("observations.csv", index_col="time", parse_dates=True)

   fig, ax = plt.subplots(figsize=(12, 4))
   observed["ch4_enhancement_ppm"].plot(ax=ax, label="Observed", color="k", alpha=0.7)
   modeled["enhancement_ppm"].plot(ax=ax, label="Modeled enhancement", color="tab:red")
   ax.legend()
   ax.set_ylabel("CH4 enhancement (ppm)")
   plt.tight_layout()

The model gives only the enhancement. Before comparing, subtract a background
from the observations, or add one to the model. The background is the
concentration of the air arriving from outside the domain (see
:doc:`../guides/background`).

This is the forward half of an inversion. An inversion goes the other way. It
adjusts the emissions until the modeled enhancements best match the
observations.
