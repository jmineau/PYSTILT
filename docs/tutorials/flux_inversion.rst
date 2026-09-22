Tutorial: From Footprints To Concentrations
===========================================

A footprint says how much each grid cell influences a measurement. Multiply it
by how much each cell emits, add it all up, and you get the concentration
increase the measurement should show (its *enhancement* above background).
This is the link between transport and emissions that inversions are built
on.

The units make this work. Footprints are in ppm per (µmol m⁻² s⁻¹). Multiply
by fluxes in µmol m⁻² s⁻¹ and sum over cells and hours, and the result is in
ppm.

What you'll learn
-----------------

- how to model enhancements from a few point sources
- how to model enhancements from a gridded emissions inventory
- how to compare modeled and observed concentrations

Starting point
--------------

This tutorial uses the project and ``model`` from :doc:`wbb_stationary`.
Any project with footprints works.

A few point sources
-------------------

For a handful of known sources, add up each footprint over a small window
around each source with :meth:`stilt.Footprint.aggregate`, then multiply by
the source's flux:

.. code-block:: python

   import numpy as np
   import pandas as pd

   import stilt

   footprints = model.footprints["wbb"].load(mets="hrrr")

   # longitude, latitude, flux (µmol m⁻² s⁻¹, averaged over the window)
   sources = {
       "landfill": (-111.970, 40.515, 45.0),
       "wwtp": (-112.015, 40.779, 120.0),
       "refinery": (-111.890, 40.650, 30.0),
   }
   targets = stilt.Mesh.from_windows(
       [(lon, lat) for lon, lat, _ in sources.values()],
       size=0.01,  # window size in degrees around each source
       ids=list(sources),
   )
   fluxes = np.array([flux for _, _, flux in sources.values()])

   rows = []
   for foot in footprints:
       start, end = foot.time_range
       bins = pd.interval_range(start=start, end=end, freq="1h")
       sensitivity = foot.aggregate(target=targets, time_bins=bins)  # indexed by cell id
       enhancement = (sensitivity.to_numpy() * fluxes[:, None]).sum(axis=0)
       rows.append(
           pd.Series(
               enhancement,
               index=[interval.mid for interval in bins],
               name=foot.receptor.id,
           )
       )

   modeled = pd.concat(rows, axis=1).T

A gridded inventory
-------------------

For an emissions map, put the inventory on the footprint's grid, multiply cell
by cell, and sum. The inventory must be in µmol m⁻² s⁻¹. Summing the
footprint over time first assumes emissions are constant over the 24 hours:

.. code-block:: python

   import xarray as xr

   inventory = xr.open_dataarray("inventory.nc")

   enhancements = []
   for foot in footprints:
       integrated = foot.integrate_over_time().data
       inventory_on_grid = inventory.interp(
           lat=integrated.lat,
           lon=integrated.lon,
           method="linear",
       )
       enhancement = float((integrated * inventory_on_grid).sum(["lat", "lon"]))
       enhancements.append(
           {"time": foot.receptor.time, "enhancement_ppm": enhancement}
       )

   modeled = pd.DataFrame(enhancements).set_index("time").sort_index()

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

The modeled values are enhancements only. Before comparing, subtract a
background (the concentration of air arriving from outside the domain) from
the observations, or add one to the model.

This is the forward half of an inversion. An inversion goes the other way:
it adjusts the emissions until the modeled enhancements best match the
observations.
