Tutorial: Footprint Aggregation And Flux Estimation
===================================================

This tutorial shows the next common step after generating footprints: combine
them with a flux description to estimate concentration enhancements.

What you'll learn
-----------------

- how to aggregate footprint sensitivity at discrete source locations
- how to convolve integrated footprints with a gridded inventory
- how to build a simple modeled enhancement time series

Starting point
--------------

Assume you already have a project with footprints, for example from
:doc:`wbb_stationary`.

Point-source aggregation
------------------------

Use :meth:`stilt.Footprint.aggregate` when you have discrete source locations:

.. code-block:: python

   import numpy as np
   import pandas as pd

   footprints = model.footprints["wbb"].load(mets="hrrr")

   sources = [
       (-111.970, 40.515, 45.0),
       (-112.015, 40.779, 120.0),
       (-111.890, 40.650, 30.0),
   ]
   coords = [(lon, lat) for lon, lat, _ in sources]
   fluxes = np.array([flux for _, _, flux in sources])

   rows = []
   for foot in footprints:
       start, end = foot.time_range
       bins = pd.interval_range(start=start, end=end, freq="1h")
       sensitivity = foot.aggregate(coords=coords, time_bins=bins)
       enhancement = (sensitivity.to_numpy() * fluxes[:, None]).sum(axis=0)
       rows.append(
           pd.Series(
               enhancement,
               index=[interval.mid for interval in bins],
               name=foot.receptor.id,
           )
       )

   modeled = pd.concat(rows, axis=1).T

Gridded inventory convolution
-----------------------------

For gridded inventories, integrate the footprint over time and multiply cell by
cell:

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

   observed = pd.read_csv("observations.csv", index_col="time", parse_dates=True)

   fig, ax = plt.subplots(figsize=(12, 4))
   observed["ch4_ppm"].plot(ax=ax, label="Observed", color="k", alpha=0.7)
   modeled["enhancement_ppm"].plot(ax=ax, label="Modeled enhancement", color="tab:red")
   ax.legend()
   ax.set_ylabel("CH4 (ppm)")
   plt.tight_layout()

This is still only the footprint-times-flux part of an inversion workflow, but
it is the central bridge from transport to emissions analysis.
