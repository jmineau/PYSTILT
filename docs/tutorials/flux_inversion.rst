.. _tutorial_flux_inversion:

Tutorial: Footprint Aggregation and Flux Estimation
====================================================

This tutorial shows how to use PYSTILT footprints to estimate CH₄ emissions
in the Salt Lake Valley, a common workflow in top-down emission estimation.

**What you'll learn:**

* How to aggregate multiple footprints over a flux inventory.
* How to compute time series of modelled concentration enhancements.
* How to compare modelled and observed enhancements.

**Prerequisites:**

* Completed WBB footprints from :doc:`wbb_stationary` (or any PYSTILT project
  with footprint NetCDF files).
* A gridded CH₄ flux inventory (e.g., EPA gridded methane, EDGAR, or a
  custom inventory) as an :class:`xarray.DataArray` or CSV of point sources.


Overview
--------

The basic "footprint × flux" calculation is:

.. math::

   \Delta X_{CH_4}(t) = \sum_{i,j} F(x_i, y_j) \cdot H(t, x_i, y_j)

where :math:`F` is the surface flux [µmol m⁻² s⁻¹] and :math:`H` is the
footprint [ppm / (µmol m⁻² s⁻¹) · hour].


Option 1: Aggregate over Point Sources
----------------------------------------

Use :meth:`~stilt.Footprint.aggregate` when you have a discrete list of
emission locations (e.g., landfills, compressor stations, refineries).

.. code-block:: python

   import pandas as pd
   import stilt

   # Load footprints for all simulations
   footprints = model.get_footprints("wbb", mets=["hrrr"])

   # Known CH4 sources: (lat, lon, flux [µmol m-2 s-1])
   sources = [
       (40.515, -111.970, 45.0),   # Stericycle landfill
       (40.779, -112.015, 120.0),  # Chevron refinery
       (40.650, -111.890, 30.0),   # wastewater treatment plant
   ]
   coords = [(lat, lon) for lat, lon, flux in sources]
   fluxes  = [flux for lat, lon, flux in sources]

   results = []
   for foot in footprints:
       start, end = foot.time_range
       time_bins = pd.interval_range(start=start, end=end, freq="1h")
       sensitivity = foot.aggregate(coords=coords, time_bins=time_bins)

       # Multiply sensitivity [ppm/(µmol m-2 s-1)·h] × flux [µmol m-2 s-1]
       import numpy as np
       enhancement = (sensitivity.values * np.array(fluxes)).sum(axis=1)
       results.append(
           pd.Series(enhancement, index=[b.mid for b in time_bins], name=foot.receptor.id)
       )

   modelled = pd.concat(results).sort_index()
   modelled.index.name = "time"


Option 2: Convolve with a Gridded Inventory
---------------------------------------------

For gridded inventories (e.g., EDGAR, EPA gridded CH₄):

.. code-block:: python

   import xarray as xr

   # Load inventory [µmol m-2 s-1] on same grid as footprints
   inventory = xr.open_dataarray("edgar_ch4_2023.nc")

   enhancements = []
   for foot in footprints:
       # Time-integrate the footprint [ppm / (µmol m-2 s-1)]
       integrated = foot.integrate_over_time(*foot.time_range).data

       # Regrid inventory to footprint grid
       inv_regridded = inventory.interp(
           lat=integrated.lat, lon=integrated.lon, method="linear"
       )

       # Convolve: sum(footprint × inventory) over the domain
       enhancement = float((integrated * inv_regridded).sum(["lat", "lon"]))
       enhancements.append(
           {"time": foot.receptor.time, "enhancement_ppm": enhancement}
       )

   modelled = pd.DataFrame(enhancements).set_index("time").sort_index()


Comparing to Observations
--------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   obs = pd.read_csv(
       "tests/stilt-tutorials/01-wbb/obs_ch4.csv",
       index_col="time", parse_dates=True,
   )
   obs.index = obs.index.tz_localize("UTC")

   fig, ax = plt.subplots(figsize=(12, 4))
   obs["ch4_ppm"].plot(ax=ax, label="Observed", color="k", alpha=0.7)
   modelled["enhancement_ppm"].plot(ax=ax, label="Modelled enhancement",
                                    color="tab:red")
   ax.set_ylabel("CH₄ (ppm)")
   ax.set_title("WBB — Observed vs Modelled CH₄ Enhancement")
   ax.legend()
   plt.tight_layout()
   plt.show()


.. seealso::

   :doc:`../advanced/footprint_aggregation`: detailed API reference for
   :meth:`~stilt.Footprint.aggregate`.

   :doc:`../advanced/error_trajectories`: quantify transport uncertainty
   before performing an inversion.
