.. _footprint_aggregation:

Footprint Aggregation
=====================

:meth:`~stilt.Footprint.aggregate` convolves a footprint with a set of surface
source coordinates to produce a source-receptor sensitivity time series.  The
result answers the question: *"how much does the emission from location X
influence the receptor at each hour?"*

This is the key step before multiplying footprints by a flux inventory in an
atmospheric inversion.

.. note::

   The method signature is ``aggregate(coords, time_bins)``; there is no
   ``freq`` shortcut. You must pass an explicit
   :class:`pandas.IntervalIndex` for ``time_bins``.


Method Signature
----------------

.. code-block:: python

   foot.aggregate(
       coords: list[tuple[float, float]],   # (lat, lon) pairs to sample
       time_bins: pd.IntervalIndex,          # time intervals to sum over
   ) -> pd.DataFrame

Returns a :class:`pandas.DataFrame` with shape
``(len(time_bins), len(coords))``, where each cell is the summed footprint
influence from that source coordinate over that time interval, in units of
ppm / (µmol m⁻² s⁻¹) × hours.


Worked Example
--------------

Suppose you have an inventory of emission locations (e.g., oil-and-gas wells
or landfills) and you want to know how much each contributes to your receptor
at hourly resolution over one day.

.. code-block:: python

   import pandas as pd
   import stilt

   # Footprint from a simulation
   sim = list(model.simulations.values())[0]
   foot = sim.get_footprint("default")

   # Source coordinates: (lat, lon)
   emission_sources = [
       (40.50, -112.10),   # landfill north-west of SLC
       (40.82, -111.92),   # refinery near SLC
       (40.65, -111.85),   # industrial zone
   ]

   # Hourly time bins covering the back-trajectory window
   start, end = foot.time_range
   time_bins = pd.interval_range(start=start, end=end, freq="1h")

   # Aggregate
   result = foot.aggregate(coords=emission_sources, time_bins=time_bins)

   print(result)
   # rows = hourly intervals, columns = source coordinates
   #                          (40.5,-112.1)  (40.82,-111.92)  (40.65,-111.85)
   # [2023-07-15 17:00, 18:00)   1.23e-03         3.45e-05        0.00e+00
   # [2023-07-15 16:00, 17:00)   ...


Convolution with Flux Inventories
----------------------------------

To go from footprint sensitivity to modelled concentration enhancement,
multiply by the flux at each source:

.. code-block:: python

   import numpy as np

   # Flux at each emission source [µmol m⁻² s⁻¹]
   flux = np.array([50.0, 120.0, 35.0])   # one value per coord

   # Multiply footprint sensitivity × flux → concentration enhancement [ppm]
   # result shape: (n_time_bins, n_sources)
   enhancement = result.values * flux[np.newaxis, :]

   # Total enhancement at receptor (summed over all sources)
   total_enhancement = enhancement.sum(axis=1)
   print(pd.Series(total_enhancement, index=result.index))


.. seealso::

   :doc:`../tutorials/flux_inversion`: end-to-end tutorial combining multiple
   footprints with a gridded methane inventory.
