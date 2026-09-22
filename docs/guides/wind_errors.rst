Wind Error Statistics
=====================

A transport-error run (:doc:`transport_error`) needs four numbers that
describe the meteorology's wind errors: their standard deviation and how far
they stay correlated in time, height and horizontal distance. Lin and
Gerbig (2005) derive them from the difference between the analysed wind and
observed winds, and this page does the same for your meteorology and region.
Do not copy values from another analysis or another place: the correlation
scales decide whether the perturbation does anything at all.

What the numbers are
--------------------

For each wind component, the *error* at an observation is the analysis
minus the observation. Its standard deviation is ``siguverr``. Its
correlation scales come from the variogram, half the mean squared
difference of the error between pairs of observations a separation ``h``
apart, which for an exponentially correlated error is

.. math::

   \gamma(h) = \sigma^2 \left(1 - e^{-h/l}\right)

so fitting the curve with the sill fixed at the known variance gives ``l``.
Done over pairs of heights it gives ``zcoruverr``, over pairs of times
``tluverr``, over pairs of stations ``horcoruverr``.

Each scale needs data that resolves it. Radiosonde profiles resolve height
but launch twelve hours apart, so they cannot say anything about a time
scale of a few hours; hourly surface stations resolve time and horizontal
distance but sit at the bottom of the layer. So the recipe below takes the
standard deviation and the vertical scale from the profiles, and the time
and horizontal scales from the stations.

Step 1: the errors
------------------

Sample the meteorology at the observations with
`arlmet <https://github.com/jmineau/arl-met>`_. Winds in ARL files on
projected grids are stored relative to the grid, so ask for earth-relative
components (arlmet 0.1.0a8 or later). Radiosondes report geopotential
height, so sample in metres above sea level:

.. code-block:: python

   import arlmet
   import pandas as pd

   sondes = pd.read_csv("slc_2024_sondes.csv")   # time, lon, lat, height (m MSL), elevation, u, v
   points = sondes.rename(columns={"height": "z"})
   met = arlmet.sample_points(hrrr_files, points, ["UWND", "VWND"],
                              z_kind="msl", earth_relative=True)
   upper = pd.DataFrame({
       "time": sondes["time"],
       "height": sondes["height"] - sondes["elevation"],   # m above ground
       "u_err": met["UWND"] - sondes["u"],
       "v_err": met["VWND"] - sondes["v"],
   })

Surface stations report a height above ground (10 m for a standard
anemometer); sample the 10 m wind or the lowest level:

.. code-block:: python

   points = stations.assign(z=10.0)
   met = arlmet.sample_points(hrrr_files, points, ["U10M", "V10M"],
                              z_kind="agl", earth_relative=True)
   surface = pd.DataFrame({
       "time": stations["time"], "site": stations["site"],
       "lon": stations["lon"], "lat": stations["lat"],
       "u_err": met["U10M"] - stations["u"],
       "v_err": met["V10M"] - stations["v"],
   })

Reading the observations is up to you. The sonde reader in
`lair <https://github.com/jmineau/lair>`_ was used for the Salt Lake
Valley numbers below.

Step 2: the scales
------------------

:func:`~stilt.observations.variogram` builds the empirical variogram from
an error array, the coordinate the separation is measured in, and a
``group`` label that says which points may be paired; the lag can also be
two columns of longitude and latitude, for great-circle distances in
kilometres. :func:`~stilt.observations.fit_variogram` fits the exponential
model with the sill fixed at the known variance. Everything else is the
choices below, which are yours to change:

.. code-block:: python

   import numpy as np
   from stilt.observations import fit_variogram, variogram

   def scale(errors, lag, group, bins):
       table = variogram(errors, lag, group=group, bins=bins)
       return fit_variogram(table["lag"], table["gamma"], sigma=errors.std()).length

   layer = upper[upper["height"].between(0, 3000)]        # m above ground
   minutes = (surface["time"] - pd.Timestamp("2000-01-01")) / pd.Timedelta("1min")
   components = ("u_err", "v_err")

   siguverr = np.mean([layer[c].std() for c in components])
   zcoruverr = np.mean([scale(layer[c], layer["height"], layer["time"],
                              range(0, 3100, 100)) for c in components])
   tluverr = np.mean([scale(surface[c], minutes, surface["site"],
                            range(0, 14401, 60)) for c in components])
   horcoruverr = np.mean([scale(surface[c], surface[["lon", "lat"]], surface["time"],
                                range(0, 51)) for c in components])

Line by line: the layer is the height range whose errors matter for your
transport, the boundary layer and a little above for a surface receptor
and more for a column. The vertical variogram pairs levels within one
launch (``group`` is the launch time) at 100 m separations. The time
variogram pairs hours within one station out to ten days. The horizontal
variogram pairs stations at one time out to 50 km. Each is fitted with the
standard deviation of the errors it was built from, and u and v are done
separately and averaged. The bias (mean error) is ignored, as in Lin and
Gerbig; the perturbation does not represent it.

The four values go into ``config.yaml`` under the same names, or straight
into :class:`~stilt.config.ErrorParams`. Look at the fits before trusting
them:

.. code-block:: python

   import matplotlib.pyplot as plt

   table = variogram(layer["u_err"], layer["height"], group=layer["time"],
                     bins=range(0, 3100, 100))
   fit = fit_variogram(table["lag"], table["gamma"], sigma=layer["u_err"].std())
   plt.scatter(table["lag"], table["gamma"], s=8)
   plt.plot(table["lag"], fit(table["lag"]))

Seasonal and diurnal values are worth a look: filter both tables and run
the recipe again. For the Salt Lake Valley the horizontal scale ranged
from 9 km in summer to 20 km in winter. Without surface stations the time
scale can only come from pairs of launches at the same height, which says
whether the errors are still correlated twelve hours later and little
else, and ``horcoruverr`` has to come from somewhere else.

Salt Lake Valley, HRRR, 2024
----------------------------

Radiosondes from Salt Lake City and 20 surface stations in the valley,
against the 3 km HRRR analysis:

.. list-table::
   :header-rows: 1

   * - Component
     - σ [m/s]
     - l_z [m]
     - l_x [km]
   * - u
     - 2.37
     - 365
     - 12.1
   * - v
     - 2.87
     - 548
     - 16.5

which became ``siguverr: 2.6``, ``zcoruverr: 450``, ``horcoruverr: 14`` and
``tluverr: 260``. The time scale from the hourly stations, 207 min for u
and 221 min for v, agrees with the sondes' extrapolated 212 to 312 min, and
the recipe above gives 2.5 m/s, 480 m, 214 min and 14 km. For comparison Lin and Gerbig found about 120 km, 4 hours
and 900 m for an 80 km analysis over the eastern United States, and
X-STILT's HRRR defaults of 5 km, 60 min and 100 m produce no detectable
perturbation at all (:doc:`transport_error`).
