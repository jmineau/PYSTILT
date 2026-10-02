Wind Error Statistics
=====================

A transport-error run (:doc:`transport_error`) needs four numbers that
describe the meteorology's wind errors. They are the standard deviation of
the error and how long, how high, and how far it stays correlated.
`Lin and Gerbig (2005) <https://doi.org/10.1029/2004GL021127>`_ derive
them by comparing the analysed wind with observed winds. This page does the
same for your meteorology and region. Derive your own values instead of
copying them from another analysis or place. The correlation scales decide
whether the perturbation has any effect at all.

What the numbers are
--------------------

For each wind component, the error at an observation is the analysed wind
minus the observed wind. Its standard deviation is ``siguverr``.

The correlation scales come from the variogram. At a separation ``h``, the
variogram is half the mean squared difference between the errors of pairs
of observations ``h`` apart. For an error with an exponential correlation
it is

.. math::

   \gamma(h) = \sigma^2 \left(1 - e^{-h/l}\right)

where σ is the standard deviation and ``l`` the correlation scale. Fitting
this curve with σ fixed gives ``l``. Pairs of heights give ``zcoruverr``,
pairs of times give ``tluverr``, and pairs of stations give
``horcoruverr``.

Each scale needs data that resolves it. Radiosonde profiles resolve height,
but launches are twelve hours apart, so they cannot measure a time scale of
a few hours. Hourly surface stations resolve time and horizontal distance,
but they only sample the bottom of the layer. The recipe below takes the
standard deviation and the vertical scale from radiosondes, and the time
and horizontal scales from surface stations.

Step 1: the errors
------------------

Sample the meteorology at the observations with
`arlmet <https://github.com/jmineau/arl-met>`_. Winds in ARL files on
projected grids are stored relative to the grid, so ask for earth-relative
components (``earth_relative=True``, arlmet 0.1.0a8 or later). Radiosondes
report geopotential height, so sample in metres above sea level:

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

Surface stations report wind at a height above ground, 10 m for a standard
anemometer. Sample the 10 m wind, or the lowest model level:

.. code-block:: python

   # stations: time, site, lon, lat, u, v
   points = stations.assign(z=10.0)
   met = arlmet.sample_points(hrrr_files, points, ["U10M", "V10M"],
                              z_kind="agl", earth_relative=True)
   surface = pd.DataFrame({
       "time": stations["time"], "site": stations["site"],
       "lon": stations["lon"], "lat": stations["lat"],
       "u_err": met["U10M"] - stations["u"],
       "v_err": met["V10M"] - stations["v"],
   })

Reading the observations is up to you. The Salt Lake Valley numbers below
used the sonde reader in `lair <https://github.com/jmineau/lair>`_.

For radiosondes anywhere in the world, the usual source is NOAA's
Integrated Global Radiosonde Archive (IGRA2).
`siphon <https://unidata.github.io/siphon/>`_ reads it
(``pip install siphon``). IGRA2 names a WMO station by its two-letter
country code, ``M``, and the WMO number padded to eight digits. Salt Lake
City, WMO 72572, is ``USM00072572``. The full list is
``igra2-station-list.txt`` in the NCEI archive. This builds the ``sondes``
table above:

.. code-block:: python

   import datetime as dt
   from siphon.simplewebservice.igra2 import IGRAUpperAir

   levels, launches = IGRAUpperAir.request_data(
       [dt.datetime(2024, 1, 1), dt.datetime(2024, 12, 31, 23)], "USM00072572"
   )
   launches = launches.drop_duplicates("date")[["date", "latitude", "longitude"]]
   levels = levels.merge(launches, on="date")
   ground = levels["lvltyp2"] == 1                  # IGRA2's surface level
   elevation = levels["height"].where(ground).groupby(levels["date"]).transform("max")
   sondes = pd.DataFrame({
       "time": levels["date"],
       "lon": levels["longitude"], "lat": levels["latitude"],
       "height": levels["height"],                  # geopotential height, m
       "elevation": elevation,
       "u": levels["u_wind"], "v": levels["v_wind"],
   }).dropna()

Each call downloads the station's whole record and keeps the dates you ask
for, so ask for the full period once, not one launch at a time. ``time`` is
the nominal launch hour, 00 or 12 UTC. The balloon goes up as much as an
hour earlier. The last line drops levels without a height, which IGRA2 has
on some significant levels. It also drops launches with no surface level,
since their heights above ground are unknown.

Step 2: the scales
------------------

:func:`~stilt.observations.variogram` builds the empirical variogram. It
takes the errors, the coordinate the separation is measured in, and a
``group`` label. Only points with the same label are paired. The coordinate
can also be two columns of longitude and latitude, and the separation is
then the great-circle distance in kilometres.
:func:`~stilt.observations.fit_variogram` fits the exponential model with
σ fixed.

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

The recipe makes these choices, and you can change any of them:

- ``layer`` is the height range whose errors matter for your transport.
  Use the boundary layer and a little above for a surface receptor, and
  more for a column.
- The vertical variogram pairs levels within one launch (``group`` is the
  launch time), in 100 m bins up to 3 km.
- The time variogram pairs hours at one station, out to ten days.
- The horizontal variogram pairs stations at one time, out to 50 km.
- Each fit fixes σ at the standard deviation of the errors it was built
  from.
- u and v are done separately and averaged.
- The mean error (the bias) is ignored, as in Lin and Gerbig. The
  perturbation does not represent it.

The four values go into ``config.yaml`` under the same names, or into
:class:`~stilt.config.TransportParams`. Look at the fits before you trust
them:

.. code-block:: python

   import matplotlib.pyplot as plt

   table = variogram(layer["u_err"], layer["height"], group=layer["time"],
                     bins=range(0, 3100, 100))
   fit = fit_variogram(table["lag"], table["gamma"], sigma=layer["u_err"].std())
   plt.scatter(table["lag"], table["gamma"], s=8)
   plt.plot(table["lag"], fit(table["lag"]))

Seasonal and diurnal values are worth a look. Filter both tables and run
the recipe again. For the Salt Lake Valley the horizontal scale ranged
from 9 km in summer to 20 km in winter.

Without surface stations, the time scale can only come from pairs of
launches at the same height. That tells you whether the errors are still
correlated twelve hours later, and little else. ``horcoruverr`` then has to
come from another source.

Salt Lake Valley, HRRR, 2024
----------------------------

Radiosondes from Salt Lake City and 20 surface stations in the valley,
compared with the 3 km HRRR analysis, gave:

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

These became the values in the :doc:`transport_error` example:
``siguverr: 2.6``, ``zcoruverr: 450``, ``horcoruverr: 14`` and
``tluverr: 260``. The hourly stations gave time scales of 207 min for u
and 221 min for v, and the sondes, extrapolated, 212 to 312 min. The recipe
above gives 2.5 m/s, 480 m, 214 min and 14 km. For comparison, Lin and
Gerbig found about 120 km, 4 hours and 900 m for an 80 km analysis over the
eastern United States.
