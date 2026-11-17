Mobile Measurements
===================

An analyzer on a car, a train, an aircraft, or a ship measures every second
or so as it moves. Measurements a few seconds apart see nearly the same air,
so a simulation for each one would cost thousands of transport model runs
for little gain. :func:`~stilt.observations.receptors_from_track` groups the
measurements along the track and makes one :class:`~stilt.PointReceptor` for
each group. It also returns the track with each measurement's receptor id,
so after the run you can put the modeled value next to every measurement.

From a track to receptors
-------------------------

A track is a table with one row per measurement and the columns ``time``,
``longitude``, ``latitude``, and ``altitude``
(:data:`~stilt.observations.TRACK_SCHEMA`). Other columns, such as the
measured mole fractions, come along. Here is half an hour of a car driving
east and then north, measured every second:

.. ipython:: python

   import numpy as np
   import pandas as pd
   from stilt.observations import receptors_from_track

   rng = np.random.default_rng(42)
   seconds = np.arange(1800)
   track = pd.DataFrame({
       "time": pd.date_range("2024-07-01 18:00", periods=1800, freq="1s"),
       "longitude": -111.93 + 0.0001 * np.minimum(seconds, 900),
       "latitude": 40.76 + 0.00009 * np.maximum(seconds - 900, 0),
       "altitude": 3.0,                  # inlet height, meters above ground
       "ch4": 2.0 + 0.02 * rng.standard_normal(1800),   # ppm
   })

   receptors, table = receptors_from_track(track, every="5min")
   len(receptors)
   receptors[0]

Each receptor sits at the mean time and position of its group, so the six
five-minute groups give six receptors. ``table`` is the track with a
``receptor`` column:

.. ipython:: python

   table.head()

The receptors follow the track. Each color below is one group, and each
cross is its receptor:

.. ipython:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
   ax.scatter(table.longitude, table.latitude, s=2,
              c=pd.factorize(table.receptor)[0], cmap="tab10");
   ax.plot([r.longitude for r in receptors], [r.latitude for r in receptors],
           "kx", markersize=10);
   ax.set_xlabel("Longitude");
   @savefig tracks_groups.png width=5in
   ax.set_ylabel("Latitude");

Choosing the groups
-------------------

Group by time with ``every``, or by distance along the track with
``spacing``, in meters. We recommend five minutes, the default, for a
vehicle on the ground, because a car at city speeds covers a few
kilometers in that time, about one cell of a 3 km meteorology grid. Groups
start on the clock (18:00, 18:05, ...), whatever time the track starts.

An aircraft covers 30 km or more in five minutes, so give it ``spacing``
instead, about the size of a meteorology cell:

.. ipython:: python

   receptors_1km, _ = receptors_from_track(track, spacing=1000)
   len(receptors_1km)

``spacing`` gives the same number of receptors for each kilometer however
fast the platform moves. A vehicle that stops and keeps measuring stays in
one group until it moves on, so for a track with long stops, use ``every``
or drop the stopped rows first.

A pause in the track longer than ``gap``, 10 minutes by default, ends a
group, so no receptor stands for measurements from both sides of a break
in the data.

Altitudes
---------

The ``altitude`` of a vehicle or a ship is the inlet's height above the
ground or the water, the default ``altitude_ref="agl"``. An aircraft
usually records its GPS altitude above sea level, so pass
``altitude_ref="msl"``.

In a project
------------

Add the receptors, and keep the table with the project's inputs as
``tables/track.parquet``:

.. code-block:: python

   import stilt

   project = stilt.Project("./mobile")      # made with stilt init or Project.init
   project.add_receptors(receptors)
   project.add_table("track", table)
   project.run()

After the run, average the measurements in each group and set them beside
the modeled enhancement of its receptor
(``foot.stilt.enhancement``, :doc:`aggregation`):

.. code-block:: python

   import xarray as xr

   flux = xr.open_dataarray("ch4_flux.nc")   # µmol m⁻² s⁻¹
   track = pd.read_parquet(project.table_path("track"))

   compare = track.groupby("receptor")[["ch4"]].mean()
   for rid in compare.index:
       foot = project.simulation(rid, "hrrr").footprint
       if foot is not None:               # None when no particle reached the grid
           compare.loc[rid, "modeled"] = float(foot.stilt.enhancement(flux).sum())

``track.merge(compare["modeled"], on="receptor")`` instead gives each
measurement the modeled value of its receptor. When a simulation fails or a
footprint looks wrong, see :doc:`checking`.
