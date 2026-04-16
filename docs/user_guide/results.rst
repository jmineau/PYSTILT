.. _results:

Accessing Results
=================

After execution, PYSTILT exposes trajectory and footprint outputs through
:class:`~stilt.Simulation` and :class:`~stilt.Model`.

This page focuses on read patterns that match alpha execution semantics,
including explicit empty-success footprint states.


Trajectories
------------

Each simulation stores a :class:`~stilt.Trajectories` object containing the
HYSPLIT particle ensemble as a :class:`pandas.DataFrame`:

.. code-block:: python

   sim = list(model.simulations.values())[0]

   trajs = sim.trajectories        # Trajectories | None
   if trajs is not None:
       df = trajs.data              # pandas DataFrame
       print(df.columns.tolist())

Key columns in the trajectory DataFrame:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Column
     - Description
   * - ``long``
     - Particle longitude (°E)
   * - ``lati``
     - Particle latitude (°N)
   * - ``zagl``
     - Particle height above ground level (m)
   * - ``time``
     - Minutes from receptor time (negative for backward runs)
   * - ``datetime``
     - Absolute UTC timestamp (derived from ``time``)
   * - ``foot``
     - Instantaneous surface-influence value (ppm / (µmol m⁻² s⁻¹) per particle)
   * - ``indx``
     - Particle index within the ensemble
   * - ``xhgt``
     - Release height reconstructed by PYSTILT for column / multipoint receptors (m AGL); absent for point receptors
   * - ``mlht``
     - Mixed-layer height at particle position (m)
   * - ``sigw``
     - Vertical turbulence standard deviation (m s⁻¹)
   * - ``tlgr``
     - Lagrangian time scale (s)
   * - ``pres``
     - Pressure at particle position (hPa)

Read from disk directly:

.. code-block:: python

   from stilt import Trajectories
   trajs = Trajectories.from_parquet(sim.paths.trajectories)


Footprints
----------

Footprints are :class:`~stilt.Footprint` objects wrapping an
:class:`xarray.DataArray` with dimensions ``(time, lat, lon)``:

.. code-block:: python

   foot = sim.get_footprint("default")  # Footprint | None

   if foot is not None:
       da = foot.data          # xarray.DataArray, dims=(time, lat, lon)
       print(da.attrs)         # CF-convention metadata
       print(foot.time_range)  # (start, end) Timestamps

Read from NetCDF directly:

.. code-block:: python

   from stilt import Footprint
   foot = Footprint.from_netcdf(sim.paths.get_footprint_file("default"))


Footprint terminal outcomes
---------------------------

Footprint state is tracked per footprint name in the repository:

- ``complete``: successful footprint file write.
- ``complete-empty``: successful terminal outcome with no footprint file.
- ``failed``: footprint generation failed.

When loading with :meth:`~stilt.Model.get_footprints`, simulations with
successful terminal states are considered eligible, and missing files are
skipped gracefully. This avoids false failures when a footprint is
``complete-empty``.


Batch Access via Model
-----------------------

Retrieve footprints for all simulations at once:

.. code-block:: python

   footprints = model.get_footprints(
       name="default",
       mets=["hrrr"],
       time_range=(
           pd.Timestamp("2023-07-15", tz="UTC"),
           pd.Timestamp("2023-07-16", tz="UTC"),
       ),
   )

:meth:`~stilt.Model.get_footprints` returns a list of :class:`~stilt.Footprint`
objects. Filter by location with ``location_ids=["my_tower"]``.

If some selected simulations are terminal ``complete-empty`` for the requested
footprint name, they are omitted from the returned object list because no
NetCDF artifact exists to load.


Time Integration
----------------

The ``foot.data`` xarray DataArray has a ``time`` dimension (one time step per
hour of the back-trajectory). Sum over the full record or a subset with
standard xarray:

.. code-block:: python

   # Sum all time steps
   foot_total = foot.data.sum("time")

   # Or use the convenience method
   foot_total = foot.integrate_over_time()          # all steps
   foot_total = foot.integrate_over_time(*foot.time_range)  # explicit range


Footprint Aggregation
---------------------

:meth:`~stilt.Footprint.aggregate` samples a footprint at source coordinates
and integrates over time bins, producing a concentration-influence time series.
See :doc:`../advanced/footprint_aggregation` for a full worked example.

.. code-block:: python

   import pandas as pd

   coords = [(-112.0, 40.5), (-111.9, 40.8)]  # (lon, lat) tuples
   time_bins = pd.interval_range(
       start=pd.Timestamp("2023-07-15 00:00", tz="UTC"),
       end=pd.Timestamp("2023-07-16 00:00", tz="UTC"),
       freq="1h",
   )
   result = foot.aggregate(coords=coords, time_bins=time_bins)
   # result: DataFrame, rows=coords, columns=time_bin left edges


Plotting
--------

Every PYSTILT object exposes a ``.plot`` accessor. All methods accept an
optional ``ax`` argument; if omitted and cartopy is installed, a GeoAxes is
created automatically.

**Trajectory map** — scatter particles colored by time, altitude, or footprint
influence:

.. code-block:: python

   trajs.plot.map(color_by="zagl")   # or "time" (default) or "foot"

**Footprint map** — log-scaled pcolormesh of the 2-D footprint. Use ``time=``
to select a single time step; omit to sum all steps:

.. code-block:: python

   foot.plot.map()                   # summed over all times
   foot.plot.map(time="2023-07-15 18:00", log=False)

**Footprint facet grid** — one subplot per time step with a shared colorbar:

.. code-block:: python

   fig, axes = foot.plot.facet(ncols=4, log=True)

**Composite simulation map** — footprint + trajectory scatter + receptor in one
call (any layer silently skipped if data is unavailable):

.. code-block:: python

   sim.plot.map(foot_name="default", show_traj=True)

**Model availability chart** — Gantt-style bar chart of simulation completion
across all locations and times:

.. code-block:: python

   model.plot.availability()

**Receptor location map** — useful for sanity-checking your domain:

.. code-block:: python

   receptor.plot.map(domain=config.footprints["default"].grid)
