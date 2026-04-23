Trajectory And Footprint Outputs
================================

PYSTILT writes two main science outputs:

- trajectory ensembles as Parquet
- footprints as NetCDF

Both are available through ``Simulation`` objects and through model-level
collections.

Trajectory outputs
------------------

Each successful simulation writes a self-contained trajectory parquet:

.. code-block:: python

   sim = next(model.simulations.values())
   print(sim.trajectories_path)

   trajectories = sim.trajectories
   if trajectories is not None:
       df = trajectories.data
       print(df.columns.tolist())

Important trajectory columns commonly used in analysis include:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Meaning
   * - ``long`` / ``lati``
     - Particle longitude and latitude
   * - ``zagl``
     - Particle height above ground level
   * - ``time``
     - Minutes from receptor time
   * - ``datetime``
     - Absolute timestamp derived by PYSTILT
   * - ``foot``
     - Instantaneous surface influence at the particle position
   * - ``indx``
     - Particle identifier within the ensemble
   * - ``xhgt``
     - Reconstructed release height for column or multipoint workflows when present
   * - ``mlht``, ``sigw``, ``tlgr``, ``pres``
     - Mixed-layer height, turbulence statistics, and pressure fields often
       used in diagnostics

You can also load a trajectory object directly from disk:

.. code-block:: python

   from stilt import Trajectories

   traj = Trajectories.from_parquet(sim.trajectories_path)

Footprint outputs
-----------------

Footprints are stored as NetCDF and exposed as :class:`stilt.Footprint`
wrappers around an ``xarray.DataArray``:

.. code-block:: python

   foot = sim.get_footprint("default")
   if foot is not None:
       print(foot.data.dims)
       print(foot.time_range)

The standard footprint data shape is ``(time, lat, lon)`` unless
``time_integrate=True`` was requested in the footprint config.

You can also load a footprint directly:

.. code-block:: python

   from stilt import Footprint

   foot = Footprint.from_netcdf(sim.footprint_path("default"))

Terminal footprint states
-------------------------

Named footprints are tracked durably with one of three terminal outcomes:

- ``complete``
- ``complete-empty``
- ``failed``

``complete-empty`` is important. It means the run succeeded, but no footprint
file is expected. Model-level footprint loaders skip those cases gracefully
instead of treating them as missing-data failures.

Cross-simulation access
-----------------------

The model collections are usually the cleanest way to work across many runs:

.. code-block:: python

   all_traj_paths = model.trajectories.paths()
   missing_traj = model.trajectories.missing()

   footprint_paths = model.footprints["default"].paths()
   footprints = model.footprints["default"].load()

Time integration and aggregation
--------------------------------

Footprints expose two especially useful analysis helpers:

.. code-block:: python

   total = foot.integrate_over_time()

   sampled = foot.aggregate(
       coords=[(-111.97, 40.515), (-112.015, 40.779)],
       time_bins=pd.interval_range(
           start=foot.time_range[0],
           end=foot.time_range[1],
           freq="1h",
       ),
   )

``integrate_over_time()`` collapses the time dimension.

``aggregate()`` samples the footprint at source coordinates and groups the
result by time bins, which is useful for point-source or inventory-style flux
applications.

Plotting shortcuts
------------------

Common quick-look methods are:

- ``trajectories.plot.map()``
- ``foot.plot.map()``
- ``foot.plot.facet()``
- ``sim.plot.map()``
- ``model.plot.availability()``
