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

   aggregated = foot.aggregate(
       target=[(-111.97, 40.515), (-112.015, 40.779)],
       time_bins=pd.interval_range(
           start=foot.time_range[0],
           end=foot.time_range[1],
           freq="1h",
       ),
   )

``integrate_over_time()`` collapses the time dimension.

``aggregate()`` conservatively regrids the footprint onto a *spatial
geometry* and groups the result by time bins.  Because a footprint is an
extensive, per-cell sensitivity, native cells are **summed** (by area
overlap) into each target cell rather than sampled -- the right behavior for
inventory- or grid-style flux applications.

Footprints are always *computed* on a rectilinear raster (this keeps the
STILT kernel and R-STILT parity).  Any other state geometry is reached by a
sparse overlap-weight matrix that is built once per (raster, geometry) pair
and cached, so aggregating thousands of footprints is one matmul each.

The target is the state geometry of your inversion:

- :class:`stilt.Grid` -- every cell of a rectilinear grid; ``grid.index``
  gives the matching ``(lon, lat)`` state index.
- :class:`stilt.Mesh` -- arbitrary polygons with ids: a shapefile
  (``Mesh.from_file``), H3 hexagons (``Mesh.from_h3``), nested grids, or
  point-source windows (``Mesh.from_windows``).  Results are indexed by
  cell id.
- :class:`stilt.Zones` -- labels that merge the cells of a grid or
  mesh into super-cells.
- An xarray grid (``lon``/``lat`` or ``x``/``y`` coordinates; ``NaN`` cells
  in a 2-D DataArray are masked out) or a plain list of ``(x, y)`` cell
  centers on a regular lattice.

.. code-block:: python

   import stilt

   state = stilt.Grid(xmin=-112.3, xmax=-111.6, ymin=40.4, ymax=41.0,
                      xres=0.02, yres=0.02)
   by_cell = foot.aggregate(state, time_bins=bins)        # index == state.index

   sources = stilt.Mesh.from_windows(
       [(-111.97, 40.515), (-112.015, 40.779)], 0.01, ids=["landfill", "wwtp"]
   )
   by_source = foot.aggregate(sources, time_bins=bins)    # index == ["landfill", "wwtp"]

   counties = stilt.Mesh.from_file("counties.shp", ids="NAME")
   by_county = foot.aggregate(counties, time_bins=bins)   # reprojected as needed

   sectors = stilt.Zones.from_labels(state, labels)   # one label per grid cell
   by_sector = foot.aggregate(sectors, time_bins=bins)

Geometries in another CRS are reprojected onto the footprint's raster.  The
raster must be fine enough to resolve the target cells: ``aggregate`` warns
when the smallest target cell spans fewer than two native cells.

Polygon overlaps are computed with shapely by default.  If the optional
`exactextract <https://github.com/isciences/exactextract>`_ package is
installed (``pip install exactextract``; it is not a PYSTILT dependency) it is
used automatically and is roughly a hundred times faster on large rasters,
with identical fractions.  Pass ``backend="shapely"`` or
``backend="exactextract"`` to :func:`stilt.geometry.overlap_weights` to force
one.  To choose a
raster for a geometry up front, or to rebuild a stored footprint at higher
fidelity from its particles:

.. code-block:: python

   hexes = stilt.Mesh.from_h3(8, bounds=state)
   grid = stilt.Grid.from_geometry(hexes, cells_per_target=4)   # snapped, rounded
   traj = model.trajectories.load_one(path)
   foot = traj.footprint(stilt.FootprintConfig(grid=grid))

Plotting shortcuts
------------------

Common quick-look methods are:

- ``trajectories.plot.map()``
- ``foot.plot.map()``
- ``foot.plot.facet()``
- ``sim.plot.map()``
- ``model.plot.availability()``
