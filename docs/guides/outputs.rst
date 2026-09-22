Load And Plot Results
=====================

Every simulation produces two things:

- a **footprint** for each footprint name in your settings (NetCDF files),
- the **trajectories**: every particle's path (a Parquet file).

This page shows how to look at them, load them for analysis, and add
footprints up over areas you care about.

Quick look
----------

Open the project, pick a simulation, and plot:

.. code-block:: python

   import stilt

   model = stilt.Model(project="./my_project")
   sim = next(model.simulations.values())      # the first simulation

   sim.get_footprint("slv").plot.map()          # footprint, summed over time
   sim.trajectories.plot.map()                  # particle paths
   sim.plot.map("slv")                          # receptor, particles, and footprint together

To pick a particular simulation, index by its ID:

.. code-block:: python

   model.simulations.keys()                     # all simulation IDs
   sim = model.simulations["hrrr_202307151800_-111.848_40.766_10"]

Plotting needs the ``visualization`` extra. With cartopy installed, maps get
coastlines and state borders. Other plots:

- ``foot.plot.facet()``: one panel per hour
- ``receptor.plot.map()``: where the receptor is
- ``model.plot.availability()``: which receptor times and locations have
  results

Footprints
----------

.. code-block:: python

   foot = sim.get_footprint("slv")      # None if it doesn't exist yet
   foot.data                            # an xarray.DataArray
   foot.time_range                      # (start, end) of the footprint's hours
   foot.receptor                        # the receptor it belongs to

``foot.data`` has dimensions ``(time, lat, lon)``, one map per hour back from
the receptor, in units of ppm per (µmol m⁻² s⁻¹). (With
``time_integrate: true`` in the footprint settings there is a single map
instead.) To sum over time:

.. code-block:: python

   total = foot.integrate_over_time()

To open a footprint file directly, without a model:

.. code-block:: python

   foot = stilt.Footprint.from_netcdf("path/to/..._slv_foot.nc")

Each file also records the receptor and the settings used to make it.

Many simulations at once
------------------------

``model.footprints`` and ``model.trajectories`` work across the whole
project:

.. code-block:: python

   footprints = model.footprints["slv"].load()           # list of Footprint
   paths = model.footprints["slv"].paths()               # file paths only
   not_done = model.footprints["slv"].missing()          # simulation IDs

   trajectories = model.trajectories.load()

All of these accept filters:

.. code-block:: python

   model.footprints["slv"].load(
       mets="hrrr",
       time_range=("2023-07-01", "2023-07-31 23:00"),   # receptor times, inclusive
   )

Trajectories
------------

.. code-block:: python

   traj = sim.trajectories        # None if it doesn't exist yet
   df = traj.data                 # pandas DataFrame, one row per particle per time step

The columns you are most likely to use:

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

To open a trajectory file directly:

.. code-block:: python

   traj = stilt.Trajectories.from_parquet("path/to/..._traj.parquet")

Empty footprints
----------------

Sometimes a simulation runs fine but no particle ever reaches the footprint
grid, usually because the grid is too small or is not upwind. PYSTILT then
writes a small ``<simulation id>_<name>_foot.empty`` file instead of a NetCDF.
The simulation counts as finished (so reruns skip it), but ``load()`` and
``paths()`` leave it out because there is nothing to load. If you see many
of these, make your footprint grid bigger.

Adding footprints up over areas
-------------------------------

To get the influence of specific areas, such as counties, hexagons, or small
windows around point sources, use :meth:`~stilt.Footprint.aggregate`. It adds
up footprint cells into your areas and, optionally, into time bins:

.. code-block:: python

   import pandas as pd

   bins = pd.interval_range(
       start=foot.time_range[0], end=foot.time_range[1], freq="1h"
   )
   aggregated = foot.aggregate(
       target=[(-111.97, 40.515), (-112.015, 40.779)],
       time_bins=bins,
   )

Each footprint cell's value is added into the target area it overlaps
(split by area where a cell straddles two targets), so the total influence is
preserved. This is what you want when multiplying by emissions.

Footprints are always calculated on a regular latitude/longitude grid, which
keeps them identical to STILT-R's. The overlap between that grid and your
areas is worked out once and reused, so adding up thousands of footprints is
fast.

The target can be:

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
with identical results.  Pass ``backend="shapely"`` or
``backend="exactextract"`` to :func:`stilt.geometry.overlap_weights` to force
one.  To choose a
raster for a geometry up front, or to rebuild a stored footprint at higher
fidelity from its particles:

.. code-block:: python

   hexes = stilt.Mesh.from_h3(8, bounds=state)
   grid = stilt.Grid.from_geometry(hexes, cells_per_target=4)   # snapped, rounded
   traj = sim.trajectories
   foot = traj.footprint(stilt.FootprintConfig(grid=grid))
