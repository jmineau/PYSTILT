Load And Plot Results
=====================

Each simulation is one receptor run under one variant
(:doc:`configuration`). It has up to two results in the output directory:

- the **trajectories**, every particle's path, as a Parquet file
- the **footprint**, when the variant has a grid, as a Parquet file of the
  cells the particles touched

Variants that differ only in footprint settings share one set of
trajectories; ``sim.trajectories`` returns the shared particles.

This page shows how to plot these outputs, load them for analysis, and add
footprints up over the areas you care about.

Quick look
----------

Open the project, pick a simulation, and plot it:

.. code-block:: python

   import stilt

   model = stilt.Model(project="./my_project")
   sim = next(iter(model.simulations))          # the first simulation

   sim.footprint.plot.map()                     # footprint, summed over time
   sim.trajectories.plot.map()                  # particle paths
   sim.plot.map()                               # receptor, particles, and footprint together

To pick a particular simulation, index by receptor id and variant:

.. code-block:: python

   model.simulations.keys()                     # all (receptor, variant) ids
   sim = model.simulations["202307151800_-111.848_40.766_10", "hrrr"]
   sim = model.simulations["202307151800_-111.848_40.766_10/hrrr"]   # same thing

Plotting needs the ``visualization`` extra. If cartopy is installed, maps
also show coastlines and state borders. There are a few other plots:

- ``foot.plot.facet()`` draws one panel per hour.
- ``receptor.plot.map()`` shows where the receptor is.
- ``model.plot.availability()`` shows which receptor times and locations
  have results.

Footprints
----------

.. code-block:: python

   foot = sim.footprint                 # None if there is no footprint file
   foot.data                            # an xarray.DataArray
   foot.time_range                      # (start, end) of the footprint's hours
   foot.receptor                        # the receptor it belongs to

``foot.data`` has dimensions ``(time, lat, lon)``, or ``(time, y, x)`` on a
projected grid. There is one map for each hour back from the receptor time,
in units of ppm per (µmol m⁻² s⁻¹). With ``time_integrate: true`` in the
footprint settings there is a single time step. To sum over time:

.. code-block:: python

   total = foot.integrate_over_time()

To write a footprint as a CF NetCDF file for other tools, and read one
back:

.. code-block:: python

   foot.to_netcdf("wbb_2023-07-15_18.nc")
   foot = stilt.Footprint.from_netcdf("wbb_2023-07-15_18.nc")

The file records the receptor and the settings used to make it.

Many simulations at once
------------------------

``model.simulations`` holds every receptor under every variant. Narrow it
with ``sel``, then load an output from the result:

.. code-block:: python

   sims = model.simulations.sel(
       variant="hrrr",
       time=slice("2023-07-01", "2023-07-31 23:00"),    # receptor times, both ends included
   )
   footprints = sims.footprint.load()                   # {simulation id: Footprint}
   paths = sims.footprint.paths()                       # {simulation id: Path}
   trajectories = sims.trajectories.load()

The results are dictionaries keyed by simulation id, so you always know
which receptor a result belongs to:

.. code-block:: python

   for sid, foot in footprints.items():
       print(sid.receptor, float(foot.integrate_over_time().sum()))

``sel`` takes these filters, each as one value or a list:

- ``receptor``, receptor ids
- ``variant``, variant names. The name of a realization group, such as
  ``hrrr-err``, selects all of its realizations.
- ``time``, one receptor time or a ``slice`` of times
- ``location``, location ids
- ``where``, a function that takes a receptor and returns ``True`` to keep
  it

Each call narrows the one before. A receptor id or variant name that the
project does not have raises ``KeyError``. The other filters may select
nothing.

Extra columns in ``receptors.csv`` are on each receptor as ``attrs``. Use
them with ``where`` to gather one satellite scene or one site:

.. code-block:: python

   scene = model.simulations.sel(variant="hrrr", where=lambda r: r.attrs["scene"] == "A")

To see what is left to do:

.. code-block:: python

   model.simulations.incomplete()   # a selection, like sel()
   model.simulations.status()       # a DataFrame, one row per simulation

``status()`` has a ``trajectory`` and a ``footprint`` column that say
whether each output exists. They are blank where the variant does not make
that output. The ``empty`` column marks footprints that are empty (`Empty
footprints`_), and the ``complete`` column says whether the simulation is
done.
``model.status()`` returns the same table. It checks every simulation, so
it is slow on a large project stored in the cloud. From the command line,
``stilt status`` prints the totals, per variant when there are several.

Trajectories
------------

.. code-block:: python

   traj = sim.trajectories        # None if there is no trajectory file
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
     - Particle height above ground, m
   * - ``time``
     - Minutes from the receptor time (negative for a backward run)
   * - ``datetime``
     - Time of the row, UTC
   * - ``foot``
     - The particle's influence from the surface at this step, in
       ppm per (µmol m⁻² s⁻¹)
   * - ``indx``
     - Particle number
   * - ``xhgt``
     - Release height, for column and multipoint receptors
   * - ``mlht``, ``sigw``, ``tlgr``, ``pres``
     - Mixed-layer height, vertical velocity spread, Lagrangian time scale,
       and pressure

To open a trajectory file without a model:

.. code-block:: python

   traj = stilt.Trajectories.from_parquet("path/to/..._traj.parquet")

Empty footprints
----------------

Sometimes a simulation runs fine but no particle ever reaches the footprint
grid. Usually the grid is too small or is not upwind. PYSTILT then writes a
footprint file with no cells and the reason inside. The simulation counts
as finished, so reruns skip it. ``sim.footprint`` is ``None``,
``sim.empty_reason`` says why, and ``load()`` leaves the simulation out
because there is nothing to load. The ``empty`` column of
``model.simulations.status()`` lists them. If you see many, make your
footprint grid bigger.

An empty footprint is not a footprint of zeros. It means the transport never
connected the receptor to your grid, so treating it as "the model says zero"
in a comparison or an inversion would be wrong. Drop those observations, or
find out why the particles never arrived.

Adding footprints up over areas
-------------------------------

Footprints are always calculated on the regular grid in your config, as in
STILT-R. To get the influence of other areas, such as counties, hexagons, or
small windows around point sources, use :meth:`~stilt.Footprint.aggregate`.
It adds up the footprint cells in each area, and the hours in each time
bin:

.. code-block:: python

   import pandas as pd
   import stilt

   bins = pd.interval_range(
       start=foot.time_range[0], end=foot.time_range[1], freq="1h"
   )
   state = stilt.Grid(xmin=-112.3, xmax=-111.6, ymin=40.4, ymax=41.0,
                      xres=0.02, yres=0.02)
   by_cell = foot.aggregate(state, time_bins=bins)        # index == state.index

The result is a DataFrame with one row per area and one column per time
bin, labelled by the start of the bin. A footprint cell that straddles two
areas is split between them by area, so the total influence is kept. This
is what you want before multiplying by emissions. Influence that falls
outside every area is dropped.

The target can be:

- a :class:`stilt.Grid`. Rows follow ``grid.index``, one ``(lon, lat)``
  pair per cell.
- a :class:`stilt.Mesh` of polygons with ids: a shapefile
  (``Mesh.from_file``), H3 hexagons (``Mesh.from_h3``), or windows around
  points (``Mesh.from_windows``). Rows are the polygon ids.
- a :class:`stilt.Zones`, which merges the cells of a grid or mesh into
  larger groups by label.

For cells given some other way, such as an xarray grid or a list of cell
centres, build the :class:`stilt.Grid` they lie on and select the rows you
need from the result.

.. code-block:: python

   sources = stilt.Mesh.from_windows(
       [(-111.97, 40.515), (-112.015, 40.779)], 0.01, ids=["landfill", "wwtp"]
   )
   by_source = foot.aggregate(sources, time_bins=bins)    # index == ["landfill", "wwtp"]

   counties = stilt.Mesh.from_file("counties.shp", ids="NAME")
   by_county = foot.aggregate(counties, time_bins=bins)

   sectors = stilt.Zones.from_labels(state, labels)       # one label per cell of state
   by_sector = foot.aggregate(sectors, time_bins=bins)

Polygons in another coordinate system are reprojected onto the footprint
grid. The overlaps between the footprint grid and your areas are worked out
once and reused, so adding up thousands of footprints is fast. Polygon
overlaps use shapely. If the optional
`exactextract <https://github.com/isciences/exactextract>`_ package is
installed (``pip install exactextract``), PYSTILT uses it instead. It gives
the same result and is about a hundred times faster on large grids.

The footprint grid must be fine enough to resolve your areas.
``aggregate`` warns when the smallest area spans fewer than two footprint
cells. In that case, calculate the footprint again from its particles on a
finer grid:

.. code-block:: python

   hexes = stilt.Mesh.from_h3(8, bounds=state)
   grid = stilt.Grid.from_geometry(hexes, cells_per_target=4)
   fine = sim.generate_footprint(sim.footprint_config.model_copy(update={"grid": grid}))
   by_hex = fine.aggregate(hexes, time_bins=bins)

``Grid.from_geometry`` picks a grid that covers the areas with at least four
cells across the smallest one. ``generate_footprint`` applies the
variant's particle transforms, as the stored footprint did, and does not
overwrite the stored file.
