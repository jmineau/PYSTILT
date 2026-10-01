Load And Plot Results
=====================

Each simulation is one receptor run under one variant
(:doc:`configuration`). It has up to two results in the output directory:

- the **particles**, where every particle went, as a Parquet file
- the **footprint**, when the variant has a grid, as a Parquet file of the
  cells the particles touched

Variants that differ only in footprint settings share one set of
particles.

This page shows how to plot these outputs, load them for analysis, and add
footprints up over the areas you care about.

Quick look
----------

Open the project, pick a simulation, and plot it:

.. code-block:: python

   import stilt

   project = stilt.Project("./my_project")
   sim = project.simulation("202307151800_-111.848_40.766_10", "hrrr")

   sim.footprint.stilt.plot.map()   # footprint, summed over time
   sim.particles.stilt.plot.map()   # particle paths
   sim.plot.map()                   # receptor, particles, and footprint together

A simulation is named by its receptor id and its variant. To see them all,
look at ``project.simulations``, a DataFrame with one row per simulation:

.. code-block:: python

   project.simulations[["receptor", "variant"]]

Plotting needs the ``visualization`` extra. If cartopy is installed, maps
also show coastlines and state borders. There are a few other plots:

- ``foot.stilt.plot.facet()`` draws one panel per hour.
- ``receptor.plot.map()`` shows where the receptor is.
- ``project.plot.availability()`` shows the receptor times at each
  location.

Footprints
----------

A footprint is an :class:`xarray.DataArray`:

.. code-block:: python

   sim.has_footprint        # True once the footprint file exists
   foot = sim.footprint     # raises FileNotFoundError before that

It has dimensions ``(time, lat, lon)``, or ``(time, y, x)`` on a projected
grid. There is one map for each hour back from the receptor time, in units
of ppm per (µmol m⁻² s⁻¹). With ``time_integrate: true`` in the footprint
settings there is a single time step. Use xarray as you would on any other
array:

.. code-block:: python

   total = foot.sum("time")                                     # summed over time
   morning = foot.sel(time=slice("2023-07-15 06:00", "2023-07-15 11:00"))

Methods that need the receptor or the grid are under ``foot.stilt``:

.. code-block:: python

   foot.stilt.receptor      # the receptor
   foot.stilt.grid          # the grid it was made on
   foot.stilt.config        # all the footprint settings

The receptor id is also a coordinate, ``foot.receptor``. It stays on the
array through sums and arithmetic.

To write a footprint as a CF NetCDF file for other tools, and read one
back:

.. code-block:: python

   foot.stilt.to_netcdf("wbb_2023-07-15_18.nc")
   foot = stilt.read_footprint("wbb_2023-07-15_18.nc")

The file records the receptor and the settings used to make it.
``stilt.read_footprint`` also opens a footprint file from the output
directory, such as ``sim.footprint_path``, with nothing else around it.

Many simulations at once
------------------------

``project.simulations`` is a table with one row per receptor under each
variant. Its columns are:

- ``receptor``, the receptor id
- ``variant``, the variant name
- ``group``, the variant's name in ``config.yaml``. The realizations
  ``hrrr-err-0`` and ``hrrr-err-1`` share the group ``hrrr-err``.
- ``time``, ``kind``, and ``location`` of the receptor
- one column for each extra column of ``receptors.csv``

Select rows the way you would in pandas, then load the results of the
selection:

.. code-block:: python

   sims = project.simulations
   july = sims[
       (sims.variant == "hrrr")
       & sims.time.between("2023-07-01", "2023-07-31 23:00")   # both ends included
   ]
   footprints = july.load_footprints()   # {simulation id: DataArray}
   particles = july.load_particles()     # one table, with receptor and variant columns

The footprints come back in a dictionary keyed by simulation id, so you
always know which receptor a footprint belongs to:

.. code-block:: python

   for sid, foot in footprints.items():
       print(sid.receptor, float(foot.sum()))

The particles come back as one table, so pandas can group them:

.. code-block:: python

   particles.groupby("receptor")["foot"].sum()

``project.simulations.load_footprints()`` loads every simulation. A
simulation whose result does not exist yet is left out. Loading particles
takes 10 to 20 MB per simulation. For thousands of simulations, read the
``particles/`` folder of the output directory with pyarrow, DuckDB, or
polars instead. To find a single file, use ``sim.footprint_path`` or
``sim.particles_path``.

A selection also gives you each simulation in turn:

.. code-block:: python

   for sim in july:
       print(sim.id, sim.is_complete())

A selection only understands columns (``sims.variant`` or
``sims["site"]``) and picking rows with a condition. For anything else,
use its table, ``sims.frame``. To turn a table back into a selection, for
example after a merge with your own data, give it to
``stilt.Simulations`` with the project:

.. code-block:: python

   matched = sims.frame.merge(observations, on="receptor")
   stilt.Simulations(project, matched).load_footprints()

The extra columns of ``receptors.csv`` select one satellite scene or one
site:

.. code-block:: python

   scene = sims[(sims.variant == "hrrr") & (sims.scene == "A")]
   ensemble = sims[sims.group == "hrrr-err"]          # every realization

To see what is left to do:

.. code-block:: python

   sims.incomplete()   # the simulations that are not complete
   sims.status()       # every row, with four more columns
   july.status()       # the same, for a selection

``status()`` adds a ``particles`` and a ``footprint`` column that say
whether each output exists. They are blank where the variant does not make
that output. The ``empty`` column marks footprints that are empty (`Empty
footprints`_), and the ``complete`` column says whether the simulation is
done. Finding empty footprints opens each footprint file, so ``status()``
is slower than ``incomplete()`` on a large project. From the command line,
``stilt status`` prints the totals, per variant when there are several.

Particles
---------

.. code-block:: python

   sim.has_particles           # True once the particle file exists
   particles = sim.particles   # raises FileNotFoundError before that

``sim.particles`` is a pandas DataFrame with one row per particle per time
step.

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

PYSTILT's methods for the particle table are under ``particles.stilt``:

.. code-block:: python

   particles.stilt.endpoints()    # where each particle ends, one row each
   particles.stilt.plot.map()     # map of every particle position

To open a particle file without a project:

.. code-block:: python

   path = (
       "output/particles/settings=hrrr-a3f9c2/date=2023-07-15/"
       "202307151800_-111.848_40.766_10.parquet"
   )
   particles = stilt.read_particles(path)
   receptor, params, met_files = stilt.particles_metadata(path)

The file holds the receptor, the transport settings, and the meteorology
files it was made with.

Empty footprints
----------------

Sometimes a simulation runs fine but no particle ever reaches the footprint
grid. Usually the grid is too small or is not upwind. PYSTILT then writes a
footprint file with no cells and the reason inside. The simulation counts
as finished, so reruns skip it. ``sim.footprint`` is ``None``,
``sim.empty_reason`` says why, and ``load_footprints()`` leaves the
simulation out because there is nothing to load. The ``empty`` column of
``status()`` lists them. If you see many, make your
footprint grid bigger.

An empty footprint is not a footprint of zeros. It means the transport never
connected the receptor to your grid, so treating it as "the model says zero"
in a comparison or an inversion would be wrong. Drop those observations, or
find out why the particles never arrived.

Adding footprints up over areas
-------------------------------

Footprints are always calculated on the regular grid in your config, as in
STILT-R. To get the influence of other areas, such as counties, hexagons, or
small windows around point sources, use ``foot.stilt.aggregate``.
It adds up the footprint cells in each area, and the hours in each time
bin:

.. code-block:: python

   import pandas as pd
   import stilt

   hours = foot.indexes["time"]   # the start of each footprint hour
   bins = pd.interval_range(
       start=hours.min(), periods=len(hours), freq="1h", closed="left"
   )
   state = stilt.Grid(xmin=-112.3, xmax=-111.6, ymin=40.4, ymax=41.0,
                      xres=0.02, yres=0.02)
   by_cell = foot.stilt.aggregate(state, time_bins=bins)   # index == state.index

The result is a DataFrame with one row per area and one column per time
bin, labelled by the start of the bin. Footprint times are the start of
each hour, so the bins must be closed on the left (``closed="left"``). A footprint cell that straddles two
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
   by_source = foot.stilt.aggregate(sources, time_bins=bins)   # index == ["landfill", "wwtp"]

   counties = stilt.Mesh.from_file("counties.shp", ids="NAME")
   by_county = foot.stilt.aggregate(counties, time_bins=bins)

   sectors = stilt.Zones.from_labels(state, labels)       # one label per cell of state
   by_sector = foot.stilt.aggregate(sectors, time_bins=bins)

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
   by_hex = fine.stilt.aggregate(hexes, time_bins=bins)

``Grid.from_geometry`` picks a grid that covers the areas with at least four
cells across the smallest one. ``generate_footprint`` applies the
variant's particle transforms, as the stored footprint did, and does not
overwrite the stored file.
