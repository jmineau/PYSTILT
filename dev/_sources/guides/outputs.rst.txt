Footprints And Particles
========================

Each simulation is one receptor run under one variant
(:doc:`projects`). It has up to two results in the output directory:

- the **particles**, where every particle went, as a Parquet file
- the **footprint**, when the variant has a grid, as a Parquet file of the
  cells the particles touched

Variants that differ only in footprint settings share one set of
particles.

This page shows how to plot them and load them for analysis.
:doc:`aggregation` sums footprints over areas, and many at once into a
Jacobian.

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
   * - ``lon`` / ``lat``
     - Particle longitude and latitude (STILT-R's ``long`` / ``lati``)
   * - ``zagl``
     - Particle height above ground, m
   * - ``time``
     - Minutes from the receptor time (negative for a backward run)
   * - ``datetime``
     - Time of the row, UTC
   * - ``foot``
     - The particle's influence from the surface at this step, in
       ppm per (µmol m⁻² s⁻¹)
   * - ``particle``
     - Particle number (STILT-R's ``indx``)
   * - ``xhgt``
     - Release height, for column and multipoint receptors
   * - ``mlht``, ``sigw``, ``tlgr``, ``pres``
     - Mixed-layer height, vertical velocity spread, Lagrangian time scale,
       and pressure

:doc:`../reference/particles` says what each column is, its units, and
what a transport model must write.

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
   meta = stilt.particles.particles_metadata(path)
   meta.receptor, meta.settings, meta.met_files, meta.realization

The file holds the receptor, the meteorology files it was made with, and
the run's settings, the same ones its folder's ``_settings.yaml`` holds:
the transport model's settings, the met's, the model build, and whether
it is an ensemble (``ensemble: true``, with the base seed). ``realization``
is which realization of an ensemble the file is, or ``None`` for a single
run.
``stilt.identity.transport_from_settings(settings)``
rebuilds the transport model's config from them.

Empty footprints
----------------

A footprint that no particle reached is a file with no cells: the
simulation is complete, ``sim.has_footprint`` is true, and
``sim.footprint`` is ``None``. ``project.footprints()`` lists it in
``attrs["empty"]``, and a Jacobian in ``H.empty``. It is not a footprint of
zeros; :doc:`checking` says what it means and what to do.

Many simulations at once
------------------------

``project.simulations`` is a table with one row per receptor under each
variant. Its columns are:

- ``receptor``, the receptor id
- ``variant``, the variant name
- ``realization``, ``0`` to ``N - 1`` for a variant declared with
  ``realizations: N``, and empty for one that runs once.
- ``time``, ``kind``, and ``location`` of the receptor
- one column for each extra column of ``receptors.csv``

Select rows the way you would in pandas, then hand the selection to the
project to load its results:

.. code-block:: python

   sims = project.simulations
   july = sims[
       (sims.variant == "hrrr")
       & sims.time.between("2023-07-01", "2023-07-31 23:00")   # both ends included
   ]
   footprints = project.footprints(july)   # one dataset: receptor, hour, lat, lon
   particles = project.particles(july)     # one table, with receptor and variant columns

Leave the selection out to load every simulation. Any table with
``receptor`` and ``variant`` columns works as a selection, such as your
observations merged with ``sims``, or a polars or pyarrow table, and so
does a boolean mask over ``project.simulations``.

The footprints of one variant come back as one :class:`xarray.Dataset`.
Receptors at different times line up on ``hour``, the hours after each
receptor's time, so a backward run's first hour is -1. The ``time``
coordinate says when each of a receptor's hours starts:

.. code-block:: python

   footprints.foot.sum("hour").mean("receptor").plot()   # the mean footprint
   footprints.foot.sel(receptor=rid)                     # one receptor's hours
   footprints.time.sel(receptor=rid)                     # and when they start

The values are read from the files only when a computation needs them, a
day's receptors at a time, so opening a month of footprints is quick.
Loading them all at once takes about 35 MB per footprint on a 300 by 300
grid over 24 hours. Select one variant first, as above; a selection of
several raises an error. Receptors whose footprint
is empty are listed in ``footprints.attrs["empty"]``, and those not run yet
in ``footprints.attrs["missing"]``.

For one receptor, ``project.simulation(rid, "hrrr").footprint`` is its
footprint as a :class:`xarray.DataArray` with absolute times, which the
``foot.stilt`` methods work on.

To sum a selection's footprints onto your flux cells, as one matrix, use
``project.jacobian`` (:doc:`aggregation`).

The particles come back as one table, so pandas can group them:

.. code-block:: python

   particles.groupby("receptor")["foot"].sum()

A simulation whose result does not exist yet is left out. Loading
particles takes 10 to 20 MB per simulation. For thousands of simulations,
read the ``particles/`` folder of the output directory with pyarrow,
DuckDB, or polars instead. To find a single file, use
``sim.footprint_path`` or ``sim.particles_path``.

To work with one simulation of a selection, look it up by its row:

.. code-block:: python

   for receptor, variant in july[["receptor", "variant"]].itertuples(index=False):
       sim = project.simulation(receptor, variant)

The extra columns of ``receptors.csv`` select one satellite scene or one
site:

.. code-block:: python

   scene = sims[(sims.variant == "hrrr") & (sims.scene == "A")]
   ensemble = sims[sims.variant == "hrrr-err"]        # every realization

To see what is left to do:

.. code-block:: python

   project.incomplete()       # the simulations that are not complete
   st = project.status()      # every simulation, with six more columns
   st.state.value_counts()
   st[st.state == "failed"]   # the failed simulations, and why
   project.status(july)       # the same, for a selection

``status()`` adds a ``particles`` and a ``footprint`` column that say
whether each output exists. They are blank where the variant does not make
that output. The ``state`` column is ``complete``, ``failed``,
``interrupted`` (the run started and was stopped before it finished, by a
time limit, preemption, or a killed process), or ``pending`` (not run
yet). A run that stops partway leaves a log saying when and where it
started. For a failed
simulation, ``step``, ``reason``, and ``message`` say why. ``status()``
reads folder listings and the failure records of failed simulations, and
opens no result file, so it is quick on a large project. From the command
line, ``stilt status`` prints the totals, per variant when there are
several.

Without PYSTILT
---------------

The output directory is a set of Parquet files in folders named
``settings=...`` and ``date=...``, which many tools read as columns. With
`DuckDB <https://duckdb.org>`_, from Python, R, or its own shell, a
question about thousands of simulations is one query:

.. code-block:: sql

   -- footprints per settings folder and day
   SELECT settings, date, count(DISTINCT receptor) AS footprints
   FROM read_parquet('output/footprints/*/*/*.parquet', hive_partitioning = true)
   GROUP BY settings, date
   ORDER BY settings, date;

   -- the total influence of each receptor's footprint
   SELECT receptor, sum(foot) AS total
   FROM read_parquet('output/footprints/settings=hrrr-93278c/*/*.parquet', hive_partitioning = true)
   GROUP BY receptor;

   -- where each particle of one day's receptors ended up
   SELECT receptor, particle, arg_min(lon, time) AS lon, arg_min(lat, time) AS lat
   FROM read_parquet('output/particles/settings=hrrr-a3f9c2/date=2023-07-15/*.parquet', hive_partitioning = true)
   GROUP BY receptor, particle;

A footprint file holds the cells the particles reached, as ``hour``,
``y``, and ``x`` (the cell's row and column in the grid that the folder's
``_settings.yaml`` gives) and ``foot``. An empty footprint is a file with
no rows, so a count of rows leaves it out. An ensemble's folders have a
``realization=k`` level as well, so its files are one folder deeper
(``*/*/*/*.parquet``) and ``realization`` is a column.
