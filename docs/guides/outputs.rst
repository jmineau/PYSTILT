Footprints And Particles
========================

Each simulation is one release run under one variant
(:doc:`projects`). It has up to two results in the output directory:

- the **particles**, where every particle went, as a Parquet file
- the **footprint**, when the variant has a grid, as a Parquet file of the
  cells the particles touched

Variants that differ only in footprint settings share one set of
particles.

Look at both before you trust a batch. We plot the particles of a few
simulations first, because they show where the air went whatever the
footprint grid, and only then load the footprints for analysis.
:doc:`aggregation` sums footprints over areas, and many at once into a
Jacobian.

Quick look
----------

.. This page runs when the docs are built, on a small sample project: four
   releases an hour apart on the morning of 15 January 2021, run on the met
   files the tests cache (docs/_ext/docs_examples.py).

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir(sample=True)

Open the project, pick a simulation, and plot it:

.. ipython:: python

   import stilt

   project = stilt.Project("./my_project")
   sim = project.simulation("202101150800_-111.848_40.766_10", "hrrr")

   @savefig outputs_footprint.png width=6in
   sim.footprint.stilt.plot.map();   # footprint, summed over time

   @savefig outputs_particles.png width=6in
   sim.particles.stilt.plot.map();   # particle paths

   @savefig outputs_simulation.png width=6in
   sim.plot.map();                   # release, particles, and footprint together

A simulation is named by its release id and its variant. To see them all,
look at ``project.simulations``, a DataFrame with one row per simulation:

.. ipython:: python

   project.simulations[["release", "variant"]]

Plotting needs the ``visualization`` extra. If cartopy is installed, maps
also show coastlines and state borders. There are a few other plots:

- ``foot.stilt.plot.facet()`` draws one panel per hour.
- ``release.plot.map()`` shows where the release is.
- ``project.plot.availability()`` shows the release times at each
  location.

.. ipython:: python

   @savefig outputs_facet.png width=6in
   sim.footprint.stilt.plot.facet();

   @savefig outputs_availability.png width=6in
   project.plot.availability();

   @savefig outputs_release.png width=6in
   sim.release.plot.map();

A map takes a ``tiler``, any cartopy tiler, to draw map tiles under the data.
The maps in these docs are drawn on shaded-relief tiles of the USGS National
Map, which are kept in the repository so that building the docs needs no
network for them. Your own maps can use the same tiles. cartopy downloads them
when the map is drawn, and ``cache=True`` keeps what it downloaded:

.. code-block:: python

   from cartopy.io.img_tiles import GoogleTiles

   usgs = GoogleTiles(
       url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}",
       cache=True,
   )
   sim.plot.map(tiler=usgs)

Footprints
----------

A footprint is an :class:`xarray.DataArray`:

.. ipython:: python

   sim.has_footprint        # True once the footprint file exists
   foot = sim.footprint     # raises FileNotFoundError before that

It has dimensions ``(time, lat, lon)``, or ``(time, y, x)`` on a projected
grid. There is one map for each hour back from the release time (each
``tres`` back, when the footprint settings give another length), in
units of ppm per (µmol m⁻² s⁻¹). ``foot.sum("time")`` is one map of the
whole run. Use xarray as you would on any other array:

.. ipython:: python

   total = foot.sum("time")                                     # summed over time
   last_hours = foot.sel(time=slice("2021-01-15 02:00", "2021-01-15 07:00"))

Methods that need the release or the grid are under ``foot.stilt``:

.. ipython:: python

   foot.stilt.release      # the release
   foot.stilt.grid          # the grid it was made on
   foot.stilt.config        # all the footprint settings

The release id is also a coordinate, ``foot.release``. It stays on the
array through sums and arithmetic.

``stilt.read_footprint`` opens a footprint file from the output directory,
such as ``sim.footprint_path``, with nothing else around it. To write one as
NetCDF for other tools, see `Results in other tools`_.

Particles
---------

.. ipython:: python

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
   * - ``age``
     - Minutes since release, negative for a backward run (STILT-R's
       ``time``)
   * - ``time``
     - Time of the row, UTC, built from ``age`` when the table is read
   * - ``foot``
     - The particle's influence from the surface at this step, in
       ppm per (µmol m⁻² s⁻¹)
   * - ``particle``
     - Particle number (STILT-R's ``indx``)
   * - ``release_height``
     - Release height, for column and multipoint releases (STILT-R's
       ``xhgt``)
   * - ``mlht``, ``sigw``, ``tlgr``, ``pres``
     - Mixed-layer height, vertical velocity spread, Lagrangian time scale,
       and pressure

:doc:`../reference/particles` says what each column is, its units, and
what a transport model must write.

PYSTILT's methods for the particle table are under ``particles.stilt``:

.. ipython:: python

   particles.stilt.endpoints()    # where each particle ends, one row each

   @savefig outputs_all_particles.png width=6in
   particles.stilt.plot.map();    # map of every particle position

To open a particle file without a project:

.. code-block:: python

   path = (
       "output/particles/settings=hrrr-a3f9c2/date=2021-01-15/"
       "202101150800_-111.848_40.766_10.parquet"
   )
   particles = stilt.read_particles(path)
   meta = runtraj.particles.particles_metadata(path)
   meta.release, meta.settings, meta.met_files, meta.realization

The file holds the release, the meteorology files the run read, and the
settings it ran with, the same ones as its folder's ``_settings.yaml``.
``realization`` is which realization of an ensemble the file is, or
``None`` for a single run. :doc:`../reference/layout` lists every key.

Empty footprints
----------------

A footprint that no particle reached is a file with no cells: the
simulation is complete, ``sim.has_footprint`` is true, and
``sim.footprint`` is ``None``. ``project.footprints()`` lists it in
``attrs["empty"]``, and a Jacobian in ``H.empty``. It is not a footprint of
zeros; :doc:`checking` says what it means and what to do.

Many simulations at once
------------------------

``project.simulations`` is a table with one row per release under each
variant. Its columns are:

- ``release``, the release id
- ``variant``, the variant name
- ``realization``, ``0`` to ``N - 1`` for a variant declared with
  ``realizations: N``, and empty for one that runs once.
- ``time``, ``kind``, and ``location`` of the release
- one column for each extra column of ``releases.csv``

Select rows the way you would in pandas, then hand the selection to the
project to load its results:

.. ipython:: python

   sims = project.simulations
   morning = sims[
       (sims.variant == "hrrr")
       & sims.time.between("2021-01-15 06:00", "2021-01-15 09:00")   # both ends included
   ]
   footprints = project.footprints(morning)   # one dataset: release, hour, lat, lon
   particles = project.particles(morning)     # one table, with release and variant columns

Leave the selection out to load every simulation. Any table with
``release`` and ``variant`` columns works as a selection, such as your
observations merged with ``sims``, or a polars or pyarrow table, and so
does a boolean mask over ``project.simulations``.

The footprints of one variant come back as one :class:`xarray.Dataset`
with dimensions ``(release, hour, lat, lon)``. Releases at different
times cannot share a time axis, so they line up on ``hour``: the hours
after each release's time, where a backward run's first hour is -1. With
other layers the dimension is named for the unit ``tres`` is written in
(``day`` for ``1D``, ``minute`` for ``15min``) and counts in it. The
``time`` coordinate gives the clock time each release's hours start:

.. ipython:: python

   import numpy as np

   rid = morning.release.iloc[0]                         # the first release

   mean = footprints.foot.sum("hour").mean("release")    # the mean footprint
   @savefig outputs_mean_footprint.png width=6in
   np.log10(mean.where(mean > 0)).plot();                 # most cells are 0: use a log scale
   footprints.foot.sel(release=rid)                      # one release's hours
   footprints.time.sel(release=rid)                      # and when they start

The values are read from the files only when a computation needs them, a
day's releases at a time, so opening a month of footprints is quick.
Loading them all at once takes about 35 MB per footprint on a 300 by 300
grid over 24 hours. Select one variant first, as above; a selection of
several raises an error. Releases whose footprint is empty are listed in
``footprints.attrs["empty"]``, and those not run yet in
``footprints.attrs["missing"]``.

For one release, ``project.simulation(rid, "hrrr").footprint`` is its
footprint as a :class:`xarray.DataArray` with absolute times, which the
``foot.stilt`` methods work on.

To sum a selection's footprints onto your flux cells, as one matrix, use
``project.jacobian`` (:doc:`aggregation`).

The particles come back as one table, so pandas can group them:

.. ipython:: python

   particles.groupby("release")["foot"].sum()

A simulation whose result does not exist yet is left out. Loading
particles takes 10 to 20 MB per simulation. For thousands of simulations,
read the ``particles/`` folder of the output directory with pyarrow,
DuckDB, or polars instead. To find a single file, use
``sim.footprint_path`` or ``sim.particles_path``.

To work with one simulation of a selection, look it up by its row:

.. ipython:: python

   for release, variant in morning[["release", "variant"]].itertuples(index=False):
       sim = project.simulation(release, variant)

The extra columns of ``releases.csv`` select one satellite scene or one
site:

.. code-block:: python

   scene = sims[(sims.variant == "hrrr") & (sims.scene == "A")]
   ensemble = sims[sims.variant == "hrrr-err"]        # every realization of every release

To see what is left to do:

.. ipython:: python

   project.incomplete()       # the simulations that are not complete
   st = project.status()      # every simulation, with what it has and where it stands
   st.state.value_counts()
   st[st.state == "failed"]   # the failed simulations, and why
   project.status(morning)    # the same, for a selection

None failed here, so the table of failed simulations is empty.

``status()`` adds a ``particles`` and a ``footprint`` column that say
whether each result exists, and a ``state`` column: ``complete``,
``failed``, ``interrupted``, or ``pending``. For a failed simulation,
``stage``, ``reason``, and ``message`` say why. :doc:`checking` explains
each state and reason. ``status()`` opens no result file, so it is quick
on a large project.

Results in other tools
----------------------

A footprint can be written as a CF NetCDF file, the particles as CF
trajectories, and a footprint in the PARIS format that openghg reads.

Footprints as NetCDF
~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

   foot.stilt.to_netcdf("footprint_2021-01-15_08.nc")
   foot = stilt.read_footprint("footprint_2021-01-15_08.nc")

The file follows the CF-1.8 conventions and records the release and the
settings used to make it, so ``stilt.read_footprint`` reads it back whole.

Particles in trajan and other CF tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sim.particles.stilt.to_cf_trajectories()`` returns the particles as an
:class:`xarray.Dataset` that follows the CF-1.10 conventions for
trajectories, which `trajan <https://opendrift.github.io/trajan/>`_,
cf-xarray, and other tools that read CF trajectory files open:

.. ipython:: python

   ds = sim.particles.stilt.to_cf_trajectories()
   ds

Write it with :meth:`xarray.Dataset.to_netcdf`. A particle table from
:func:`stilt.run_trajectories` converts the same way.

Each particle is one trajectory. The coordinates are:

``trajectory``
   The particle number, marked ``cf_role: trajectory_id``.
``time``
   The particle table's ``time``, in UTC.
``lon``, ``lat``
   Longitude and latitude in degrees.
``zagl``
   Height above ground in meters (``standard_name: height``).

``age``, the minutes since release, is a data variable, and so is every other
numeric column of the table (``foot``, ``mlht``, ...). The columns PYSTILT
and HYSPLIT write have units and long names; another column is kept without
units. A table with several realizations is refused: select one first,
``particles[particles.realization == k]``.

Every variable has dimensions ``(trajectory, obs)``, CF's incomplete
multidimensional array (CF-1.10, appendix H.4). ``obs`` counts output steps
from the release, so in a backward run ``time`` decreases along it. A
particle that leaves the meteorology domain stops early, and the rest of its
row is missing values.

trajan reads the file through its ``traj`` accessor:

.. code-block:: python

   import trajan  # registers ds.traj
   import xarray as xr

   ds.to_netcdf("trajectories.nc")
   ds = xr.open_dataset("trajectories.nc", decode_timedelta=False)
   ds.traj.plot()

``decode_timedelta=False`` keeps ``age`` in minutes rather than reading it
as a duration.

Footprints for openghg
~~~~~~~~~~~~~~~~~~~~~~

``sim.footprint.stilt.to_paris(sim.particles)`` returns the footprint in the
PARIS format, the NetCDF layout openghg reads with ``source_format="paris"``.
openghg can then combine it with its fluxes and boundary conditions as it
does footprints from NAME and FLEXPART. The footprint must have hourly layers
(``tres: 1h``, the default) on a longitude and latitude grid and be backward.

.. ipython:: python

   paris = sim.footprint.stilt.to_paris(sim.particles)
   paris

openghg stores a time-resolved footprint along ``H_back``, hours back from
the footprint's time, and pairs ``H_back = k`` with the flux hour that
*starts* ``k`` hours before it. The footprint's time is the release time.
The first layer of a backward run, layer -1, covers the hour before the
release time, so it goes to ``H_back = 1``. With the release at 18:00:

.. list-table::
   :header-rows: 1

   * - ``H_back``
     - Flux hour openghg pairs it with
     - PYSTILT layer
   * - 0
     - 18:00 to 19:00
     - none, after the observation; always zero
   * - 1
     - 17:00 to 18:00
     - -1
   * - 2
     - 16:00 to 17:00
     - -2
   * - 23
     - 19:00 to 20:00 the day before
     - -23

Layers -24 and earlier are summed into ``srr_residual``, which openghg
multiplies by a monthly mean flux, and ``srr`` is the sum of all layers.
openghg resolves hours within one day, so 23 layers are kept by hour, not
24. The values are PYSTILT's unchanged: ppm per (µmol m⁻² s⁻¹) is the same
quantity as the PARIS units, (mol/mol)/(mol/m2/s).

.. ipython:: python

   per_hour = paris.srr_time_resolved.sum(["latitude", "longitude"])
   per_hour.isel(time=0).values[:4]      # H_back 0 is empty; 1 is layer -1
   float(paris.srr.sum() - paris.srr_time_resolved.sum() - paris.srr_residual.sum())

One release of ``project.footprints().foot``,
``footprints.foot.sel(release=rid)``, converts as ``sim.footprint`` does.

Pass the particles, as above: openghg also expects ``particle_locations_n``,
``_e``, ``_s`` and ``_w``, the fraction of particles that first left the
footprint grid through each edge, by cell along the edge and by 1 km height
layer up to 20 km (:meth:`~runtraj.particles.ParticlesAccessor.grid_exits`).
Without the particles these are left out, and openghg refuses the file
unless told not to expect them. A particle that never left the grid counts
nowhere, so the four add up to the share that left:

.. ipython:: python

   float(sum(paris[f"particle_locations_{e}"].sum() for e in "nesw"))

A PARIS file holds one site, with one entry along ``time`` for each release
time. Convert each release, concatenate along ``time``, write the file, and
add it to openghg's object store:

.. code-block:: python

   import xarray as xr

   site = [project.simulation(r, v) for r, v in morning[["release", "variant"]].itertuples(index=False)]
   ds = xr.concat(
       [s.footprint.stilt.to_paris(s.particles) for s in site],
       dim="time",
       combine_attrs="drop_conflicts",
   )
   ds.to_netcdf("WBB-10magl_STILT_HRRR_SLV_co2_202401.nc")

   from openghg.standardise import standardise_footprint

   standardise_footprint(
       filepath="WBB-10magl_STILT_HRRR_SLV_co2_202401.nc",
       source_format="paris",
       site="wbb",
       domain="slv",
       model="STILT",
       inlet="10m",
       met_model="hrrr",
       species="co2",
       time_resolved=True,
   )

The file name, ``site`` and ``domain`` are this example's: WBB is a tower
in Salt Lake City and ``slv`` names the Salt Lake Valley domain. Use your
site code and a domain name of your own.

openghg is not a dependency of PYSTILT, and the versions of Python, pandas
and numpy it supports can differ from PYSTILT's. Write the files with
PYSTILT and add them from an environment of their own
(``pip install openghg``).

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Without PYSTILT
---------------

:doc:`../reference/layout` describes every file and metadata key, for a
reader in another language.

The output directory is a set of Parquet files in folders named
``settings=...`` and ``date=...``, which many tools read as columns. With
`DuckDB <https://duckdb.org>`_, from Python, R, or its own shell, a
question about thousands of simulations is one query:

.. code-block:: sql

   -- footprints per settings folder and day
   SELECT settings, date, count(DISTINCT release) AS footprints
   FROM read_parquet('output/footprint/*/*/*.parquet', hive_partitioning = true)
   GROUP BY settings, date
   ORDER BY settings, date;

   -- the total influence of each release's footprint
   SELECT release, sum(foot) AS total
   FROM read_parquet('output/footprint/settings=hrrr-93278c/*/*.parquet', hive_partitioning = true)
   GROUP BY release;

   -- where each particle of one day's releases ended up
   SELECT release, particle, arg_min(lon, time) AS lon, arg_min(lat, time) AS lat
   FROM read_parquet('output/particles/settings=hrrr-a3f9c2/date=2021-01-15/*.parquet', hive_partitioning = true)
   GROUP BY release, particle;

A footprint file holds the cells the particles reached, as ``hour``,
``y``, and ``x`` (the cell's row and column in the grid that the folder's
``_settings.yaml`` gives) and ``foot``. An empty footprint is a file with
no rows, so a count of rows leaves it out. An ensemble's folders have a
``realization=k`` level as well, so its files are one folder deeper
(``*/*/*/*.parquet``) and ``realization`` is a column.
