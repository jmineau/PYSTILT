Your First Footprint
====================

You will make the footprint of one measurement, a map of where its air came
from, with two function calls in Python. Then you will run the same
measurement as a project, which is how you run thousands of them and keep
the results. The command-line version of the project is at the end.

Before you start
----------------

STILT moves particles with wind fields from a weather model, stored as
:term:`ARL`-format files. You need files covering your measurement time and
every hour of the run before it: for a 24-hour backward run, from 24 hours
before the measurement to just after it. A simulation with a file missing
fails, naming the hours.

If you already have ARL files, for example in a group archive, tell PYSTILT
the folder they are in. Each file's header says which times it holds, so their
names do not matter:

.. code-block:: python

   met = "/path/to/arl/hrrr"

If you don't have ARL files, PYSTILT can download them from NOAA's archive
into a folder you choose. This needs ``pip install "pystilt[download]"``.

.. code-block:: python

   met = {
       "download": "hrrr",                   # which NOAA ARL archive
       "directory": "/path/to/download/folder",
       "subgrid_enable": True,               # keep only your region
       "subgrid_bounds": {"xmin": -114, "xmax": -110, "ymin": 39, "ymax": 42},
   }

.. warning::

   NOAA's files cover whole continents, so the first download for a given
   period is several gigabytes even though only your region is kept.
   Downloaded files are reused on later runs. See
   :doc:`../guides/meteorology` for the available products.

Step 1: Describe your measurement
---------------------------------

A release (receptor) is where and when you measured. This one is a sensor 10 m above
the ground on the University of Utah campus, at noon local time (19:00 UTC) on
14 January 2021:

.. ipython:: python

   import stilt

   release = stilt.PointRelease(
       time="2021-01-14 19:00",   # UTC
       longitude=-111.848,
       latitude=40.766,
       altitude=10,               # meters above ground level
   )

.. This page runs when the docs are built. It uses the met files the tests
   cache, which cover 14 and 15 January 2021 around Salt Lake, and works in a
   temporary folder so the project it makes is not left in the repository
   (docs/_ext/docs_examples.py).

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   met = docs_examples.met()
   docs_cwd = docs_examples.temp_workdir()

Step 2: Follow the particles
----------------------------

:func:`stilt.run_trajectories` releases particles at the release and
follows them back in time through the meteorology:

.. ipython:: python

   particles = stilt.run_trajectories(
       release,
       met,
       model="hysplit", # the transport model that moves the particles
       n_hours=-12,     # follow the air 12 hours back in time
       numpar=200,      # number of particles to release
       krand=2,         # the random numbers a seed can fix
       seed=42,         # same seed, same particles
   )

``n_hours``
   How far back in time to follow the air. Negative means backward, which is
   what you want for measurements. 24 to 72 hours is typical. This page uses 12
   because its met files start 13 hours before the measurement.

``numpar``
   How many particles to release. More particles give a smoother footprint
   but take longer. 200 is enough to see the shape. For results you will
   publish we use 1000, STILT-R's default, because a footprint from fewer
   particles is noisy far from the release.

``krand`` and ``seed``
   HYSPLIT draws random numbers for the turbulence that moves each particle.
   ``krand=2`` with a ``seed`` makes the draws the same on every run, so your
   particles match the ones on this page. A project's config sets
   ``krand: 4``, STILT-R's value, under which every run draws its own, and the
   particles and the footprint change a little from run to run.

``directory``
   Where you unpacked HYSPLIT, when ``hycs_std`` is not on your ``PATH``
   (:doc:`installation`). Add ``directory="/path/to/hysplit"`` to the call.

Every other HYSPLIT setting takes HYSPLIT's own default here. A few of those
differ from the values STILT-R runs HYSPLIT with, such as whether particles
mix by convection. A project, below, writes STILT-R's values into its config.

This runs HYSPLIT, the transport model, and usually takes a minute or two.
The particles are a :class:`pandas.DataFrame`, one row per particle per
time step:

.. ipython:: python

   particles.head()

   @savefig quickstart_particles.png width=6in
   particles.stilt.plot.map();

The particles start at the release and spread out. The met files here cover
only a small region around Salt Lake, and a particle that leaves the met stops
there. This is how many hours they went (``age`` is in seconds):

.. ipython:: python

   print(f"{particles['age'].abs().max() / 3600:.1f} hours")

If that is short of ``n_hours``, the footprint covers only those hours.
:doc:`../guides/checking` says when that matters.

Step 3: Calculate the footprint
-------------------------------

The footprint is calculated on a grid. Make it big enough to include the
areas upwind of your site. Here the cells are 0.01°, about 1 km:

.. ipython:: python

   grid = stilt.Grid(
       xmin=-113.0, xmax=-110.5,   # longitude range
       ymin=40.0,   ymax=42.0,     # latitude range
       xres=0.01,   yres=0.01,     # grid cell size in degrees
   )
   foot = stilt.calc_footprint(particles, release, grid)

   @savefig quickstart_footprint.png width=6in
   foot.stilt.plot.map();

The footprint starts at the release and stretches in the direction the air
came from, here to the north-northwest and then curving west. The map shows the
footprint summed over time, on a log scale. Cells with more color influenced the
measurement more.

The footprint is an :class:`xarray.DataArray` with dimensions
``(time, lat, lon)``, in units of ppm per (µmol m⁻² s⁻¹). It has one map for
each hour the particles were still inside the met files. Sum it over time with
xarray:

.. ipython:: python

   foot.sizes
   foot.sum("time").sizes

PYSTILT's own methods on the particles and the footprint are under
``.stilt``.

Step 4: Run it as a project
---------------------------

Two calls are enough for one release. For many releases, make a
:class:`~stilt.Project`: a folder that holds your releases and settings,
runs every release, and keeps the results. :meth:`Project.init
<stilt.Project.init>` creates the folder and writes both to it. The
settings are the ones above, each in the block of what it belongs to: the
transport model's under ``models``, and the footprint's under ``stages``,
in a block that names the footprint stage as its ``class``:

.. ipython:: python

   project = stilt.Project.init(
       "./my_first_project",
       releases=[release],
       mets={"hrrr": met},          # "hrrr" is a name you choose
       models={
           "hysplit": {"n_hours": -12, "numpar": 200, "krand": 2, "seed": 42}
       },
       stages={
           "footprint": {"class": "stilt.footprint.FootprintStage", "grid": grid}
       },
       variants={"hrrr": {}},       # one variant: these settings, met hrrr
   )

Before running, check the project. ``project.check()`` looks for what would
stop a run before it starts: met files missing for a release's hours, a
release outside the grid, and a met folder that does not exist. It returns a
table with a row for each thing found, so an empty table means nothing is in
the way:

.. ipython:: python

   project.check()

The check looks for files, not inside them, so a run that passes it can
still fail later, if a met file is damaged. :doc:`../guides/checking` says how
to tell.

.. ipython:: python

   project.run();

``run()`` runs the particles and the footprint of every release, and
returns when they are done.

PYSTILT runs each release once per :term:`variant`. With the settings above
there is one variant, named ``hrrr`` after the met. One release under
one variant is a :term:`simulation`. Look yours up by the release's id and
the variant name:

.. ipython:: python

   sim = project.simulation(release.id, "hrrr")
   sim.particles.head()

   @savefig quickstart_sim_footprint.png width=6in
   sim.footprint.stilt.plot.map();

This footprint can differ a little from the one in Step 3. The seed is the
same, but ``Project.init`` wrote STILT-R's HYSPLIT values into the project's
``config.yaml`` where they differ from HYSPLIT's own defaults, which
``run_trajectories`` used in Step 2. Open the file to see them;
:doc:`../guides/migration/stilt_r` lists them.

What PYSTILT wrote
------------------

Your inputs are in the project folder, and the results in its ``output``
folder:

.. code-block:: text

   my_first_project/
     config.yaml        # your settings
     releases.csv       # your releases
     output/
       particles/       # where each particle went, one file per simulation
       footprint/       # the footprints, one file per simulation
       logs/            # HYSPLIT's log of each run

Each result is kept under the settings it was made with. Change a setting and
the next run writes new files beside the old ones, so nothing is
overwritten. :doc:`../reference/layout` has the full layout.

Your settings and releases are saved in the folder, so you can open the
project again later without repeating them:

.. code-block:: python

   project = stilt.Project("./my_first_project")

``Project.init`` runs once per project. It stops with a
``FileExistsError`` if the folder already has a ``config.yaml``. To change a
setting later, edit ``config.yaml``.

If you call ``project.run()`` again, nothing happens. PYSTILT sees that the
outputs already exist and skips them. If you add releases with
``project.add_releases(...)``, only the new ones run.

The same run from the command line
----------------------------------

Create a project folder with a starter ``config.yaml`` and ``releases.csv``:

.. code-block:: bash

   stilt init ./my_first_project

Open ``my_first_project/config.yaml`` and set the meteorology folder and the
footprint grid. The blocks are the same as in the Python example
above:

.. code-block:: yaml

   mets:
     hrrr: /path/to/arl/hrrr

   models:
     hysplit:
       n_hours: -12
       numpar: 200
       krand: 2
       seed: 42

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid:
         xmin: -113.0
         xmax: -110.5
         ymin: 40.0
         ymax: 42.0
         xres: 0.01
         yres: 0.01

   variants:
     hrrr: {}

Add your release to ``my_first_project/releases.csv`` below the header
line, so the file reads:

.. code-block:: text

   time,longitude,latitude,altitude
   2021-01-14 19:00:00,-111.848,40.766,10

Then check the project, run it, and look at the result:

.. code-block:: bash

   stilt check ./my_first_project
   stilt run ./my_first_project
   stilt status ./my_first_project

``stilt check`` lists what would stop a run, such as a met file that is not
there, and exits with 1 when it lists anything. When everything is in place it
prints ``No problems found.`` and exits with 0.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Next steps
----------

- Run many releases: :doc:`../tutorials/wbb_stationary`
- Understand the different release types: :doc:`../guides/releases`
- More plotting and analysis: :doc:`../guides/outputs`
- Run thousands of simulations on a cluster: :doc:`../guides/slurm`
