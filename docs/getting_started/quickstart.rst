Your First Footprint
====================

This page takes you from one measurement to a footprint map. It uses Python.
The first part makes one footprint with two function calls. The second
part runs the same thing as a project, which is how you run many
receptors and keep the results. The same project from the command line is
at the end.

You will:

1. describe one measurement (a :term:`receptor`)
2. point PYSTILT at :term:`meteorology`
3. follow particles back in time from the receptor
4. calculate and plot the footprint
5. run the same receptor as a project

Before you start
----------------

STILT moves particles with wind fields from a weather model, stored as
:term:`ARL`-format files. You need files covering your measurement time and
every hour of the run before it: for a 24-hour backward run, from 24 hours
before the measurement to just after it. A simulation with a file missing
fails, naming the hours.

If you already have ARL files, for example in a group archive, tell PYSTILT
the folder they are in, a pattern for their names, and how many hours each
file covers. For files named like ``20230715_18``, each holding six hours:

.. code-block:: python

   met = {
       "directory": "/path/to/arl/hrrr",
       "file_format": "%Y%m%d_%H",   # date codes: %Y year, %m month, %d day, %H hour
       "file_tres": "6h",            # each file covers 6 hours
   }

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

A receptor is where and when you measured. This one is a sensor 10 m above
the ground on the University of Utah campus, at 18:00 UTC on 15 July 2023:

.. code-block:: python

   import stilt

   receptor = stilt.PointReceptor(
       time="2023-07-15 18:00",   # UTC
       longitude=-111.848,
       latitude=40.766,
       altitude=10,               # meters above ground level
   )

Step 2: Follow the particles
----------------------------

:func:`stilt.run_trajectories` releases particles at the receptor and
follows them back in time through the meteorology:

.. code-block:: python

   particles = stilt.run_trajectories(
       receptor,
       met,
       n_hours=-24,     # follow the air 24 hours back in time
       numpar=200,      # number of particles to release
   )

``n_hours``
   How far back in time to follow the air. Negative means backward, which is
   what you want for measurements. 24 to 72 hours is typical.

``numpar``
   How many particles to release. More particles give a smoother footprint
   but take longer. 200 is fine for a first try. Research runs often use
   500 to 1000.

This runs HYSPLIT, the transport model, and usually takes a minute or two.
The particles are a :class:`pandas.DataFrame`, one row per particle per
time step:

.. code-block:: python

   particles.head()
   particles.stilt.plot.map()

Step 3: Calculate the footprint
-------------------------------

The footprint is calculated on a grid. Make it big enough to include the
areas upwind of your site. Here the cells are 0.01°, about 1 km:

.. code-block:: python

   grid = stilt.Grid(
       xmin=-113.0, xmax=-110.5,   # longitude range
       ymin=40.0,   ymax=42.0,     # latitude range
       xres=0.01,   yres=0.01,     # grid cell size in degrees
   )
   foot = stilt.calc_footprint(particles, receptor, grid)

   foot.stilt.plot.map()

You should see the most influence near the receptor, trailing off in the
direction the air came from. The map shows the footprint summed over all 24
hours, on a log scale. Cells with more color influenced the measurement more.

The footprint is an :class:`xarray.DataArray` with dimensions
``(time, lat, lon)``, one map per hour, in units of ppm per
(µmol m⁻² s⁻¹). Sum it over time with xarray:

.. code-block:: python

   foot.sum("time")

PYSTILT's own methods on the particles and the footprint are under
``.stilt``.

Step 4: Run it as a project
---------------------------

Two calls are enough for one receptor. For many receptors, make a
:class:`~stilt.Project`: a folder that holds your receptors and settings,
runs every receptor, and keeps the results. :meth:`Project.init
<stilt.Project.init>` creates the folder and writes both to it. The
settings are the ones above:

.. code-block:: python

   project = stilt.Project.init(
       "./my_first_project",
       receptors=[receptor],
       mets={"hrrr": met},          # "hrrr" is a name you choose
       variants={"hrrr": {}},       # one variant: these settings, met hrrr
       n_hours=-24,
       numpar=200,
       grid=grid,
   )
   project.run()

``run()`` runs the particles and the footprint of every receptor, and
returns when they are done.

PYSTILT runs each receptor once per :term:`variant`. With the settings above
there is one variant, named ``hrrr`` after the met. One receptor under
one variant is a :term:`simulation`. Look yours up by the receptor's id and
the variant name:

.. code-block:: python

   sim = project.simulation(receptor.id, "hrrr")
   sim.footprint.stilt.plot.map()
   sim.particles.head()

What PYSTILT wrote
------------------

Your inputs are in the project folder, and the results in its ``output``
folder.

.. code-block:: text

   my_first_project/
     config.yaml                 # your settings
     receptors.csv               # your receptors
     output/
       particles/settings=hrrr-a3f9c2/date=2023-07-15/202307151800_-111.848_40.766_10.parquet
       footprints/settings=hrrr-93278c/date=2023-07-15/202307151800_-111.848_40.766_10.parquet
       logs/settings=hrrr-a3f9c2/date=2023-07-15/202307151800_-111.848_40.766_10.log

Each folder under ``particles``, ``footprints``, and ``logs`` is a
:term:`settings folder <settings folder>`: the variant's name, ``hrrr``,
plus a short code made from its settings. Change a setting and the next
run writes to a new folder, so nothing is overwritten. The file name is
the :term:`receptor id`: the time, longitude, latitude, and altitude.

Your settings and receptors are saved in the folder, so you can open the
project again later without repeating them:

.. code-block:: python

   project = stilt.Project("./my_first_project")

``Project.init`` runs once per project. It stops with a
``FileExistsError`` if the folder already has a ``config.yaml``. To change a
setting later, edit ``config.yaml``.

If you call ``project.run()`` again, nothing happens. PYSTILT sees that the
outputs already exist and skips them. If you add receptors with
``project.add_receptors(...)``, only the new ones run.

The same run from the command line
----------------------------------

Create a project folder with a starter ``config.yaml`` and ``receptors.csv``:

.. code-block:: bash

   stilt init ./my_first_project

Open ``my_first_project/config.yaml`` and set the meteorology folder, file
pattern, and footprint grid. The keys are the same as in the Python example
above:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /path/to/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 6h

   variants:
     hrrr: {}

   grid:
     xmin: -113.0
     xmax: -110.5
     ymin: 40.0
     ymax: 42.0
     xres: 0.01
     yres: 0.01

   n_hours: -24
   numpar: 200

Add your receptor to ``my_first_project/receptors.csv`` below the header
line, so the file reads:

.. code-block:: text

   time,longitude,latitude,altitude
   2023-07-15 18:00:00,-111.848,40.766,10

Then run it and check the result:

.. code-block:: bash

   stilt run ./my_first_project
   stilt status ./my_first_project

Next steps
----------

- Run many receptors: :doc:`../tutorials/wbb_stationary`
- Understand the different receptor types: :doc:`../guides/receptors`
- More plotting and analysis: :doc:`../guides/outputs`
- Run thousands of simulations on a cluster: :doc:`../guides/slurm`
