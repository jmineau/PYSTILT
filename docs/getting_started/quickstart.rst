Your First Footprint
====================

This page takes you from one measurement to a footprint map. It uses Python.
The same run from the command line is at the end.

You will:

1. describe one measurement (a :term:`receptor`)
2. point PYSTILT at :term:`meteorology`
3. run the simulation
4. plot the footprint

Before you start
----------------

STILT moves particles with wind fields from a weather model, stored as
:term:`ARL`-format files. You need files covering your measurement time and
the hours before it.

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
into a folder you choose. This needs ``pip install "pystilt[cloud]"``.

.. code-block:: python

   met = {
       "source": "hrrr",                     # which NOAA product
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
       altitude=10,               # metres above ground level
   )

Step 2: Set up the model
------------------------

A :class:`~stilt.Model` combines your receptors, your meteorology, and the
settings for the run. Everything is saved in a project folder, which PYSTILT
creates if it doesn't exist.

.. code-block:: python

   model = stilt.Model(
       project="./my_first_project",
       receptors=[receptor],
       mets={"hrrr": met},          # "hrrr" is a name you choose
       n_hours=-24,                 # follow the air 24 hours back in time
       numpar=200,                  # number of particles to release
       grid={                       # the footprint grid
           "xmin": -113.0, "xmax": -110.5,   # longitude range
           "ymin": 40.0,   "ymax": 42.0,     # latitude range
           "xres": 0.01,   "yres": 0.01,     # grid cell size in degrees
       },
   )

What these settings mean:

``n_hours``
   How far back in time to follow the air. Negative means backward, which is
   what you want for measurements. 24 to 72 hours is typical.

``numpar``
   How many particles to release. More particles give a smoother footprint
   but take longer. 200 is fine for a first try. Research runs often use
   500 to 1000.

``grid``
   The map grid the footprint is calculated on. Make it big enough to
   include the areas upwind of your site. The resolution here, 0.01°, is
   about 1 km.

Step 3: Run
-----------

.. code-block:: python

   model.run()

This runs HYSPLIT to move the particles, then calculates the footprint. A
single simulation like this usually takes a minute or two. ``run()`` returns
when it is done.

Step 4: Look at the footprint
-----------------------------

PYSTILT runs each receptor once per :term:`variant`. With the settings above
there is one variant, named ``hrrr`` after the met source. One receptor under
one variant is a :term:`simulation`. Look yours up by the receptor's id and
the variant name, and plot its footprint:

.. code-block:: python

   sim = model.simulations[receptor.id, "hrrr"]
   foot = sim.footprint

   foot.plot.map()

You should see the most influence near the receptor, trailing off in the
direction the air came from. The map shows the footprint summed over all 24
hours, on a log scale. Cells with more color influenced the measurement more.

The footprint's data is an :class:`xarray.DataArray` with dimensions
``(time, lat, lon)`` and units of ppm per (µmol m⁻² s⁻¹):

.. code-block:: python

   foot.data                     # hourly footprint
   foot.integrate_over_time()    # summed over time

The particle paths are a :class:`pandas.DataFrame`, one row per particle per
time step:

.. code-block:: python

   sim.trajectories.data.head()
   sim.trajectories.plot.map()

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

The folders are named after the variant, ``hrrr``, plus a short hash of the
settings it ran with. The file name is the receptor (time, longitude,
latitude, and altitude); receptor and variant together are the
:term:`simulation ID`. HYSPLIT's own input files, such as ``CONTROL`` and
``SETUP.CFG``, are written to scratch and removed when a run succeeds.

Your settings and receptors are saved in the folder, so you can open the
project again later without repeating them:

.. code-block:: python

   model = stilt.Model(project="./my_first_project")

If you call ``model.run()`` again, nothing happens. PYSTILT sees that the
outputs already exist and skips them. If you add receptors, only the new ones
run.

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

   grid:
     xmin: -113.0
     xmax: -110.5
     ymin: 40.0
     ymax: 42.0
     xres: 0.01
     yres: 0.01

   n_hours: -24
   numpar: 200

Replace the example line in ``my_first_project/receptors.csv`` with your
receptor, so the file reads:

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
- Run thousands of simulations on a cluster: :doc:`../guides/execution/slurm`
