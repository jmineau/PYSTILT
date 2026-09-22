Your First Footprint
====================

This page takes you from one measurement to a footprint map. It uses Python;
the same run from the command line is at the end.

You will:

1. describe one measurement (a :term:`receptor`)
2. point PYSTILT at :term:`meteorology`
3. run the simulation
4. plot the footprint

Before you start: meteorology
-----------------------------

STILT moves particles with wind fields from a weather model, stored as
:term:`ARL`-format files. You need files covering your measurement time and
the hours before it. There are two ways to get them.

**You already have ARL files** (for example a group archive). You need three
things: the folder they are in, a pattern describing their names, and how many
hours each file covers. For files named like ``20230715_18``, each holding
six hours:

.. code-block:: python

   met = {
       "directory": "/path/to/arl/hrrr",
       "file_format": "%Y%m%d_%H",   # date codes: %Y year, %m month, %d day, %H hour
       "file_tres": "6h",            # each file covers 6 hours
   }

**You don't have ARL files.** PYSTILT can download them from NOAA's archive
into a folder you choose. This needs ``pip install "pystilt[cloud]"``:

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
settings for the run. Everything lives in a project folder, created if it
doesn't exist:

.. code-block:: python

   model = stilt.Model(
       project="./my_first_project",
       receptors=[receptor],
       mets={"hrrr": met},          # "hrrr" is a name you choose
       n_hours=-24,                 # follow the air 24 hours back in time
       numpar=200,                  # number of particles to release
       footprints={
           "slc": {                 # a name you choose for this footprint
               "xmin": -113.0, "xmax": -110.5,   # longitude range
               "ymin": 40.0,   "ymax": 42.0,     # latitude range
               "xres": 0.01,   "yres": 0.01,     # grid cell size in degrees
           }
       },
   )

What these settings mean:

``n_hours``
   How far back in time to follow the air. Negative means backward, which is
   what you want for measurements. 24 to 72 hours is typical.

``numpar``
   How many particles to release. More particles give a smoother footprint
   but take longer. 200 is fine for a first try; 500 to 1000 is common for
   research.

``footprints``
   The map grid the footprint is calculated on. Make it big enough to
   include the areas upwind of your site. The resolution here, 0.01°, is
   about 1 km.

Step 3: Run
-----------

.. code-block:: python

   model.run()

This runs HYSPLIT, moves the particles, and calculates the footprint. A
single simulation like this usually takes a minute or two. ``run()`` returns
when it is done.

Step 4: Look at the footprint
-----------------------------

Each receptor and met combination is one :term:`simulation`. Grab the one you
just ran and plot its footprint:

.. code-block:: python

   sim = next(model.simulations.values())
   foot = sim.get_footprint("slc")

   foot.plot.map()

You should see influence concentrated near the receptor and trailing off in
the direction the air came from. The values are summed over all 24 hours; cells
with more color influenced the measurement more.

The footprint's data is an :class:`xarray.DataArray` with dimensions
``(time, lat, lon)`` and units of ppm per (µmol m⁻² s⁻¹):

.. code-block:: python

   foot.data                                      # hourly footprint
   foot.integrate_over_time().data                # summed over time

The particle paths are a :class:`pandas.DataFrame`, one row per particle per
time step:

.. code-block:: python

   sim.trajectories.data.head()
   sim.trajectories.plot.map()

What PYSTILT wrote
------------------

Everything is in the project folder:

.. code-block:: text

   my_first_project/
     config.yaml                 # your settings
     receptors.csv               # your receptors
     simulations/
       by-id/
         hrrr_202307151800_-111.848_40.766_10/
           stilt.log                                          # HYSPLIT log
           hrrr_202307151800_-111.848_40.766_10_traj.parquet  # particle paths
           hrrr_202307151800_-111.848_40.766_10_slc_foot.nc   # footprint

The folder name is the :term:`simulation ID`: met name, receptor time,
longitude, latitude, and altitude.

Because your settings and receptors are saved in the folder, you can open the
project again later without repeating them:

.. code-block:: python

   model = stilt.Model(project="./my_first_project")

Run ``model.run()`` again and nothing happens: PYSTILT sees the outputs
already exist and skips them. Add more receptors and only the new ones run.

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

   footprints:
     slc:
       xmin: -113.0
       xmax: -110.5
       ymin: 40.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

   n_hours: -24
   numpar: 200

Add your receptor to ``my_first_project/receptors.csv``:

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
