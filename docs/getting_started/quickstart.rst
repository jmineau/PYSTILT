.. _quickstart:

Quickstart
==========

This page walks through a minimal end-to-end STILT simulation. You will need
meteorological data in ARL format on disk. See :doc:`../user_guide/meteorology`
for how to obtain it.

The complete example below runs backward trajectories from a rooftop receptor
at the University of Utah, then computes a footprint showing the surface areas
that influenced that measurement.

.. code-block:: python

   import pandas as pd
   import stilt

   # 1. Define a receptor: where and when you made (or want to simulate) a measurement
   receptor = stilt.Receptor(
       time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
       latitude=40.766,
       longitude=-111.848,
       altitude=10,         # metres above ground level
   )

   # 2. Configure meteorology: tell PYSTILT where your ARL files live
   met = stilt.MetConfig(
       directory="/path/to/met/hrrr",
       file_format="hrrr_%Y%m%d.arl",
       file_tres="1h",
   )

   # 3. Configure the output footprint grid
   grid = stilt.Grid(
       xmin=-113.0, xmax=-110.5,
       ymin=40.0,   ymax=42.0,
       xres=0.01,   yres=0.01,
   )

   # 4. Build a project-level config (all STILT parameters + met + footprint)
   config = stilt.ModelConfig(
       n_hours=-24,                # 24-hour back-trajectories
       numpar=100,                 # number of particles
       mets={"hrrr": met},
       footprints={"default": stilt.FootprintConfig(grid=grid)},
   )

   # 5. Create the Model and run
   model = stilt.Model(
       project="/path/to/my_project",
       receptors=[receptor],
       config=config,
   )
   handle = model.run()
   handle.wait()

   # 6. Access results
   sim = list(model.simulations.values())[0]

   # Particle trajectories (pandas DataFrame)
   trajs = sim.trajectories
   print(trajs.data.head())

   # Gridded footprint (xarray DataArray)
   foot = sim.get_footprint("default")
   print(foot.data)


Understanding the Output
------------------------

**Trajectories** are stored as ``sim.trajectories.data``, a :class:`pandas.DataFrame`
with columns for particle position (``lon``, ``lat``, ``zagl``),
time step, and HYSPLIT diagnostic fields.

**Footprints** are stored as ``sim.get_footprint(name).data``, an
:class:`xarray.DataArray` with dimensions ``(time, lat, lon)`` and units of
``ppm / (µmol m⁻² s⁻¹)``. To obtain a time-integrated footprint:

.. code-block:: python

   foot_integrated = foot.integrate_over_time()  # sum all time steps
   # or restrict to a subset:
   foot_integrated = foot.integrate_over_time(*foot.time_range)


CLI Quick Reference
-------------------

The same workflow is available from the command line:

.. code-block:: bash

   # Initialise a project (creates config.yaml + receptors.csv)
   stilt init /path/to/my_project

   # Run trajectories and footprints, block until done
   stilt run /path/to/my_project

   # Submit to the configured backend (Slurm: returns immediately)
   stilt submit /path/to/my_project

   # Drain queued work with a batch worker
    stilt pull-worker /path/to/my_project --cpus 4

   # Long-lived queue worker
   stilt serve /path/to/my_project --cpus 4

   # Check simulation counts
   stilt status /path/to/my_project


Next Steps
----------

* :doc:`../user_guide/index`: full documentation of every configuration option.
* :doc:`../user_guide/service`: queue workers, `stilt.Service`, and cloud-facing helpers.
* :doc:`../user_guide/observations`: observation records, scenes, sensors, and transforms.
* :doc:`../tutorials/wbb_stationary`: step-by-step tutorial with real data.
* :doc:`../user_guide/execution`: parallelise across cores or a Slurm cluster.
