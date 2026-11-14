PYSTILT For HYSPLIT Users
=========================

If you run HYSPLIT to find where air came from or where a release went,
PYSTILT runs the same particle model, ``hycs_std``, from Python. It writes
``CONTROL`` and ``SETUP.CFG`` from a few settings, finds or downloads the
ARL met files, and returns every particle's position at every time step as
a pandas table. For many starting times and places, and for the same runs
under other settings, a project runs them all, on your computer or as a
Slurm job array, and keeps the particles of each set of settings in a
folder of its own.

.. ipython:: python
   :suppress:

   import docs_examples

   met_dir = docs_examples.met()

HYSPLIT's words in PYSTILT
--------------------------

PYSTILT comes from STILT, which follows air backward from a measurement and
calls the time and place its particles start from a *receptor*. For you, a
receptor is the starting time and starting locations of ``CONTROL``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - In HYSPLIT
     - In PYSTILT
   * - A starting time and one starting location
     - A :class:`~stilt.PointReceptor`
   * - Several starting locations at one time
     - A :class:`~stilt.MultiPointReceptor`
   * - Two heights at one location, a line source
     - A :class:`~stilt.ColumnReceptor`
   * - ``KMSL``, heights above ground or above sea level
     - The receptor's ``altitude_ref``, ``"agl"`` or ``"msl"``
   * - The settings of one ``CONTROL`` and ``SETUP.CFG``
     - A variant, a named set of settings
   * - The particle positions you read after a run
     - The particle table, one row per particle per time step

One run
-------

:func:`stilt.run_trajectories` runs HYSPLIT once and returns the particles.
Here 1,000 particles leave Salt Lake City at 500 m above ground during one
hour and are followed 6 hours forward:

.. ipython:: python

   import stilt

   start = stilt.PointReceptor(
       time="2021-07-15 01:00", longitude=-111.85, latitude=40.77, altitude=500
   )
   cloud = stilt.run_trajectories(
       start, met_dir, n_hours=6, numpar=1000, emisshrs=1.0,
       krand=2, seed=42, hnf_plume=False,
   )
   cloud[["time", "particle", "lon", "lat", "zagl"]].tail()

``n_hours`` is ``CONTROL``'s total run time: positive runs forward, negative
backward. ``emisshrs`` is the hours of emission, and ``zagl`` the height
above ground in meters. The other columns are the met at each particle that
``varsiwant`` asks for, by HYSPLIT's names (``mlht``, ``pres``, ``dens``,
...). We set ``hnf_plume=False`` whenever you want positions and no
footprint, because PYSTILT's near-field correction rewrites the ``foot``
column for STILT's footprints. ``krand=2`` with a ``seed`` makes the run
repeatable.

The met is the folder of your ARL files. PYSTILT reads which times each
file holds from its header, and gives HYSPLIT the ones the run needs.
:doc:`meteorology` shows how to download them from NOAA instead.

.. _single-trajectory:

A single trajectory
-------------------

``hyts_std``, HYSPLIT's trajectory model, follows the mean wind. One
particle with turbulence off follows the same path, so ``numpar=1`` and
``nturb=1`` give you a trajectory:

.. ipython:: python

   path = stilt.run_trajectories(
       start, met_dir, n_hours=6, numpar=1, nturb=1, hnf_plume=False
   )
   path[["time", "lon", "lat", "zagl"]].iloc[[0, 59, 179, 359]]

We compared this run with ``hyts_std`` on the same met. After 6 hours the two
are 0.001 degrees apart. The mean position of 1,000 particles with
turbulence on ended about 21 km away and 400 m lower:

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - After 6 hours
     - Longitude
     - Latitude
     - Height (m)
   * - ``hyts_std``
     - -110.717
     - 40.534
     - 1585
   * - One particle, turbulence off
     - -110.718
     - 40.533
     - 1644
   * - Mean of 1,000 particles
     - -110.961
     - 40.583
     - 1244

So the mean of many particles is the center of a cloud that spreads as it
goes, which is a different thing from a trajectory. For a trajectory, run one
particle.
For several trajectories from one time, use a
:class:`~stilt.MultiPointReceptor` with ``numpar`` equal to its number of
locations: HYSPLIT gives each location one particle.

Many runs: a project
--------------------

A project runs every starting point under every variant, and runs again
only what is missing. A ``config.yaml`` with no ``grid`` makes particles
and no footprints. Here every start runs on HRRR with two vertical motion
options, each kept in its own folder:

.. code-block:: yaml

   mets:
     hrrr: /data/arl/hrrr

   variants:
     hrrr: {}                 # the settings below
     hrrr-w6: {w_option: 6}   # the same runs with another vertical motion

   n_hours: 24
   numpar: 1000
   emisshrs: 1.0
   hnf_plume: false

The starting points go in ``receptors.csv``, one row each:

.. code-block:: text

   time,longitude,latitude,altitude
   2021-07-15 01:00:00,-111.85,40.77,500
   2021-07-15 13:00:00,-111.85,40.77,500

``stilt run ./my_project`` runs them here, and ``stilt submit
./my_project`` as a Slurm job array (:doc:`slurm`). Then read the particles
of any selection of runs as one table:

.. code-block:: python

   project = stilt.Project("my_project")
   sims = project.simulations
   particles = project.particles(sims[sims.variant == "hrrr-w6"])

For an ensemble of the same run with different turbulence, give a variant
``realizations: 10`` with ``krand: 2`` and a ``seed``
(:doc:`projects`).

Where each ``CONTROL`` and ``SETUP.CFG`` line comes from
--------------------------------------------------------

Every ``SETUP.CFG`` entry HYSPLIT reads is a setting of the same name in
lowercase, such as ``numpar``, ``kmixd`` or ``tlfrac``
(:doc:`../reference/configuration` lists them). PYSTILT writes the rest
itself:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Line
     - Comes from
   * - ``CONTROL`` starting time and locations
     - The receptor, latitude, longitude and height
   * - ``CONTROL`` total run time
     - ``n_hours``
   * - ``CONTROL`` vertical motion option, top of model domain
     - ``w_option``, ``z_top``
   * - ``CONTROL`` met grids and files
     - The met's files for the run's hours, as one grid
   * - ``CONTROL`` hours of emission
     - ``emisshrs``; the release starts at the starting time
   * - ``CONTROL`` pollutant, concentration grid and deposition
     - Fixed: one pollutant, a placeholder grid, no deposition
   * - ``SETUP.CFG`` ``KMSL``
     - The receptor's ``altitude_ref``
   * - ``SETUP.CFG`` ``MAXPAR``
     - ``maxpar``, or when unset enough for ``numpar`` shared over the locations
   * - ``SETUP.CFG`` ``IVMAX``
     - The number of ``varsiwant`` names
   * - ``SETUP.CFG`` ``SEED``
     - ``seed``, written as ``-(abs(seed) + 1)``
   * - ``SETUP.CFG`` ``WINDERRTF``, ``ZICONTROLTF``
     - The wind and mixed-layer error settings, and ``ziscale``

To read the files of one run, pass ``workdir="run"`` to
:func:`~stilt.run_trajectories`: the folder keeps ``CONTROL``,
``SETUP.CFG``, HYSPLIT's ``MESSAGE`` and the particle file.

What PYSTILT leaves to other tools
----------------------------------

PYSTILT runs ``hycs_std`` in its STILT mode (``ichem: 8``), which writes
particle positions and the met along them. Concentration grids
(``cdump``), the deposition and decay of a pollutant's mass, puffs and
``tdump`` trajectory files come from HYSPLIT run on its own, and NOAA's
`monetio <https://github.com/noaa-oar-arl/monetio>`_ reads those files in
Python.

When a run fails, :doc:`checking` says how to find why.
