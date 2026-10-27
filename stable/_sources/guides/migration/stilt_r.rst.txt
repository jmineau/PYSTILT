Coming From STILT-R
===================

PYSTILT does the same science as STILT-R, and its footprints match STILT-R's
cell by cell. Two cases differ on purpose, both in the near-field
correction (``hnf_plume``): forward runs, where PYSTILT fixes how the plume
grows, and a ``veght`` above 1, which PYSTILT reads as meters above ground,
as HYSPLIT does (see :ref:`stilt-r-parity`).

What changes is where the settings live. Instead of editing variables in
``run_stilt.r``, you write them in ``config.yaml`` or pass them to
:meth:`stilt.Project.init`. Receptors go in ``receptors.csv``.

The workflow side by side
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Step
     - STILT-R
     - PYSTILT
   * - Start a project
     - ``Rscript -e "uataq::stilt_init('my_project')"``
     - ``stilt init my_project``
   * - Set options
     - edit variables in ``r/run_stilt.r``
     - edit ``config.yaml``
   * - Define receptors
     - build a ``receptors`` data frame in ``run_stilt.r``
     - ``receptors.csv``, or receptor objects in Python
   * - Run
     - ``Rscript r/run_stilt.r``
     - ``stilt run my_project``
   * - Outputs
     - ``out/by-id/<id>/``, ``_traj.rds`` and ``_foot.nc``
     - the output directory: ``particles/`` and ``footprints/`` Parquet files
       by variant settings and day (:doc:`../projects`)

``run_stilt.r`` settings in PYSTILT
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``run_stilt.r``
     - ``config.yaml``
   * - ``met_path``
     - ``mets: <name>: directory``
   * - ``met_file_format``
     - ``mets: <name>: file_format`` (same strftime codes)
   * - ``met_file_tres``
     - ``mets: <name>: file_tres``
   * - ``n_met_min``
     - No setting. Every file a run needs must be there and whole, or the
       run fails (``MET_COVERAGE``). STILT-R runs with the files it finds
       and writes the shorter footprint.
   * - ``met_subgrid_enable``, ``met_subgrid_levels``
     - ``mets: <name>: subgrid_enable``, ``subgrid_levels``. PYSTILT also
       needs ``subgrid_bounds``, the crop box, and ``subgrid_dir`` for your
       own files.
   * - ``met_subgrid_buffer``
     - No setting. STILT-R crops to the footprint grid widened by this
       fraction of its size on each side. In PYSTILT, write the widened box
       as ``subgrid_bounds``: for a grid from -114 to -111 and a buffer of
       0.2, ``xmin: -114.6``, ``xmax: -110.4``.
   * - ``xmn``, ``xmx``, ``ymn``, ``ymx``
     - ``grid: xmin``, ``xmax``, ``ymin``, ``ymax``
   * - ``xres``, ``yres``, ``projection``
     - ``grid: xres``, ``yres``, ``crs`` (``projection`` is read too)
   * - ``smooth_factor``, ``time_integrate``
     - same names, top level
   * - ``n_hours``, ``numpar``, ``hnf_plume``, ``varsiwant``
     - same names, top level
   * - ``timeout``
     - ``execution: timeout``
   * - ``rm_dat``
     - no setting: a successful run's working directory is removed;
       ``execution: keep_workdir`` keeps it
   * - HYSPLIT settings (``capemin``, ``delt``, ``kmix0``, ``tlfrac``,
       ``veght``, …)
     - same names, top level
   * - wind and mixing-depth error settings (``siguverr``, ``tluverr``,
       ``sigzierr``, …)
     - same names, usually on a variant of their own
       (:doc:`../transport_error`)
   * - ``n_nodes`` and ``n_cores`` with Slurm
     - ``execution: backend: slurm``, with ``n_workers`` array tasks of
       ``cpus`` CPUs each (:doc:`../slurm`)
   * - ``slurm_options``
     - ``time``, ``mem``, ``partition``, ``account``, and ``qos`` under
       ``execution``, and any other ``sbatch`` option under
       ``execution: slurm:``
   * - ``n_cores`` without Slurm
     - ``execution: cpus``
   * - ``before_footprint``
     - ``transforms`` (:doc:`../transforms`)
   * - ``before_trajec``
     - no equivalent
   * - ``run_trajec``, ``run_foot``
     - not needed. ``stilt run`` runs whatever outputs are missing, and
       ``stilt run --no-skip`` reruns everything. To make a second
       footprint from the same particles, add a variant that changes only
       footprint settings; it shares the particles
       (:doc:`../projects`).
   * - ``simulation_id`` (run a subset)
     - ``project.run(receptors=[...])`` with the receptor ids you want,
       or ``stilt run --receptors ids.txt``
   * - ``ziscale`` as one list per receptor
     - ``ziscale`` per variant, as one factor or one factor per hour. It is
       the same for every receptor. Per-receptor factors are not supported
       yet.
   * - ``stilt_wd``, ``output_wd``
     - the project folder
   * - ``lib.loc``
     - not needed

Other differences
-----------------

A project runs its receptors under named *variants*. A variant is a
met plus any settings that differ from the defaults (see
:doc:`../projects`). One project can run the same receptors with
several mets, a range of ``ziscale`` values, or a second
footprint grid. Each variant gets its own folder.

The transport error run is a variant too. STILT-R runs HYSPLIT a second time
with the wind-error settings and stores both sets of particles in one
``_traj.rds``. In PYSTILT that second run is its own variant, with its own
folder and log, and it can repeat several times
(:doc:`../transport_error`).

``stilt run`` skips simulations whose outputs already exist. After a failure,
run it again and only the missing simulations run. Adding a variant runs
only the new variant.

PYSTILT never overwrites a result. STILT-R deletes ``out/by-id`` and starts
over whenever ``run_trajec = TRUE``. In PYSTILT a variant's results live in
:term:`settings folders <settings folder>`, so changed settings run into a
new folder and the old one stays until you delete it (see
:doc:`../projects`).

Receptors
---------

``receptors.csv`` holds the same information as STILT-R's ``receptors`` data
frame. STILT-R's column names ``long``, ``lati``, and ``zagl`` work as they
are. Rename ``run_time`` to ``time``. In the particle table, STILT-R's
``long``, ``lati``, ``indx``, ``time``, and ``xhgt`` are ``lon``, ``lat``,
``particle``, ``age``, and ``release_height``. PYSTILT's ``time`` column is
the UTC time of each row, so a script that filters on STILT-R's ``time``
in minutes must use ``age``.

.. code-block:: text

   time,long,lati,zagl
   2015-07-05 00:00:00,-111.8472,40.7665,21

STILT-R makes a column receptor from a row with several ``zagl`` values. In
``receptors.csv``, a column or multipoint receptor is several rows that share
an ``r_idx`` value. See :doc:`../receptors`.

Outputs
-------

Particles are saved as Parquet files instead of R ``.rds`` files. Open them
in Python with pandas or :func:`stilt.read_particles`, and in R with
``arrow::read_parquet``. Footprints are saved as Parquet too, holding only
the cells the particles reached. ``foot.stilt.to_netcdf(path)`` writes one
as CF NetCDF with dimensions ``(time, lat, lon)``, as STILT-R does, and
keeps the receptor and the settings in it.

A point receptor's id has the same form as STILT-R's simulation ID, and
names its files in each variant's settings folder:
``date=2015-07-05/201507050000_-111.8472_40.7665_21.parquet`` (see
:doc:`../projects`). A column receptor's id adds its bottom and top
after the ``X``.

Reading STILT-R output
----------------------

:func:`stilt.migration.read_stilt_r` reads the ``out/by-id`` folder of a
STILT-R project, so runs you already have work with PYSTILT's tools. It
reads X-STILT's ``out/by-id`` the same way. It needs the ``stilt-r`` extra,
which reads R's ``.rds`` files without R:

.. code-block:: bash

   pip install "pystilt[stilt-r]"

Each simulation gives its receptor, its particles, and its footprint, one
folder at a time:

.. code-block:: python

   from stilt.migration import read_stilt_r

   for receptor, particles, foot in read_stilt_r("my_project/out/by-id"):
       print(receptor.id, len(particles), foot is not None)

Each run's receptor comes from two files in its folder. The release points are
the ones in ``CONTROL``, the numbers HYSPLIT read, so a point receptor's id is
the folder's name. The time is the one in ``<id>_traj.rds``, which keeps the
seconds that ``CONTROL`` drops. A folder without ``CONTROL`` stops the read
with an error that names it.

A footprint's ``smooth_factor`` is not in ``<id>_foot.nc``. When STILT-R ran
through Slurm, rslurm kept every run's settings in
``my_project/_rslurm_my_project/params.RDS``, beside ``out/``, and
``smooth_factor`` is read from there. That file is used only when its
``n_hours`` is the one in each run's ``CONTROL`` and its ``numpar`` the one
HYSPLIT ran; otherwise the read stops and says which one differs. A project
run on one machine has no ``params.RDS``: give the ``smooth_factor`` your
``run_stilt.r`` set.

.. code-block:: python

   sims = read_stilt_r("my_project/out/by-id", smooth_factor=1)

Without either, reading a footprint stops with an error that names both. A
project of particles only (``run_foot = FALSE``) needs neither.

The particles are the table STILT-R saved in ``<id>_traj.rds``, with
PYSTILT's column names (``indx``, ``time``, ``long``, ``lati``, and ``xhgt``
become ``particle``, ``age``, ``lon``, ``lat``, and ``release_height``) and a
``time`` column of UTC times. The footprint is ``<id>_foot.nc`` as a PYSTILT
footprint, or ``None`` when the folder has none. Both are kept as STILT-R
made them: the near-field correction STILT-R applied stays in ``foot``, and
the footprint's values are the file's. To make a footprint on another grid,
pass the particles to :func:`stilt.calc_footprint`.

Some of what STILT-R saved is not read:

- With transport error, STILT-R saves a second set of particles in the same
  ``.rds`` (``particle_error``), the run with perturbed winds.
  ``read_stilt_r("my_project/out/by-id", error=True)`` reads that set in
  place of the first, with no footprint: STILT-R makes none of it.
- The HYSPLIT settings in the ``.rds`` (``params``), the rest of
  ``CONTROL`` (``n_hours``, the met files) and of ``params.RDS`` are not
  returned. :func:`~stilt.migration.import_stilt_r` records each run's
  settings (below). No file records the STILT-R version.
- A footprint on a projected grid (a ``projection`` other than
  longitude/latitude) raises an error; open it with
  :func:`xarray.open_dataset`.
- Other footprint files in a folder, such as X-STILT's footprints at other
  resolutions, are not read.
- X-STILT weights ``foot`` by the averaging kernel and pressure weighting
  just before it makes the footprint, after the ``.rds`` is saved. Its
  particles are unweighted and its footprint is weighted.

Importing STILT-R output into a project
---------------------------------------

:func:`stilt.migration.import_stilt_r` writes the runs of an ``out/by-id``
folder into a PYSTILT output directory, so a project reads them as its own:
``project.status``, ``project.particles``, ``project.footprints``, and
``project.jacobian``. Give it the STILT-R version the project ran, which no
file records, such as the commit ``git rev-parse --short HEAD`` prints in the
STILT-R project folder:

.. code-block:: python

   import stilt
   from stilt.migration import import_stilt_r

   receptors = import_stilt_r("my_project/out/by-id", "output", version="1d936c1")

It prints the met and the variant that name the imported folders, for
``config.yaml``:

.. code-block:: yaml

   mets:
     stilt-r:
       met_path: /data/hrrr
   variants:
     stilt-r:
       model: stilt-r
       met: stilt-r
       version: 1d936c1
       capemin: -1.0
       cmass: 0
       # ... every HYSPLIT setting, in name order
       ziscale: 1.0
       grid: {xmin: -112.5, xmax: -111.5, ymin: 40.25, ymax: 41.25, xres: 0.01,
         yres: 0.01, crs: +proj=longlat}
       geometry: null
       cells_per_target: 4.0
       smooth_factor: 1.0
       time_integrate: false
       transforms: []

and returns the receptors, for ``receptors.csv``. Paste the printed lines into
the ``config.yaml`` of a project whose ``output`` is that directory, then add
the receptors:

.. code-block:: python

   project = stilt.Project("my_project_py")
   project.add_receptors(receptors)
   project.status()

A ``stilt-r`` variant reads what the import wrote and runs nothing: a run of
one fails (``NOT_RUNNABLE``), so its receptors that STILT-R did not run stay
failed. Run them with a ``hysplit`` variant instead.

Every folder of an output directory is named by the hash of its settings
(:doc:`../projects`), so an imported run's settings hold only what STILT-R's
files record, never a default in place of a value they do not give. Two
STILT-R projects that differ only in ``n_hours`` land in two folders.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Where it comes from
   * - HYSPLIT's settings (``numpar``, ``kmix0``, ``veght``, ``varsiwant``, …)
     - each run's ``SETUP.CFG``
   * - ``n_hours``, ``w_option``, ``z_top``, ``emisshrs``
     - each run's ``CONTROL``
   * - ``ziscale``
     - the ``ZICONTROL`` file, when ``SETUP.CFG`` says STILT-R wrote one; else
       1, no scaling
   * - ``hnf_plume``
     - the particles: STILT-R's near-field correction adds
       ``foot_no_hnf_dilution``
   * - ``seed``
     - none: STILT-R sets no seed
   * - the met
     - ``met_path``, the directory the ``CONTROL`` files name, as they name it.
       Each run's met files are recorded in its particle file; they need not
       exist any more
   * - ``grid``, ``time_integrate``
     - ``<id>_foot.nc``
   * - ``smooth_factor``
     - ``params.RDS``, or the ``smooth_factor`` argument, as for
       :func:`~stilt.migration.read_stilt_r`
   * - ``version``
     - the ``version`` argument

The runs of one import must share their settings. A run whose settings
differ from the first run's stops the import, which names the setting. The
HYSPLIT settings of every run are compared before anything is written. To
import runs made with other settings, put them in a ``by-id`` folder of their
own and import it under another ``name``. When the ``CONTROL`` files name
more than one met directory, such as yearly folders, give the folder that
holds them: ``import_stilt_r(..., met_path="/data/hrrr")``. ``SETUP.CFG``
with heights above sea level (``KMSL=1``) stops the import too.

A run with transport error also writes its second set of particles, the
run with perturbed winds, to a second variant, ``stilt-r-err``, with the wind
and mixed-layer error settings STILT-R saved (``particle_error_params``) and
no grid. It is the same wind-error variant a PYSTILT project runs
(:doc:`../transport_error`). A run whose ``stilt.log`` says no particle
reached the footprint grid gets an empty footprint, as a PYSTILT run would.

Results already in the output directory are kept, so the import can run
again when more STILT-R runs finish. Reading the ``.rds`` files takes a few
seconds each, so import a large project in a Slurm job.

Moving a project over
---------------------

1. Copy your ``run_stilt.r`` settings into ``config.yaml`` using the table
   above. Start with one met and one variant that runs the defaults
   (``variants: {hrrr: {}}``).
2. Write your receptors to ``receptors.csv``.
3. Run a few receptors that you already have STILT-R results for, and
   compare the footprints.
4. Run everything.
