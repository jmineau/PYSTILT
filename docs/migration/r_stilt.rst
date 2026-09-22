Coming From STILT-R
===================

PYSTILT does the same science as STILT-R, and its footprints match STILT-R's
cell by cell. What changes is where the settings live. Instead of editing
variables in ``run_stilt.r``, you write them in ``config.yaml`` (or pass them
to :class:`stilt.Model`), and receptors go in ``receptors.csv``.

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
     - ``simulations/by-id/<id>/``, ``_traj.parquet`` and ``_foot.nc``

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
     - ``mets: <name>: n_min``
   * - ``met_subgrid_enable``, ``met_subgrid_buffer``,
       ``met_subgrid_levels``
     - ``mets: <name>: subgrid_enable``, ``subgrid_buffer``,
       ``subgrid_levels``
   * - ``xmn``, ``xmx``, ``ymn``, ``ymx``
     - ``footprints: <name>: xmin``, ``xmax``, ``ymin``, ``ymax``
   * - ``xres``, ``yres``, ``projection``
     - ``footprints: <name>: xres``, ``yres``, ``projection``
   * - ``smooth_factor``, ``time_integrate``
     - ``footprints: <name>: smooth_factor``, ``time_integrate``
   * - ``n_hours``, ``numpar``, ``hnf_plume``, ``rm_dat``, ``timeout``,
       ``varsiwant``
     - same names, top level
   * - HYSPLIT settings (``capemin``, ``delt``, ``kmix0``, ``tlfrac``,
       ``veght``, …)
     - same names, top level
   * - wind and mixing-depth error settings (``siguverr``, ``tluverr``,
       ``sigzierr``, …)
     - same names, top level
   * - ``slurm = TRUE``, ``n_nodes``, ``n_cores``, ``slurm_options``
     - ``execution: backend: slurm``, ``n_workers``, ``cpus_per_task``,
       plus ``sbatch`` options (:doc:`../guides/execution/slurm`)
   * - ``n_cores`` without Slurm
     - ``execution: n_workers``
   * - ``before_footprint`` / ``before_trajec`` functions
     - ``transforms`` on a footprint (:doc:`../advanced/transforms`)
   * - ``stilt_wd``, ``output_wd``
     - the project folder
   * - ``lib.loc``
     - not needed

A few differences worth knowing:

- **Meteorology has a name.** Each meteorology source is named (``hrrr``),
  and the name starts every simulation ID, so one project can run the same
  receptors with several meteorology products.
- **Footprints have names.** You can make several footprints (grids) from one
  set of particles, each with its own file. Adding a footprint to an existing
  project calculates it from the saved trajectories without rerunning
  HYSPLIT.
- **Reruns are automatic.** ``stilt run`` skips simulations whose outputs
  already exist, so rerunning after a failure only runs what's missing.

Receptors
---------

``receptors.csv`` uses the same information as STILT-R's receptor data frame.
STILT-R's column names (``long``, ``lati``, ``zagl``) are accepted, but the
time column must be named ``time``:

.. code-block:: text

   time,long,lati,zagl
   2015-07-05 00:00:00,-111.8472,40.7665,21

Column and multipoint receptors are rows sharing an ``r_idx``, as in
STILT-R. See :doc:`../guides/receptors`.

Outputs
-------

Trajectories are Parquet files instead of R ``.rds`` files. They open in
Python (pandas, :class:`stilt.Trajectories`) and in R (``arrow::read_parquet``).
Footprints are NetCDF as before, with dimensions ``(time, lat, lon)``.

Simulation IDs gain the met name at the front:
``hrrr_201507050000_-111.8472_40.7665_21`` instead of
``201507050000_-111.8472_40.7665_21``.

Moving a project over
---------------------

1. Copy your ``run_stilt.r`` settings into ``config.yaml`` using the table
   above. Start with one meteorology source and one footprint.
2. Write your receptors to ``receptors.csv``.
3. Run a few receptors you already have STILT-R results for and compare the
   footprints.
4. Then run everything.
