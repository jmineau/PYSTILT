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
     - ``simulations/by-id/<id>/<variant>/``, ``_traj.parquet`` and ``_foot.nc``

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
     - ``grid: xmin``, ``xmax``, ``ymin``, ``ymax``
   * - ``xres``, ``yres``, ``projection``
     - ``grid: xres``, ``yres``, ``projection``
   * - ``smooth_factor``, ``time_integrate``
     - same names, top level
   * - ``n_hours``, ``numpar``, ``hnf_plume``, ``rm_dat``, ``timeout``,
       ``varsiwant``
     - same names, top level
   * - HYSPLIT settings (``capemin``, ``delt``, ``kmix0``, ``tlfrac``,
       ``veght``, …)
     - same names, top level
   * - wind and mixing-depth error settings (``siguverr``, ``tluverr``,
       ``sigzierr``, …)
     - same names, on a separate variant (:doc:`../guides/transport_error`)
   * - ``slurm = TRUE``, ``n_nodes``, ``n_cores``, ``slurm_options``
     - ``execution: backend: slurm``, ``n_workers``, ``cpus_per_task``,
       plus ``sbatch`` options (:doc:`../guides/execution/slurm`)
   * - ``n_cores`` without Slurm
     - ``execution: n_workers``
   * - ``before_footprint`` / ``before_trajec`` functions
     - ``transforms`` (:doc:`../advanced/transforms`)
   * - ``run_trajec``, ``run_foot``
     - no flags: a missing output is what gets run. Another footprint from
       the same particles is a ``from:`` variant; to redo a variant, delete
       its outputs with ``stilt rm --variant NAME``
       (:doc:`../guides/project_layout`)
   * - ``simulation_id`` (run a subset)
     - a smaller ``receptors.csv``, or :func:`stilt.execution.run_receptors`
       with the receptor ids you want
   * - ``ziscale`` as one list per receptor
     - ``ziscale`` per variant (a scalar, or one factor per hour), the same
       for every receptor; per-receptor factors are not supported yet
   * - ``stilt_wd``, ``output_wd``
     - the project folder
   * - ``lib.loc``
     - not needed

A few differences worth knowing:

- **Settings have names.** A project runs its receptors under named
  *variants*: a meteorology source plus any settings that differ from the
  defaults (:doc:`../guides/configuration`). One project can run the same
  receptors with several meteorology products, a ``ziscale`` bracket, or a
  second footprint grid, each in its own folder.
- **The error run is a variant.** STILT-R runs HYSPLIT a second time with
  the wind-error settings and stores both particle sets in one
  ``_traj.rds``. In PYSTILT that second run is a variant of its own, with its
  own folder and log, and it can repeat several times
  (:doc:`../guides/transport_error`).
- **Reruns are automatic.** ``stilt run`` skips simulations whose outputs
  already exist, so rerunning after a failure only runs what's missing.
  Adding a variant runs only the new variant.
- **Settings are frozen once they have run.** STILT-R overwrites ``by-id``
  on every ``run_trajec = T``. PYSTILT records the settings each variant ran
  with and refuses to change them under the same name; give changed settings
  a new variant name, or ``stilt rm --variant NAME`` to start that variant
  over (:doc:`../guides/configuration`).

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

STILT-R's simulation ID is PYSTILT's receptor ID, and each variant is a
folder below it: ``201507050000_-111.8472_40.7665_21/hrrr``.

Moving a project over
---------------------

1. Copy your ``run_stilt.r`` settings into ``config.yaml`` using the table
   above. Start with one meteorology source and no ``variants``.
2. Write your receptors to ``receptors.csv``.
3. Run a few receptors you already have STILT-R results for and compare the
   footprints.
4. Then run everything.
