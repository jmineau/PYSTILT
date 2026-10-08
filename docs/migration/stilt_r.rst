Coming From STILT-R
===================

PYSTILT does the same science as STILT-R, and its footprints match STILT-R's
cell by cell. The one exception is forward runs with ``hnf_plume`` on, where
PYSTILT fixes how the plume grows (see :ref:`stilt-r-parity`).

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
       by variant settings and day (:doc:`../guides/projects`)

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
       (:doc:`../guides/transport_error`)
   * - ``n_nodes`` and ``n_cores`` with Slurm
     - ``execution: backend: slurm``, with ``n_workers`` array tasks of
       ``cpus`` CPUs each (:doc:`../guides/slurm`)
   * - ``slurm_options``
     - ``time``, ``mem``, ``partition``, ``account``, and ``qos`` under
       ``execution``, and any other ``sbatch`` option under
       ``execution: slurm:``
   * - ``n_cores`` without Slurm
     - ``execution: cpus``
   * - ``before_footprint``
     - ``transforms`` (:doc:`../guides/transforms`)
   * - ``before_trajec``
     - no equivalent
   * - ``run_trajec``, ``run_foot``
     - not needed. ``stilt run`` runs whatever outputs are missing, and
       ``stilt run --no-skip`` reruns everything. To make a second
       footprint from the same particles, add a variant that changes only
       footprint settings; it shares the particles
       (:doc:`../guides/projects`).
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
:doc:`../guides/projects`). One project can run the same receptors with
several mets, a range of ``ziscale`` values, or a second
footprint grid. Each variant gets its own folder.

The transport error run is a variant too. STILT-R runs HYSPLIT a second time
with the wind-error settings and stores both sets of particles in one
``_traj.rds``. In PYSTILT that second run is its own variant, with its own
folder and log, and it can repeat several times
(:doc:`../guides/transport_error`).

``stilt run`` skips simulations whose outputs already exist. After a failure,
run it again and only the missing simulations run. Adding a variant runs
only the new variant.

PYSTILT never overwrites a result. STILT-R deletes ``out/by-id`` and starts
over whenever ``run_trajec = TRUE``. In PYSTILT a variant's results live in
:term:`settings folders <settings folder>`, so changed settings run into a
new folder and the old one stays until you delete it (see
:doc:`../guides/projects`).

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
an ``r_idx`` value. See :doc:`../guides/receptors`.

Outputs
-------

Particles are saved as Parquet files instead of R ``.rds`` files. Open them
in Python with pandas or :func:`stilt.read_particles`, and in R with
``arrow::read_parquet``. Footprints are saved as Parquet too, holding only
the cells the particles reached. ``foot.stilt.to_netcdf(path)`` writes one
as CF NetCDF with dimensions ``(time, lat, lon)``, as STILT-R does, and
keeps the receptor and the settings in it.

A point or column receptor's id has the same form as STILT-R's simulation
ID, and names its files: ``date=2015-07-05/201507050000_-111.8472_40.7665_21.parquet``
in each variant's settings folder (see :doc:`../guides/projects`). A
simulation is that id under a variant, ``201507050000_-111.8472_40.7665_21/hrrr``.

Moving a project over
---------------------

1. Copy your ``run_stilt.r`` settings into ``config.yaml`` using the table
   above. Start with one met and one variant that runs the defaults
   (``variants: {hrrr: {}}``).
2. Write your receptors to ``receptors.csv``.
3. Run a few receptors that you already have STILT-R results for, and
   compare the footprints.
4. Run everything.
