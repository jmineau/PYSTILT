.. _migration:

Migration Guide
===============

This page maps concepts and entry points from the R-STILT ecosystem to their
PYSTILT equivalents. Use it as a reference if you are porting existing workflows.


From R-STILT (uataq/stilt)
---------------------------

R-STILT organises a project around a ``setup.r`` file and a directory
structure maintained by R functions. PYSTILT expresses the same concepts as
Python classes, with configuration validated by Pydantic before any simulation
runs.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - R-STILT concept
     - R-STILT API
     - PYSTILT equivalent
   * - Project initialisation
     - ``stilt_init(project_path)`` → writes ``run/`` directory
     - :class:`~stilt.Model` constructor — creates project directory automatically
   * - Run controls
     - ``stilt_run(n_hours, numpar, …)`` arguments
     - :class:`~stilt.ModelConfig` / :class:`~stilt.STILTParams` fields
   * - Meteorology
     - ``met_loc``, ``met_file_format``, ``met_file_tres`` arguments to ``stilt_run``
     - :class:`~stilt.MetConfig` → ``ModelConfig.mets``
   * - Receptor list
     - ``receptors`` data.frame with ``run_time``, ``lati``, ``long``, ``zagl``
     - :func:`~stilt.read_receptors` (CSV with ``time``, ``lat``, ``lon``, ``zagl``)
   * - Footprint grid
     - ``xmn``, ``xmx``, ``ymn``, ``ymx``, ``xres``, ``yres`` arguments
     - :class:`~stilt.Grid` → :class:`~stilt.FootprintConfig` → ``ModelConfig.footprints``
   * - Parallelism
     - ``slurm=TRUE``, ``slurm_options=list(…)``
     - ``execution={"backend": "slurm"}`` in :class:`~stilt.ModelConfig`
   * - Before-footprint transform
     - ``before_footprint`` argument to ``stilt_run``
     - :class:`~stilt.config.FootprintConfig` ``transforms`` field or runtime particle transforms
   * - Trajectory output
     - ``/out/by-id/<sim_id>/<sim_id>_traj.rds``
     - ``<project>/simulations/<sim_id>/trajectories.parquet`` (Parquet)
   * - Footprint output
     - ``/out/by-id/<sim_id>/<sim_id>_foot.nc``
     - ``<project>/simulations/<sim_id>/footprints/<name>.nc`` (NetCDF)
   * - Simulation ID format
     - ``<YYYYMMDDHHMM>_<lati>x<long>x<zagl>``
     - ``<met>_<YYYYMMDDHHMM>_<location_id>``


From stiltctl (uataq/stiltctl)
--------------------------------

stiltctl is a Kubernetes-based Python toolkit for running STILT at cloud scale.
PYSTILT now covers the queue/runtime substrate across local, Slurm, and
Kubernetes, but it still stops short of stiltctl's staged scene/domain
application layer.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - stiltctl concept
     - stiltctl API
     - PYSTILT equivalent
   * - Scene / pixel configuration
     - ``DomainConfig``, ``SimulationConfig``
     - :class:`~stilt.ModelConfig` (one config per project)
   * - Met aggregation to space-time domain
     - ``minimize_meteorology()``
     - :class:`~stilt.MetStream` ``get_files()`` (automatic file selection)
   * - Job dispatch
     - Kubernetes jobs + PostgreSQL queue
     - :class:`~stilt.service.Service`, :class:`~stilt.executors.KubernetesExecutor`, and repository-backed claims
   * - Simulation state tracking
     - PostgreSQL ``simulations`` table
     - :class:`~stilt.repositories.SQLiteRepository` or :class:`~stilt.repositories.PostgreSQLRepository`
   * - Fault tolerance / resume
     - Kubernetes restart policies
     - claim/lease recovery plus ``skip_existing=True`` on rerun
   * - Worker deployment helpers
     - Helm/KEDA manifests
     - ``stilt.service.kubernetes`` helpers and example Kubernetes manifests
   * - Trajectory serialisation
     - ``.rds`` → Parquet conversion
     - Parquet natively (no R dependency)


From X-STILT (uataq/X-STILT)
-------------------------------

X-STILT (`Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_) extends
R-STILT for column/satellite footprints. PYSTILT now provides native
observation-layer building blocks for X-STILT-style workflows, while more
specialized background and chemistry pipelines remain outside core.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - X-STILT concept
     - X-STILT API / file
     - PYSTILT equivalent
   * - Column receptor definition
     - ``run_xstilt.r`` — lat/lon, ``agl``, ``dpar``, ``numpar``
     - :func:`~stilt.observations.build_slant_receptor` or :meth:`~stilt.Receptor.from_points`
   * - Vertical weighting (AK × PWF)
     - ``before_footprint`` function in R
     - :class:`~stilt.config.FootprintConfig` ``transforms`` or runtime particle transforms (see :doc:`advanced/custom_hooks`)
   * - Column footprint output
     - ``_X_traj.rds``, ``_X_foot.nc``
     - Same trajectory/footprint format as point receptors
   * - Horizontal transport error
     - ``run_hor_err=T``, ``siguverr``, ``tluverr``
     - :class:`~stilt.config.ErrorParams` ``siguverr``, ``tluverr``; :attr:`~stilt.Simulation.error_trajectories`
   * - Vertical transport error
     - ``run_ver_err=T``, ``szinit``
     - :class:`~stilt.config.ErrorParams` ``szinit``
   * - Forward background trajectories
     - ``compute_bg.r``
     - Not yet implemented in PYSTILT (set ``n_hours > 0`` for forward mode)
   * - OCO-2 / TROPOMI file reading
     - Built-in R readers
     - Use `netCDF4 <https://pypi.org/project/netCDF4/>`_ or
       `xarray <https://xarray.pydata.org/>`_ directly


Key Differences to Be Aware Of
--------------------------------

* **No R dependency.** PYSTILT bundles a pre-compiled HYSPLIT binary; no R or
  Fortran compilation is required.
* **Configuration is validated.** Pydantic raises a clear error if you supply
  an invalid parameter value *before* any simulation starts, so you no longer
  discover typos in the output log.
* **Met file selection is automatic.** You supply a directory and format string;
  PYSTILT finds the correct ARL files for each receptor time. You do not need
  to manually build file lists.
* **SimID includes the met name.** R-STILT's ``<YYYYMMDDHHMM>_latiXlongXzagl``
  becomes ``<met>_<YYYYMMDDHHMM>_<location_id>``, allowing multiple met runs
  side by side in the same project.
* **Footprint signature changed.** R-STILT's ``foot_res`` / ``foot_lims`` become
  a :class:`~stilt.Grid` inside a :class:`~stilt.FootprintConfig`. Units and
  CF metadata are preserved.
