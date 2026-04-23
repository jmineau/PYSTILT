Migrating From R-STILT
======================

R-STILT organizes projects around R functions, ``setup.r`` scripts, and
filesystem conventions. PYSTILT expresses the same broad ideas through Python
classes, ``config.yaml``, and a durable simulation index.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - R-STILT concept
     - R-STILT API
     - PYSTILT equivalent
   * - Project initialization
     - ``stilt_init(project_path)``
     - ``stilt init`` or ``stilt.Model(project=...)``
   * - Run controls
     - ``stilt_run(...)`` arguments
     - :class:`stilt.ModelConfig` fields
   * - Meteorology
     - ``met_loc``, ``met_file_format``, ``met_file_tres``
     - :class:`stilt.MetConfig` in ``ModelConfig.mets``
   * - Receptors
     - receptor data.frame
     - :class:`stilt.Receptor` or :func:`stilt.read_receptors`
   * - Footprint grid
     - ``xmn``, ``xmx``, ``ymn``, ``ymx``, ``xres``, ``yres``
     - :class:`stilt.Grid` inside :class:`stilt.FootprintConfig`
   * - Parallel execution
     - ``slurm=TRUE`` and scheduler options
     - ``execution.backend: slurm`` plus executor config
   * - Before-footprint transform
     - callback hooks in R
     - ``FootprintConfig.transforms`` or runtime particle transforms
   * - Trajectory output
     - R serialized trajectory objects
     - Parquet trajectory files
   * - Footprint output
     - NetCDF footprint outputs
     - NetCDF footprint outputs with durable per-name status
   * - Simulation identity
     - timestamp plus location string
     - ``<met>_<YYYYMMDDHHMM>_<location_id>``

Key differences
---------------

- configuration is validated before execution
- project, output, and compute roots can be separate
- durable status and reruns are first-class rather than incidental
- local, Slurm, and Kubernetes execution share one runtime model

Practical advice
----------------

- Port one receptor family at a time.
- Start with one met stream and one footprint.
- Treat ``config.yaml`` as your migration boundary.
- Validate parity on a few representative receptors before scaling out.
