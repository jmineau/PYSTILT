Glossary
========

.. glossary::
   :sorted:

   release
   receptor
      Where and when particles start, in either direction: at the receptor
      in a backward run, at the source in a forward run (see the
      :ref:`diagram <release-source-receptor>`). A release placed at a
      measurement and followed backward in time is what STILT calls a
      receptor, so PYSTILT writes *release (receptor)*. A
      :class:`~stilt.PointRelease` is one point (a tower inlet), a :class:`~stilt.ColumnRelease` is a vertical
      column, and a :class:`~stilt.MultiPointRelease` is a set of points,
      such as a slanted line of sight. See :doc:`../guides/releases` and
      :doc:`../guides/slant_columns`.

   particle
      An imaginary parcel of air that starts at a release and moves with the
      winds, plus random turbulence: backward in time in a backward run, the
      usual one, and forward in time in a forward run. STILT releases many
      (:term:`numpar`) and treats them together as a sample of where the air
      could have come from.

   trajectory
      The path of one particle through time. PYSTILT keeps every path of a
      simulation in one table, the simulation's particles. On disk it is a
      Parquet file.

   footprint
      A map of how much each surface grid cell influenced a measurement.
      Each value is the concentration change at the release (receptor) per unit of
      surface emission in that cell, in ppm per (µmol m⁻² s⁻¹). Multiply by
      an emissions map and sum to get a modeled concentration increase. In
      Python it is an :class:`xarray.DataArray`; on disk, a Parquet file of
      the cells the particles reached.

   meteorology
      Gridded weather-model fields (winds, temperature, turbulence, boundary
      layer height) that STILT uses to move particles. Often shortened to
      *met*. Must be in :term:`ARL` format. See :doc:`../guides/meteorology`.

   ARL
      The file format for meteorology used by HYSPLIT and STILT, named after
      NOAA's Air Resources Laboratory. NOAA publishes HRRR, NAM, GDAS, and
      other models in this format.

   met
      One named set of meteorology files in your settings, such as
      ``hrrr``, written under ``mets`` in ``config.yaml``. Each
      :term:`variant` names the met it runs on, so the same release can run
      with several mets side by side.

   defaults
      The settings written in the blocks of ``config.yaml``: the transport
      model's under ``models`` and the footprint's under ``stages``. Every
      :term:`variant` starts from them and overrides what it changes. A
      variant with no overrides runs them as they are.

   variant
      A named set of settings that every release is run under. It is a
      :term:`met` plus any settings that differ from the :term:`defaults` in
      ``config.yaml``, such as a different ``ziscale``, a wind-error run, or
      another footprint grid. See :doc:`../guides/projects`.

   realization
      One member of a variant declared with ``realizations: N``. The same
      settings run ``N`` times, as realizations ``0`` to ``N - 1``, and
      each run gets its own random draws. Their results share the
      variant's folder, a ``realization=k`` partition each.
      Realizations are used for transport-error ensembles
      (:doc:`../guides/transport_error`).

   output directory
      Where a project's results go, named by ``output:`` in ``config.yaml``
      (``./output`` by default, or a URL such as ``s3://bucket/output``). It
      holds one :term:`settings folder` per set of settings under
      ``particles/``, ``footprint/``, and ``logs/``, and several projects
      can share it. See :doc:`../guides/projects`.

   settings folder
      One folder of results in the :term:`output directory`: everything made
      with one set of settings, named after the variant that first made it
      and a short code computed from the settings, such as
      ``settings=hrrr-b2399e``. A changed setting makes a new folder and
      never overwrites one, and projects with the same settings share a
      folder. ``stilt status`` lists them.

   workdir
      The folder one simulation's transport model runs in, new for each
      run. It is removed after a successful run and kept under
      ``scratch/`` in the output directory when the run fails
      (``sim.kept_workdir``).

   compute root
      The folder the workdirs are made in, usually fast local disk: the
      ``--compute-root`` option, else ``RUNTRAJ_COMPUTE_ROOT``, else
      ``$TMPDIR/pystilt/<project>``. Projects can share one.

   HYSPLIT
      NOAA's particle transport program, written in Fortran. PYSTILT runs it
      to move particles but does not include it: you download it from NOAA
      (:doc:`installation`).

   simulation
      One release run under one :term:`variant`: one set of particles and
      at most one footprint. A project with 100 releases and 3 variants has
      300 simulations.

   release id
      The name of a release and of its result files, such as
      ``202307151800_-111.848_40.766_10``: the release time as
      ``YYYYMMDDHHMM``, then the longitude, latitude, and altitude. Column
      releases end in ``X`` and the bottom and top (``_X0-3000``) instead
      of an altitude. Multipoint releases use ``multi_`` and a short hash
      instead of coordinates. Heights above mean sea level add ``msl`` at
      the end. A :term:`simulation` is named by its release id and
      variant, ``202307151800_-111.848_40.766_10/hrrr``, and a
      :term:`realization` adds its number.

   project
      A folder holding your settings (``config.yaml``), your releases
      (``releases.csv``), and any other inputs (``tables/``, such as
      averaging kernels). The results go to the :term:`output directory` it
      names. See :doc:`../guides/projects`.

   numpar
      The number of particles released per simulation. More particles give
      smoother footprints and take longer. 200 to 1000 is typical.

   n_hours
      How many hours to follow the particles. Negative values run backward
      in time, which is the usual case for measurements. ``-24`` means one
      day back.

   AGL
      Above ground level. Release altitudes are AGL unless you set
      ``altitude_ref="msl"``.

   MSL
      Above mean sea level.

   hyper-near field
      The area right around the release, where the particle cloud hasn't
      yet spread out enough to represent mixing well. The ``hnf_plume``
      setting (on by default) corrects the footprint there with a Gaussian
      plume model.

   transform
      A step that changes how much each particle counts before the footprint
      is calculated. Used for column measurements (:term:`averaging kernel`
      and :term:`pressure weighting`) and for gases that decay in the
      atmosphere. See :doc:`../guides/transforms`.

   averaging kernel
      How sensitive a column retrieval is to each height in the atmosphere.
      Applying it weights particles released at each height accordingly.

   pressure weighting
      Weighting column particles by the share of the column's air mass they
      represent. HYSPLIT releases particles evenly in height, but air thins
      with height, so a particle high in the column stands for less air.

   backend
      Where the simulations run. The choices are ``local`` (your computer)
      and ``slurm`` (an HPC cluster). Set it in the ``execution`` section
      of ``config.yaml``, with that backend's settings in the block of its
      name. See
      :doc:`../guides/running`.

   empty footprint
      A simulation that ran fine but whose particles never touched the
      footprint grid, for example because the grid is too small or is not
      upwind. PYSTILT writes a footprint file with no cells, marked empty,
      and counts the simulation as complete.
