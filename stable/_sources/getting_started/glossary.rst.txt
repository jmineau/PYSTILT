Glossary
========

.. glossary::
   :sorted:

   receptor
      Where and when a measurement was made, which is where STILT releases
      particles. A :class:`~stilt.PointReceptor` is one point (a tower inlet),
      a :class:`~stilt.ColumnReceptor` is a vertical column, and a
      :class:`~stilt.MultiPointReceptor` is a set of points, such as a
      slanted line of sight. See :doc:`../guides/receptors` and
      :doc:`../guides/slant_columns`.

   particle
      An imaginary parcel of air released at the receptor and moved backward
      in time by the winds, plus random turbulence. STILT releases many
      (:term:`numpar`) and treats them together as a sample of where the air
      could have come from.

   trajectory
      The path of one particle through time. PYSTILT stores all of a
      simulation's particle paths together as a :class:`~stilt.Trajectories`
      table (a Parquet file on disk).

   footprint
      A map of how much each surface grid cell influenced a measurement.
      Each value is the concentration change at the receptor per unit of
      surface emission in that cell, in ppm per (µmol m⁻² s⁻¹). Multiply by
      an emissions map and sum to get a modeled concentration increase. Stored as a
      :class:`~stilt.Footprint` (a NetCDF file on disk).

   meteorology
      Gridded weather-model fields (winds, temperature, turbulence, boundary
      layer height) that STILT uses to move particles. Often shortened to
      *met*. Must be in :term:`ARL` format. See :doc:`../guides/meteorology`.

   ARL
      The file format for meteorology used by HYSPLIT and STILT, named after
      NOAA's Air Resources Laboratory. NOAA publishes HRRR, NAM, GDAS, and
      other models in this format.

   met name
      The name you give a meteorology source in your settings, such as
      ``hrrr``. With no :term:`variants <variant>` declared, each met source
      is a variant of the same name, so the same receptor can be run with
      several meteorology products side by side.

   defaults
      The settings written at the top level of ``config.yaml``. Every
      :term:`variant` starts from them and overrides what it changes. A
      variant with no overrides runs them as they are.

   variant
      A named set of settings that every receptor is run under. It is a met
      source plus any settings that differ from the :term:`defaults` in
      ``config.yaml``, such as a different ``ziscale``, a wind-error run, or
      another footprint grid. See :doc:`../guides/configuration`.

   realization
      One member of a variant declared with ``realizations: N``. The same
      settings run ``N`` times, named ``<variant>-0`` to
      ``<variant>-(N-1)``, and each run gets its own random draws.
      Realizations are used for transport-error ensembles
      (:doc:`../guides/transport_error`).

   derived variant
      A variant declared with ``from: <other>``. It makes another footprint
      (a different grid, smoothing, or particle weighting) from that
      variant's stored particles without running HYSPLIT again.

   HYSPLIT
      NOAA's particle transport program, written in Fortran. PYSTILT runs it
      to move particles and includes a copy for Linux and macOS (Intel).

   simulation
      One receptor run under one :term:`variant`: one set of trajectories and
      at most one footprint. A project with 100 receptors and 3 variants has
      300 simulations.

   simulation ID
      The name of a simulation and of its output folder, such as
      ``202307151800_-111.848_40.766_10/hrrr``. It is the receptor id, a
      slash, and the variant name. The receptor id is the receptor time as
      ``YYYYMMDDHHMM``, then the longitude, latitude, and altitude. Column
      receptors end in ``_X`` instead of an altitude. Multipoint receptors
      use ``multi_`` and a short hash instead of coordinates.

   project
      A folder holding your settings (``config.yaml``), your receptors
      (``receptors.csv``), and all outputs (``simulations/``). See
      :doc:`../guides/project_layout`.

   numpar
      The number of particles released per simulation. More particles give
      smoother footprints and take longer. 200 to 1000 is typical.

   n_hours
      How many hours to follow the particles. Negative values run backward
      in time, which is the usual case for measurements. ``-24`` means one
      day back.

   AGL
      Above ground level. Receptor altitudes are AGL unless you set
      ``altitude_ref="msl"``.

   MSL
      Above mean sea level.

   hyper-near field
      The area right around the receptor, where the particle cloud hasn't
      yet spread out enough to represent mixing well. The ``hnf_plume``
      setting (on by default) corrects the footprint there with a Gaussian
      plume model.

   transform
      A step that changes how much each particle counts before the footprint
      is calculated. Used for column measurements (:term:`averaging kernel`
      and :term:`pressure weighting`) and for gases that decay in the
      atmosphere. See :doc:`../advanced/transforms`.

   averaging kernel
      How sensitive a column retrieval is to each height in the atmosphere.
      Applying it weights particles released at each height accordingly.

   pressure weighting
      Weighting column particles by the share of the column's air mass they
      represent. HYSPLIT releases particles evenly in height, but air thins
      with height, so a particle high in the column stands for less air.

   backend
      Where the simulations run. The choices are ``local`` (your computer),
      ``slurm`` (an HPC cluster), and ``kubernetes`` (cloud, experimental).
      Set it in the ``execution`` section of ``config.yaml``. See
      :doc:`../guides/execution/index`.

   empty footprint
      A simulation that ran fine but whose particles never touched the
      footprint grid, for example because the grid is too small or is not
      upwind. PYSTILT records it with a small ``.empty`` file instead of a
      NetCDF, and counts the simulation as finished.
