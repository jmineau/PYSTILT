Coming From X-STILT
===================

PYSTILT has the main pieces of X-STILT's column and satellite workflow:
column and slant receptors, sounding selection, averaging kernels, and
pressure weighting. In X-STILT you configure one large script. In PYSTILT
these pieces are Python functions and objects that you combine in your own
script. PYSTILT does not try to copy every X-STILT feature. The table shows
the ones that have an equivalent.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - X-STILT concept
     - X-STILT API / file
     - PYSTILT equivalent
   * - Column receptor (``minagl`` / ``maxagl``, ``agl`` levels)
     - ``get.recp.sensorv2.r``
     - :class:`stilt.ColumnReceptor` from the sounding's coordinates
   * - Slant column (``run_slant``)
     - ``get.recp.sensorv2.r``
     - :func:`stilt.observations.slant_points` + :meth:`stilt.Receptor.from_points`
   * - Sounding selection (near-field + background)
     - ``sel.obs4recpv2``
     - :func:`stilt.observations.select_observations_spatial`
   * - Jittered receptors in a pixel (``jitterTF``)
     - ``jitter.obs4recp.r``
     - :func:`stilt.observations.jitter_points`
   * - Overpass grouping
     - ``get_timestr`` / overpass search
     - :func:`stilt.observations.group_by_overpass`, then ``df.groupby("overpass")``
   * - Vertical weighting (AK × PWF)
     - ``wgt.trajec.foot*.r``
     - ``averaging_kernel`` and ``pressure_weighting`` transforms (:doc:`/guides/transforms`)
   * - Per-sounding averaging kernels (``get.wgt.funcv3``)
     - ``wgt.trajec.foot*.r``
     - ``averaging_kernel`` with ``table:`` (:func:`stilt.transforms.averaging_kernel_table`)
   * - First-order chemistry
     - ``chem_lifetime``
     - ``first_order_lifetime`` transform
   * - Column footprint outputs
     - X-STILT column products
     - standard PYSTILT footprints from column or slant receptors
   * - Product readers (OCO-2/3, TROPOMI, TCCON)
     - ``column_obs/*``
     - :func:`~stilt.observations.read_oco2`, :func:`~stilt.observations.read_tropomi_ch4`,
       :func:`~stilt.observations.read_tccon` (:doc:`/guides/readers`). For other
       products, write a reader that returns the same table.
   * - Transport error on the modeled column (``cal.trajfoot.stat``, ``cal.trans.err``)
     - ``error_functions/``
     - :meth:`sim.transport_error <stilt.Simulation.transport_error>` on the particles of an
       unperturbed variant and a wind-error variant
       (:doc:`/guides/transport_error`). X-STILT's separate ``outerr_`` tree
       becomes one more variant.
   * - Modeled enhancement from an inventory (``ff.trajfoot``)
     - ``error_functions/``, ``run.xco2ff.sim``
     - ``foot.stilt.enhancement``, ``particles.stilt.enhancement``
   * - Wind error statistics from radiosondes and surface stations
     - ``get.uverr``, ``get.siguverr``, ``cal.wind.err``, ``grab.raob``
     - :func:`~stilt.meteorology.variogram` and
       :func:`~stilt.meteorology.fit_variogram` estimate all four error
       settings. X-STILT estimates only ``siguverr`` and fixes the others.
       The met is sampled at the observations with ``arlmet.sample_points``
       instead of one HYSPLIT run per sonde, and sondes come from IGRA2
       through siphon. See :doc:`/guides/wind_errors`.
   * - Mixing-height scaling for the vertical transport error (``run_ver_err``, ``zisf``)
     - ``get.zierr``
     - ``ziscale``, one variant per factor in the same project (see the
       mixed-layer height section of :doc:`/guides/transport_error`).
       X-STILT uses one constant factor and leaves the random ``sigzierr``
       unset. PYSTILT supports both.
   * - Background from trajectory endpoints (``endpts.trajfoot``, CarbonTracker)
     - ``background/``
     - :meth:`sim.background <stilt.Simulation.background>` with any xarray field
       (:doc:`/guides/background`)
   * - Emission-error propagation (``cal.emiss.err``, footprint × inventory spread)
     - ``error_functions/``
     - the same product as the enhancement, ``foot.stilt.enhancement(sigma)``. With
       a spatial correlation, use fips's ``prior_obs_error`` (see the
       emission error section of :doc:`/guides/transport_error`).
   * - Satellite-derived plume background (``compute_bg``, forward trajectories)
     - ``background/``
     - :func:`stilt.observations.plume_polygon` and
       :func:`stilt.observations.plume_background` on forward runs
       (:doc:`/guides/plume_background`)

Runs you already have in X-STILT's ``out/by-id`` folder are read with
:func:`stilt.migration.read_stilt_r` (:doc:`stilt_r`). X-STILT weights the
footprint by the averaging kernel and pressure weighting, but saves the
particles before it does, so the particles read are unweighted.

Where the methods differ
------------------------

On the same particles, a few steps give different numbers in PYSTILT and in
X-STILT. Each row says what each one does.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Step
     - X-STILT
     - PYSTILT
   * - Transport error across release levels (``scale.dvar``)
     - Fits a weighted line of the perturbed variance against the unperturbed
       variance through the levels where the variance rose, and scales every
       level's variance with that line.
     - Uses each level's variance difference as it is, with its sign (Lin
       and Gerbig's equation 4), so the noise in it averages out over many
       receptors. ``noise`` says how large it is
       (:doc:`/guides/transport_error`).
   * - Particles kept in each level (``pct``)
     - Drops the particles above the 99th percentile of the enhancement.
     - Keeps every particle, as Lin and Gerbig do. ``percentile=0.99``
       gives X-STILT's trimming.
   * - Vertical correlation of the level errors (``Lh``)
     - 356 m, an empirical mean length scale.
     - 356 m by default (``length_scale``). ``None`` treats the levels as
       independent.
   * - Emission error (``cal.emiss.err``)
     - Reports the fully correlated sum, a footprint times a ``sigma`` from
       the spread of several inventories.
     - The same sum, and also the root-sum-square for independent cells,
       which bounds the error from below. With a spatial correlation, fips's
       ``prior_obs_error`` gives the value in between.

Moving a workflow over
----------------------

1. Write a reader that turns your product into a DataFrame with one row per
   sounding (see *Your own instrument* in :doc:`/tutorials/satellite_column`).
2. Group the soundings by overpass and select the ones to run with the
   helpers above.
3. Build one receptor per row. For slant columns, use ``slant_points``.
4. Build the averaging-kernel table with ``averaging_kernel_table`` and save
   it in the project. Then list ``averaging_kernel`` (with ``table:``) and
   ``pressure_weighting`` under ``transforms`` in ``config.yaml``.
