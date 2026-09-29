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
     - ``averaging_kernel`` and ``pressure_weighting`` transforms (:doc:`/advanced/transforms`)
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
   * - Transport error on the modelled column (``cal.trajfoot.stat``, ``cal.trans.err``)
     - ``error_functions/``
     - :func:`stilt.observations.transport_error` on the particles of an
       unperturbed variant and a wind-error variant
       (:doc:`/guides/transport_error`). X-STILT's separate ``outerr_`` tree
       becomes one more variant.
   * - Modelled enhancement from an inventory (``ff.trajfoot``)
     - ``error_functions/``, ``run.xco2ff.sim``
     - :meth:`stilt.Footprint.enhancement`, :func:`stilt.flux.particle_enhancement`
   * - Wind error statistics from radiosondes and surface stations
     - ``get.uverr``, ``get.siguverr``, ``cal.wind.err``, ``grab.raob``
     - :func:`~stilt.observations.variogram` and
       :func:`~stilt.observations.fit_variogram` estimate all four error
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
     - :func:`stilt.observations.background` with any xarray field
       (:doc:`/guides/background`)
   * - Emission-error propagation (``cal.emiss.err``, footprint × inventory spread)
     - ``error_functions/``
     - the same product as the enhancement, ``foot.enhancement(sigma)``. With
       a spatial correlation, use fips's ``prior_obs_error`` (see the
       emission error section of :doc:`/guides/transport_error`).
   * - Satellite-derived plume background (``compute_bg``, forward trajectories)
     - ``background/``
     - :func:`stilt.observations.plume_polygon` and
       :func:`stilt.observations.plume_background` on forward runs
       (:doc:`/guides/plume_background`)

Moving a workflow over
----------------------

1. Write a reader that turns your product into a DataFrame with one row per
   sounding (see *Adding your own instrument* in :doc:`/advanced/observations`).
2. Group the soundings by overpass and select the ones to run with the
   helpers above.
3. Build one receptor per row. For slant columns, use ``slant_points``.
4. Build the averaging-kernel table with ``averaging_kernel_table`` and save
   it in the project. Then list ``averaging_kernel`` (with ``table:``) and
   ``pressure_weighting`` under ``transforms`` in ``config.yaml``.
