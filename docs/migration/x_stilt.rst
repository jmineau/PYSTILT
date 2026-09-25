Coming From X-STILT
===================

PYSTILT includes the main pieces of X-STILT's column and satellite workflow:
column and slant receptors, sounding selection, averaging kernels, and
pressure weighting. They are Python functions and objects you combine in
your own script rather than one large script to configure. Full X-STILT
parity is not a goal; the table shows what has an equivalent.

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
     - :func:`stilt.observations.group_by_overpass` → ``df.groupby("overpass")``
   * - Vertical weighting (AK × PWF)
     - ``wgt.trajec.foot*.r``
     - ``averaging_kernel`` + ``pressure_weighting`` transforms (:doc:`/advanced/transforms`)
   * - Per-sounding averaging kernels (``get.wgt.funcv3``)
     - ``wgt.trajec.foot*.r``
     - ``averaging_kernel`` with ``table:`` (:func:`stilt.transforms.averaging_kernel_table`)
   * - First-order chemistry
     - ``chem_lifetime``
     - ``first_order_lifetime`` transform
   * - Column footprint outputs
     - X-STILT column products
     - standard PYSTILT footprints from column / slant receptors
   * - Product readers (OCO-2/3, TROPOMI, TCCON)
     - ``column_obs/*``
     - :func:`~stilt.observations.read_oco2`, :func:`~stilt.observations.read_tropomi_ch4`,
       :func:`~stilt.observations.read_tccon` (:doc:`/guides/readers`); other products are your
       reader producing the same table
   * - Transport error on the modelled column (``cal.trajfoot.stat``, ``cal.trans.err``)
     - ``error_functions/``
     - :func:`stilt.observations.transport_error` on the main and error particles (:doc:`/guides/transport_error`)
   * - Modelled enhancement from an inventory (``ff.trajfoot``)
     - ``error_functions/``, ``run.xco2ff.sim``
     - :meth:`stilt.Footprint.enhancement`, :func:`stilt.flux.particle_enhancement`
   * - Wind error statistics from radiosondes and surface stations
     - ``get.uverr``, ``get.siguverr``, ``cal.wind.err``, ``grab.raob``
     - :func:`~stilt.observations.variogram` and
       :func:`~stilt.observations.fit_variogram` give all four scales
       (X-STILT derives only ``siguverr`` and prescribes the rest); sampling
       the met at observations is ``arlmet.sample_points`` rather than a
       HYSPLIT run per sonde, and sondes come from IGRA2 through siphon.
       Recipe in the Wind Error Statistics guide
   * - Mixing-height error statistics
     - ``get.zierr``
     - not ported; set ``sigzierr`` and friends yourself
   * - Background from trajectory endpoints (``endpts.trajfoot``, CarbonTracker)
     - ``background/``
     - :func:`stilt.observations.background` against any xarray field
       (:doc:`/guides/background`)
   * - Emission-error propagation (``cal.emiss.err``, footprint × inventory spread)
     - ``error_functions/``
     - the same product as the enhancement, ``foot.enhancement(sigma)``; with
       a spatial correlation it is fips's ``prior_obs_error`` (see *The
       emission error* in :doc:`/guides/transport_error`)
   * - Satellite-derived plume background (``compute_bg``, forward trajectories)
     - ``background/``
     - :func:`stilt.observations.plume_polygon` and
       :func:`stilt.observations.plume_background` on forward runs
       (:doc:`/guides/plume_background`)

Moving a workflow over
----------------------

1. write a reader that turns your product into a DataFrame with one row per
   sounding (see *Adding your own instrument* in :doc:`/advanced/observations`)
2. group by overpass and select soundings with the built-in helpers
3. build one receptor per row, from ``slant_points`` for slants
4. write the soundings' averaging kernels to a table in the project with
   ``averaging_kernel_table`` and list ``averaging_kernel`` (with ``table:``)
   and ``pressure_weighting`` in the footprint config
