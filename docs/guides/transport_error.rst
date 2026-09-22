Transport Error
===============

A footprint says where a measurement's air came from according to one
meteorological analysis. The analysis has errors, so the modelled
enhancement does too. PYSTILT estimates that error the way X-STILT does
(Wu et al., 2018): run the particles a second time with an extra random
wind component, and take the additional spread of the modelled enhancement
across the ensemble as the transport error. An inversion uses the result as
the transport part of its model-data mismatch for each observation.

Run with wind errors
--------------------

Give the transport the error statistics of the meteorology in
``config.yaml`` (the same names as STILT-R):

.. code-block:: yaml

   siguverr: 2.0        # wind speed error, m/s
   tluverr: 3600        # its correlation time, s
   zcoruverr: 500       # its vertical correlation length, m
   horcoruverr: 40      # its horizontal correlation length, km

Every simulation then writes a second particle table next to the main one,
``sim.error_trajectories``, whose particles saw the perturbed winds. Mixed
layer height errors (``sigzierr``, ``tlzierr``, ``horcorzierr``) work the
same way. ``FootprintConfig.error: true`` also rasterizes the perturbed
particles as an ``{name}_error`` footprint, which is handy for plotting but
not needed for what follows. The error run doubles the transport cost; the
wind statistics come from comparing the meteorology with radiosondes or
surface stations, which PYSTILT does not do.

The modelled enhancement
------------------------

The enhancement a footprint predicts for a surface flux field is the
footprint times the flux, summed over the grid. ``flux`` is an
:class:`xarray.DataArray` on ``lat`` / ``lon`` (with an optional ``time``
dimension); it is sampled at the footprint's cell centres, and a footprint
cell outside the flux field contributes nothing.

.. code-block:: python

   flux = xr.open_dataarray("ch4_flux.nc")          # µmol m⁻² s⁻¹ on lat/lon
   foot = model.simulations[sim_id].get_footprint("column")
   enhancement = foot.enhancement(flux)             # ppm per footprint time step
   total = float(enhancement.sum())

Units are yours: a flux in µmol m⁻² s⁻¹ times a footprint in
ppm per (µmol m⁻² s⁻¹) gives ppm.

The transport error
-------------------

:func:`~stilt.observations.transport_error` takes a simulation's two
particle tables and the flux field:

.. code-block:: python

   import pandas as pd
   from stilt.observations import transport_error

   config = model.config.footprints["column"]
   rows = []
   for sim_id in model.simulations.ids(footprint="column"):
       sim = model.simulations[sim_id]
       result = transport_error(
           sim.trajectories.data,
           sim.error_trajectories.data,
           flux,
           transforms=config.transforms,
           context=sim.transform_context("column"),
       )
       rows.append({"receptor": sim.receptor.id, "enhancement": result.enhancement,
                    "transport_sd": result.sd})
   errors = pd.DataFrame(rows).set_index("receptor")

``result.sd`` is the transport-error standard deviation of the modelled
enhancement, in the same units as the enhancement. Pass the footprint's
transforms and the simulation's context so the error is weighted the way
the footprint is: the averaging kernel (including one from a per-receptor
table), pressure weighting, and any lifetime decay are applied to both
particle tables first. For a tower receptor there is nothing to pass.

What the numbers mean
---------------------

``result.levels`` shows the calculation per release level (one row for a
point receptor):

- ``mean_orig`` / ``mean_err``: the mean per-particle enhancement without
  and with the perturbation. Their weighted sum over levels is
  ``result.enhancement``, the same number the footprint gives.
- ``var_orig`` / ``var_err``: the ensemble variance of the per-particle
  enhancement, after dropping the top 1% of particles per level
  (``percentile``), which keeps a few particles that hit a point source
  from dominating.
- ``dvar``: the raw difference. It is noisy and can be negative, so
  ``sd_trans`` comes from a weighted regression of ``var_err`` on
  ``var_orig`` across the levels with a positive difference, evaluated at
  every level (Wu et al., 2018, appendix). With only one such level the
  raw difference is used, clipped at zero.
- ``weight``: the level's share of the particles, which is its share of
  the column once the transforms are applied.

The column value combines the levels with an exponential vertical error
correlation, ``sqrt(Σ w_i w_j sd_i sd_j exp(-|h_i - h_j| / L))``.
``length_scale`` is X-STILT's empirical 356 m; ``None`` treats the levels as
uncorrelated, and a very large value adds them linearly. Column particles
are grouped into ``levels`` equal-width release-height bins (20 by
default); a multipoint or slant receptor uses its own release heights.

Two limits are worth knowing. The method measures how much the perturbed
winds move particles between flux cells, so it says little when the flux
field is uniform, and it depends on the wind statistics you gave the run. And
it is the error in transport only: emission, background and retrieval
errors are separate terms.
