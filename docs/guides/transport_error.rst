Transport Error
===============

A footprint says where a measurement's air came from according to one
meteorological analysis. The analysis has errors, so the modelled
enhancement does too. PYSTILT estimates that error with the method of Lin
and Gerbig (2005): run the particles a second time with an extra random
wind component that has the statistics of the meteorology's errors, and
take the extra spread of the modelled enhancement across the ensemble as
the transport-error variance. An inversion uses the result as the transport
part of its model-data mismatch for each observation.

Run with wind errors
--------------------

Give the transport the error statistics of the meteorology in
``config.yaml`` (the same names as STILT-R):

.. code-block:: yaml

   siguverr: 2.6        # wind speed error, m/s
   tluverr: 260         # its correlation time, min
   zcoruverr: 450       # its vertical correlation length, m
   horcoruverr: 14      # its horizontal correlation length, km

Every simulation then writes a second particle table next to the main one,
``sim.error_trajectories``, whose particles saw the perturbed winds. Mixed
layer height errors (``sigzierr``, ``tlzierr``, ``horcorzierr``) are set
the same way, but HYSPLIT applies them differently: each particle's
footprint increment is multiplied by an independent random factor with that
standard deviation, and its path is unchanged. Because the factors are
independent between particles, their effect on the receptor enhancement
averages away, and their effect on the ensemble variance is small compared
with the sampling noise of a few thousand particles. PYSTILT's validation
could not resolve a 50 % mixed-layer error with 3 000 particles.
``FootprintConfig.error: true`` also rasterizes the perturbed
particles as an ``{name}_error`` footprint, which is handy for plotting but
not needed for what follows. The error run doubles the transport cost.

**The correlation scales decide whether there is anything to measure.**
HYSPLIT decorrelates the wind error both over time (``tluverr``) and over
the distance a particle travels (``horcoruverr``). At 10 m/s a particle
covers 5 km in eight minutes, so a 5 km horizontal scale makes the error
white noise that averages out along the trajectory, and the perturbed
particles spread only a percent or two more than the unperturbed ones. Lin
and Gerbig derived their scales from variograms of analysis minus radiosonde
winds and got about 120 km, 4 hours and 900 m for an 80 km analysis.
Derive yours the same way for the meteorology and region you use
(:doc:`wind_errors`); the values above were derived for HRRR over the
Salt Lake Valley. Values taken from another analysis, or guessed small to
be safe, are worse than they look: PYSTILT's validation found that scales
of a few kilometres and an hour produce no detectable perturbation at all.

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
                    "variance": result.variance, "noise": result.noise})
   errors = pd.DataFrame(rows).set_index("receptor")

``result.variance`` is the transport-error variance of the modelled
enhancement, in the enhancement's units squared, and ``result.sd`` its
square root. Pass the footprint's transforms and the simulation's context
so the error is weighted the way the footprint is: the averaging kernel
(including one from a per-receptor table), pressure weighting, and any
lifetime decay are applied to both particle tables first. For a tower
receptor there is nothing to pass.

Is the estimate meaningful?
---------------------------

``variance`` is the difference of two sample variances, each from a few
thousand particles, so it is noisy and comes out negative some of the
time. That is not a bug. Two things tell you whether a value means
anything:

- ``result.noise`` is the standard deviation of ``variance`` you would get
  with no wind error at all, estimated by splitting the unperturbed
  particles into random halves and treating one half as the perturbed run.
  A ``variance`` within two or three times ``noise`` is unresolved.
- Over many receptors, aggregate the signed ``variance`` with a median (by
  hour, season, or site) rather than clipping each value at zero; clipping
  turns noise into a positive error. ``result.sd`` clips for convenience
  and is the number to use only once the variance is resolved.

The signal is strongest when turbulence spreads the particles least: stable
nights and winter. On a convective afternoon the unperturbed particles are
already spread over the whole boundary layer, the wind perturbation adds
little, and the estimate sits at its noise floor. PYSTILT's own validation
on a Salt Lake Valley column found exactly that with 3 000 particles; a
year of tower receptors gave 12 to 15 ppb in stable conditions and 2 to
3 ppb, barely resolved, in the afternoon.

What the numbers mean
---------------------

``result.levels`` shows the calculation per release level (one row for a
point receptor):

- ``mean_orig`` / ``mean_err``: the mean per-particle enhancement without
  and with the perturbation. Their weighted sums over levels are
  ``result.enhancement`` (the same number the footprint gives) and
  ``result.enhancement_perturbed``.
- ``var_orig`` / ``var_err``: the ensemble variance of the per-particle
  enhancement, and ``dvar`` their difference: Lin and Gerbig's equation 4
  for that level.
- ``sd_trans``: the signed square root of ``dvar``.
- ``weight``: the level's share of the particles, which is its share of
  the column once the transforms are applied.

The column value combines the levels with an exponential vertical error
correlation, ``Σ w_i w_j s_i s_j exp(-|h_i - h_j| / L)`` with the signed
``dvar`` on the diagonal. ``length_scale`` is X-STILT's empirical 356 m;
``None`` treats the levels as uncorrelated, and a very large value adds
them linearly. Column particles are grouped into ``levels`` equal-width
release-height bins (20 by default); a multipoint or slant receptor uses
its own release heights.

Two options reproduce X-STILT (Wu et al., 2018) rather than Lin and Gerbig.
``percentile=0.99`` drops the top 1% of particles per level before the
variance, which tames a few particles that cross a point source at the
cost of a small bias. ``regression=True`` replaces each level's difference
with a line fitted through the levels whose difference was positive; that
selection biases the slope above one, so under pure sampling noise it
reports a positive error at every level. Both are off by default.

Two limits remain. The method measures how much the perturbed winds move
particles between flux cells, so it says little when the flux field is
uniform, and it depends on the wind statistics you gave the run. And it is
the error in transport only: emission and retrieval errors are separate
terms. The background's share of the transport error, from the wind errors
moving the trajectory endpoints, is included when you pass a background
field (:doc:`background`).
