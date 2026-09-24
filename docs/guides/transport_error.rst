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
could not resolve a 50 % mixed-layer error with 3 000 particles. For an
error shared by every particle, scale the mixed layer instead
(`Mixed-layer height`_).
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

Several realizations
~~~~~~~~~~~~~~~~~~~~

One error run is one draw of the perturbation field, and its variance
estimate carries the sampling noise of that draw. Ask for several:

.. code-block:: yaml

   error_realizations: 4

Each simulation then runs the perturbed transport that many times and
writes ``sim.error_trajectories_path(k)`` for each. Only the perturbed
runs repeat; the main run is shared. Each realization needs its own draw
of the perturbation field, and there are two ways to get one:

- ``krand: 4`` (the default). HYSPLIT seeds every run from the clock, so
  the realizations are independent but not reproducible. The clock seed
  has only about 5000 distinct values, so at ``N = 100`` there is a 60 %
  chance that two realizations are bit-identical copies; at the ``N`` of
  a few used here that chance is negligible.
- ``krand: 2`` with a ``seed``. PYSTILT runs realization ``k`` with the
  seed ``seed + k``: realization 0 shares the main run's seed, as
  STILT-R's error run does, and the others differ from it and from each
  other. A rerun reproduces every one of them bit for bit.

Any other combination would repeat the same field ``N`` times, and
PYSTILT refuses it when it reads the config. A simulation is complete
when every realization exists, and ``skip_existing`` reruns only the
realizations that are missing, so a preempted job picks up where it
stopped.

Pass the whole set to :func:`~stilt.observations.transport_error` as a
list. It averages each level's perturbed mean and variance over the
realizations before taking the difference:

.. code-block:: python

   err = transport_error(
       sim.trajectories.data,
       [t.data for t in sim.all_error_trajectories],
       flux,
   )
   err.realizations  # 4

What this buys is bounded. The perturbed side's sampling noise falls as
``1/sqrt(N)``, but the unperturbed particles are the same in every
realization, so their noise stays. The null spread of ``variance`` with
``N`` realizations is ``sqrt((1 + 1/N) / 2)`` times the single-run
``noise``, which tends to ``1/sqrt(2)``: at most a ``sqrt(2)`` tighter
estimate, never a resolved one from an unresolved one. ``noise`` already
carries the factor. Realizations earn their transport cost when a single
run's ``variance`` sits within a factor of two of its ``noise``; when it
is far below, the wind-error scales are the problem, not the sampling.

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

The emission error
------------------

The same product gives the enhancement's uncertainty from the flux field's
own uncertainty. With ``sigma`` a field of one standard deviation per cell,
in the flux's units, put it on the footprint's grid and take two limits:

.. code-block:: python

   s = sigma.reindex(lat=foot.data["lat"], lon=foot.data["lon"], method="nearest")
   err_correlated = float((foot.data * s).sum())            # every cell errs the same way
   err_independent = float(np.sqrt(((foot.data * s) ** 2).sum()))   # each cell on its own

The first is the footprint times sigma summed over the grid, what X-STILT
reports as the emission error on the column (``cal.emiss.err``, with sigma
from the spread of several inventories). It assumes one shared error
across all cells, so it is an upper bound. The second treats the cells as
independent and is a lower bound. Real inventories sit between: their
errors correlate over some distance, and the number in between needs that
covariance, which is the prior error covariance of an inversion. fips
computes it for every observation at once as
``InverseProblem.prior_obs_error``, the footprint matrix times the prior
covariance times its transpose, so an inversion setup gives the emission
error for free. Add it to the transport error and the retrieval error in
quadrature for the error budget of a modelled value.

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
  A ``variance`` within two or three times ``noise`` is unresolved. With
  several realizations, ``noise`` already includes their (bounded) gain.
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

Mixed-layer height
------------------

A real error in the mixed-layer height is shared: every particle in the
valley sees the same layer that is too shallow or too deep. ``ziscale``
represents that. It multiplies HYSPLIT's mixed-layer height by one factor
for every particle, and runs with factors above and below 1.0 show how
sensitive the enhancement is to the mixed layer.

.. code-block:: yaml

   ziscale: 0.8     # every hour of the run; a list gives one factor per hour

Three things to know before running a bracket:

- Changing ``ziscale`` in an existing project reruns nothing. A simulation
  is identified by its receptor and meteorology, not by the settings, so
  the finished ones count as complete. Run each factor as its own project
  over the same receptors.
- HYSPLIT applies ``kmix0`` (150 m by default) after the factor, so a mixed
  layer already at that floor is not lowered further. The hours above it
  still are.
- HYSPLIT holds at most 150 hourly factors. A scalar ``ziscale`` is
  repeated for every hour, so it needs ``abs(n_hours) <= 150``; a longer
  run takes a list, and hours past its end are unscaled.

How much it matters depends on the receptor. A column spans the mixed layer,
and a change in its depth mostly moves footprint around inside the column:
in PYSTILT's validation a 20 % shallower layer left a 0 to 3 km column's
enhancement within its sampling noise. A surface receptor has no such
averaging, and in a small test at a Salt Lake Valley tower a 40 % change
moved the enhancement by a few tens of percent in most cases and hardly at
all in others. Measure it for your own receptors.

The bracket is a sensitivity, not an error. Turning it into one needs how
far the meteorology's mixed-layer height is from the real one, for example
against radiosonde profiles analysed with the same bulk Richardson
definition HYSPLIT uses by default (``kmixd: 3``).
