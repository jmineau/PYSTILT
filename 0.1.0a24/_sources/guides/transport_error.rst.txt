Transport Error
===============

A footprint says where a measurement's air came from, according to one
meteorological analysis. The analysis has wind errors, so the modeled
enhancement has an error too. PYSTILT estimates it with the method of
`Lin and Gerbig (2005) <https://doi.org/10.1029/2004GL021127>`_. You run
the particles a second time with a random wind error added, one that has
the statistics of the meteorology's errors. The perturbed particles spread
further. The extra spread of the modeled enhancement across the particles
is the transport-error variance. An inversion uses it as the transport part
of each observation's model-data mismatch.

Run with wind errors
--------------------

The perturbed run is a :doc:`variant <projects>` of its own. It uses
the same receptors and meteorology, with four wind-error settings added.
The names are the same as in STILT-R.

.. code-block:: yaml

   variants:
     hrrr: {}                 # the unperturbed run
     hrrr-err:                # the same run with perturbed winds
       siguverr: 2.6          # wind error standard deviation, m/s
       tluverr: 260           # its correlation time, min
       zcoruverr: 450         # its vertical correlation length, m
       horcoruverr: 14        # its horizontal correlation length, km
       grid: null             # particles only

Every receptor then has a second simulation,
``project.simulation(rid, "hrrr-err")``, whose particles saw the perturbed
winds. You can add the error variant to a finished project, and only the
new simulations run. The error run costs as much as the original.
``grid: null`` skips the footprint, which the error calculation does not
need. Keep a grid if you want to plot a perturbed footprint.

Choosing the settings
~~~~~~~~~~~~~~~~~~~~~

The correlation scales matter as much as the standard deviation. HYSPLIT
decorrelates the wind error over time (``tluverr``) and over the distance
a particle travels (``horcoruverr``). If the scales are too short, the
error turns into noise that averages out along each trajectory. The
perturbed particles then spread no further than the unperturbed ones, and
there is nothing to measure. At 10 m/s, for example, a particle crosses a
5 km correlation length in eight minutes.

Derive the four values from wind observations for your meteorology and
region, as Lin and Gerbig did (:doc:`wind_errors`). Do not copy them from
another analysis, and do not pick small scales to be safe. The values above
are for HRRR over the Salt Lake Valley.

HYSPLIT also has mixed-layer height errors (``sigzierr``, ``tlzierr``,
``horcorzierr``), set the same way. They have little effect. HYSPLIT
multiplies each particle's footprint by its own random factor and leaves
the path alone. Because the factors are independent between particles,
they average out over a few thousand particles. For a mixed-layer error
that every particle shares, use ``ziscale`` instead
(`Mixed-layer height`_).

Several realizations
~~~~~~~~~~~~~~~~~~~~

One error run is one random draw of the wind error. To average over
several draws, set ``realizations`` on the error variant:

.. code-block:: yaml

   variants:
     hrrr-err:
       siguverr: 2.6
       # ...
       realizations: 4        # realizations 0 to 3

The variant then runs four times per receptor, as realizations ``0`` to
``3`` of ``hrrr-err``: the ``realization`` column of
``project.simulations``. The unperturbed ``hrrr`` run is not repeated.
Each realization needs a different random draw, and there are two ways to
get one:

- ``krand: 4``, the default. HYSPLIT seeds each run from the clock. The
  realizations differ, but a rerun gives different ones.
- ``krand: 2`` with a ``seed``. Realization ``k`` runs with ``seed + k``,
  and a rerun reproduces every realization exactly.

PYSTILT rejects any other setting when it reads the config, because every
realization would get the same draw. Each realization is its own
simulation, so a rerun only does the missing ones. You can raise
``realizations`` later and only the new realizations run.

Pass all the realizations to ``sim.transport_error`` as a list. It averages
them before taking the difference:

.. code-block:: python

   sim = project.simulation(rid, "hrrr")
   ensemble = [project.simulation(rid, "hrrr-err", k) for k in range(4)]
   err = sim.transport_error(ensemble, flux)
   err.realizations  # 4

More realizations help less than you might expect. They reduce the noise
of the perturbed runs, but every realization is compared with the same
unperturbed run, so the noise of the estimate falls by at most a factor of
√2 (see `Notes`_). Realizations are worth their cost when a single run's
``variance`` is within a factor of two of its ``noise``. When ``variance``
is far below ``noise``, check the wind-error scales instead.

The modeled enhancement
------------------------

The enhancement a footprint predicts for a surface flux field is the
footprint times the flux, summed over the grid. ``flux`` is an
:class:`xarray.DataArray` on ``lat`` and ``lon``, with an optional ``time``
dimension. It is sampled at the footprint's cell centers, and footprint
cells outside the flux field add nothing. A flux with finer cells than the
footprint's must be put on the footprint grid first
(:doc:`../tutorials/flux_inversion` says how).

.. code-block:: python

   flux = xr.open_dataarray("ch4_flux.nc")          # µmol m⁻² s⁻¹ on lat/lon
   foot = project.simulation(rid, "hrrr").footprint
   enhancement = foot.stilt.enhancement(flux)             # ppm per footprint time step
   total = float(enhancement.sum())

A flux in µmol m⁻² s⁻¹ times a footprint in ppm per (µmol m⁻² s⁻¹) gives
ppm.

The emission error
------------------

The flux field's own uncertainty gives another error on the enhancement.
Take ``sigma``, a field of one standard deviation per cell in the flux's
units. Put it on the footprint grid and compute two limits:

.. code-block:: python

   import numpy as np

   f = foot.sum("time")
   s = sigma.reindex(lat=f["lat"], lon=f["lon"], method="nearest")
   err_correlated = float((f * s).sum())                   # every cell errs the same way
   err_independent = float(np.sqrt(((f * s) ** 2).sum()))  # each cell on its own

The first assumes the errors in all cells move together, so it is an upper
bound. X-STILT reports this one as the emission error (``cal.emiss.err``),
with ``sigma`` from the spread of several inventories. The second assumes
the cells are independent, which gives a lower bound. Real inventory errors
are correlated over some distance, so the true value lies in between.
Computing it needs the covariance of the flux errors, which is the prior
error covariance of an inversion. If you set up the inversion with fips,
``InverseProblem.prior_obs_error`` is that covariance carried to every
observation (the footprint matrix times the prior covariance times its
transpose). The square root of its diagonal is each observation's emission
error. Add it in quadrature to the transport and retrieval errors to get
the total error of a modeled value.

The transport error
-------------------

``sim.transport_error`` takes the same receptor under the wind-error
variant and the flux field:

.. code-block:: python

   import pandas as pd

   rows = []
   for rid in project.receptors.receptor:
       sim = project.simulation(rid, "hrrr")
       err = project.simulation(rid, "hrrr-err")
       result = sim.transport_error(err, flux)
       rows.append({"receptor": sim.receptor.id, "enhancement": result.enhancement,
                    "variance": result.variance, "noise": result.noise})
   errors = pd.DataFrame(rows).set_index("receptor")

``result.variance`` is the transport-error variance of the modeled
enhancement, in the enhancement's units squared. ``result.sd`` is its
square root. The particles are weighted the way the footprint weights
them, with the variant's transforms: the averaging kernel (including one
from a per-receptor table), pressure weighting, and lifetime decay.
:func:`stilt.particles.transport_error` does the same for particle tables
you have without a project.

Is the estimate meaningful?
---------------------------

``variance`` is the difference of two sample variances, each from a few
thousand particles. It is noisy and is sometimes negative. Two checks tell
you whether a value means anything:

- ``result.noise`` is the standard deviation ``variance`` would have with no
  wind error at all. A ``variance`` within two or three times ``noise`` is
  not resolved.
- Over many receptors, take the median of the signed ``variance``, by hour,
  season, or site. Do not clip each value at zero first, because that turns
  noise into a positive error. ``result.sd`` treats a negative variance as
  zero, so use it only once the variance is resolved.

The signal is largest when turbulence spreads the particles least, at night
and in winter. On a convective afternoon the unperturbed particles already
fill the boundary layer. The wind error adds little, and the estimate is
usually within its noise.

What the numbers mean
---------------------

``result.levels`` shows the calculation for each release level. A point
receptor has one level.

- ``height`` is the level's mean release height in meters, and ``n`` its
  number of particles.
- ``weight`` is the level's share of the particles.
- ``mean_orig`` and ``mean_err`` are the mean enhancement per particle
  without and with the perturbation. Their weighted sums are
  ``result.enhancement`` and ``result.enhancement_perturbed``.
- ``var_orig`` and ``var_err`` are the variances of the per-particle
  enhancement, and ``dvar`` is their difference. This is Lin and Gerbig's
  equation 4 for the level.
- ``sd_trans`` is the square root of ``dvar``, keeping its sign.

``result.enhancement`` is close to what the footprint gives. The two differ
a little because the footprint smooths the particles onto its grid.

A column receptor's particles are grouped into ``levels`` release-height
bins of equal width, 20 by default. A multipoint or slant receptor with no
more than ``levels`` release heights gets one level per height. The levels
are then combined with an exponential error correlation in the vertical:

.. math::

   \sigma^2 = \sum_i w_i^2\, \mathrm{dvar}_i
            + \sum_{i \ne j} w_i w_j s_i s_j\, e^{-|h_i - h_j| / L}

Here ``w`` is ``weight``, ``s`` is ``sd_trans``, ``h`` is ``height``, and
``L`` is ``length_scale``, 356 m by default (X-STILT's value). With
``length_scale=None`` the levels are independent.

``percentile=0.99`` reproduces X-STILT's trimming
(`Wu et al., 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_). It drops
the top 1 % of particles in each level before taking the variance, which
damps the few particles that pass over a point source at the cost of a
small bias. X-STILT also fits a line through the levels where ``dvar`` is
positive. PYSTILT does not, because fitting only those levels biases the
result upward: under pure sampling noise it reports a positive error at
every level.

The method measures how much the wind error moves particles between flux
cells. It says little when the flux field is uniform, and it is only as
good as the wind statistics you gave the run. It covers transport only.
Emission and retrieval errors are separate terms. Wind errors also move
the trajectory endpoints, and with them the background. Pass a background
field to include that part (:doc:`background`).

Mixed-layer height
------------------

A real error in the mixed-layer height is shared. Every particle in the
valley sees the same layer, too shallow or too deep. ``ziscale`` models
this. It multiplies HYSPLIT's mixed-layer height by one factor for every
particle. Runs with factors above and below 1.0 show how sensitive the
enhancement is to the mixed layer. Declare them as variants of the same
project:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-zi06: {ziscale: 0.6}   # every hour of the run; a list gives one factor per hour
     hrrr-zi14: {ziscale: 1.4}

Before you run them:

- Set ``krand: 2`` and a ``seed`` in the defaults. Every variant then draws
  the same turbulence, and the difference between variants is the mixed
  layer alone. Without a seed each run adds its own sampling noise.
- HYSPLIT applies the minimum mixing depth ``kmix0`` (150 m by default)
  after the factor. A mixed layer already at that floor is not lowered
  further.
- HYSPLIT holds at most 150 hourly factors. A single ``ziscale`` value is
  repeated for every hour, so it needs ``abs(n_hours) <= 150``. For a
  longer run, give a list. Hours past the end of the list are unscaled.

How much it matters depends on the receptor. A column spans the mixed
layer, so a change in its depth mostly moves influence around inside the
column, and the column enhancement changes little. A surface receptor has
no such averaging and can change by tens of percent. Test it for your own
receptors.

The spread between these runs is a sensitivity. To turn it into an error
you need to know how far the meteorology's mixed-layer height is from the
real one. One way is to compare it with radiosonde profiles, analyzed with
the bulk Richardson method HYSPLIT uses by default (``kmixd: 3``).

Notes
-----

- Lin and Gerbig derived their scales from variograms of analyzed minus
  radiosonde winds, and got about 120 km, 4 hours, and 900 m for an 80 km
  analysis. For a 3 km model such as HRRR the horizontal scale is much
  shorter (:doc:`wind_errors`).
- With ``krand: 4`` the clock gives only about 5000 distinct seeds, so two
  realizations can be identical. With a few realizations this is very
  unlikely. With a hundred it is more likely than not.
- With ``krand: 2``, realization 0 uses ``seed`` itself, as STILT-R's error
  run does.
- ``noise`` comes from splitting the unperturbed particles into random
  halves and treating one half as the perturbed run
  (``noise_splits`` times, 16 by default). With ``N`` realizations the
  noise is ``sqrt((1 + 1/N) / 2)`` times that of a single run, which never
  falls below ``1/sqrt(2)``. ``result.noise`` already includes this factor.
