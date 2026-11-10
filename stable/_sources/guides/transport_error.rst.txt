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
region, as Lin and Gerbig did (:ref:`wind-error-statistics`, below). Do not copy them from
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
``realizations`` later and only the new realizations run. If the variant
also has a grid, :ref:`jacobian-per-realization` shows a Jacobian for each
realization.

Pass all the realizations to ``sim.transport_error`` as a list. It averages
them before taking the difference:

.. code-block:: python

   sim = project.simulation(rid, "hrrr")
   ensemble = [project.simulation(rid, "hrrr-err", k) for k in range(4)]
   err = sim.transport_error(ensemble, flux)
   err.realizations  # 4

More realizations help less than you might expect. They reduce the noise
of the perturbed runs, but every realization is compared with the same
unperturbed run, whose noise stays. When the two runs spread about alike,
the noise of the estimate falls by at most a factor of √2 (see `Notes`_).
Realizations are worth their cost when a single run's
``variance`` is within a factor of two of its ``noise``. When ``variance``
is far below ``noise``, check the wind-error scales instead.

The modeled enhancement
------------------------

The enhancement a footprint predicts for a surface flux field is the
footprint times the flux, summed over the grid. ``flux`` is an
:class:`xarray.DataArray` on ``lat`` and ``lon``, with an optional ``time``
dimension. It is sampled at the footprint's cell centers, and footprint
cells outside the flux field add nothing. Each footprint hour takes the
flux step whose CF time bounds hold it, if the flux has them; without
bounds, the ``time`` labels are read as the starts of the flux's steps, so
each footprint hour takes the last step that starts at or before it
(:ref:`flux-step-labels` says what to do with CF time bounds, and with
labels not at the start of a step). A flux with finer cells than the
footprint's must be put on the footprint grid first
(:doc:`../tutorials/flux_inversion` says how), and one with steps shorter
than the footprint's layers (``tres``) must be averaged onto the layers,
or the footprints made with a shorter ``tres``.

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
bound. ``sigma`` can come from the spread of several inventories. The
second assumes the cells are independent. Both values sum the footprint
over time first, so they take each cell's flux error to be the same at
every hour. The second is a lower bound only if that holds and no errors
are negatively correlated: errors that vary independently from hour to
hour, or in opposite directions in neighboring cells, can give a smaller
value. Real inventory errors are correlated over some distance, so for
errors that do not change with time and are never negatively correlated,
the true value lies between the two. Computing it needs the covariance of
the flux errors, which is the prior error covariance of an inversion. If
you set up the inversion with fips, ``InverseProblem.prior_obs_error`` is
that covariance carried to every observation (the footprint matrix times
the prior covariance times its transpose). The square root of its diagonal
is each observation's emission error. Add it in quadrature to the transport
and retrieval errors to get the total error of a modeled value.

The transport error
-------------------

``project.transport_error`` takes a selection of the unperturbed run, the
name of the wind-error variant, and the flux field:

.. code-block:: python

   sims = project.simulations
   wbb = sims[(sims.variant == "hrrr") & (sims.site == "WBB")]
   errors = project.transport_error(wbb, "hrrr-err", flux)
   errors.groupby(errors.time.dt.month).variance.mean()

``errors`` is ``wbb`` with six more columns. ``variance`` is the
transport-error variance of the modeled enhancement, in the enhancement's
units squared, and ``noise`` is its standard error
(`Is the estimate meaningful?`_).
``enhancement`` and ``enhancement_perturbed`` are the modeled enhancement
without and with the perturbation, and ``background`` is 0 unless you pass
a background field (:doc:`background`). ``realizations`` is how many
realizations of the error variant went into the estimate: every one that
has run. A receptor without particles in either variant gets NaN and
``realizations`` 0.

``errors`` has a row for every receptor in the selection. A Jacobian built
from the same selection leaves out the receptors with an empty footprint, so
match the two by ``H.receptors`` (:ref:`the rows of H <jacobian-rows>`).

Each receptor's particles are read whole. A table of 1.4 million rows
takes about a second to read, so a selection of hundreds of receptors takes
minutes. For a whole large project, make the call in a Slurm job.
``workers`` sets how many receptors are read at once (up to 8 by
default).

``sim.transport_error`` gives the same estimate for one receptor, and
keeps the per-level table described below:

.. code-block:: python

   sim = project.simulation(rid, "hrrr")
   result = sim.transport_error(project.simulation(rid, "hrrr-err"), flux)
   result.variance, result.noise

``result.sd`` is the square root of ``result.variance``. In both calls the
particles are weighted the way the unperturbed variant's footprint weights
them, with its transforms: the averaging kernel (including one from a
per-receptor table), pressure weighting, and lifetime decay.
:func:`stilt.particles.transport_error` does the same for particle tables
you have without a project.

Is the estimate meaningful?
---------------------------

``variance`` is the difference of two sample variances, each from a few
thousand particles. It is noisy and is sometimes negative. Two checks tell
you whether a value means anything:

- ``result.noise`` is the standard error of ``variance``, from the
  particles of both runs. A ``variance`` within two or three times ``noise``
  is not resolved.
- Over many receptors, take the mean of the signed ``variance``, by hour,
  season, or site. Its uncertainty is the standard deviation of the
  receptors' values divided by the square root of their number. Do not clip
  each value at zero first, because that turns noise into a positive error.
  ``result.sd`` treats a negative variance as zero, so use it only once the
  variance is resolved.

Take the mean and not the median. Most particles miss the sources and a few
cross them, so the per-particle enhancement is skewed, and so is the
estimate. The median of a skewed estimate is below its mean. In a test with
skewed enhancements at a point receptor, the median over receptors was 17
to 35 % low, and the mean was within 4 %.

Near a strong source a few particles carry most of the enhancement. One
receptor's ``variance`` then depends on those few particles and is rarely
resolved. Combine receptors.

For a column receptor the levels are added so that the noise in each level
averages out over receptors, as it does for a point receptor
(`What the numbers mean`_). Where the error changes sharply with height,
the column variance comes out high, by up to about 20 %.

The signal is largest when turbulence spreads the particles least, at night
and in winter. On a convective afternoon the unperturbed particles already
fill the boundary layer. The wind error adds little, and the estimate is
usually within its noise.

One variance per receptor
-------------------------

An inversion that uses ``variance`` as the transport part of the
model-data mismatch gets a diagonal matrix, because PYSTILT gives one
variance per receptor. That treats the transport errors of different
observations as independent, and they are not. A wind error with
``tluverr: 260`` stays correlated for about four hours, so hourly
observations at one tower share most of their transport error with their
neighbors, and nearby sites share it too. PYSTILT does not estimate that
covariance. Used as it is, the diagonal counts correlated observations as
independent information, so the posterior uncertainty tends to come out too
small. Averaging the observations over the correlation time before the
inversion, or adding off-diagonal terms from your own estimate, reduces
the effect.

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
- ``noise`` is the standard error of ``dvar``.
- ``factor`` is what the level's ``dvar`` is multiplied by in the column.
  ``result.variance`` is ``(factor * dvar).sum()``.

``result.enhancement`` is close to what the footprint gives. The two differ
a little because the footprint smooths the particles onto its grid.

A column receptor's particles are grouped into ``levels`` release-height
bins of equal width, 20 by default. A multipoint or slant receptor with no
more than ``levels`` release heights gets one level per height. The levels
are then added with an exponential error correlation in the vertical:

.. math::

   \sigma^2 = \sum_i \mathrm{factor}_i\, \mathrm{dvar}_i,
   \qquad
   \mathrm{factor}_i = w_i \sum_j w_j R_{ij}

Here ``w`` is ``weight``. ``R`` is the correlation between two levels: the
mean of :math:`e^{-|h_p - h_q| / L}` over the particles ``p`` of one level
and ``q`` of the other, where ``h`` is a particle's release height. ``L`` is
``length_scale``, 356 m by default (X-STILT's value). With
``length_scale=None`` the levels are independent, and ``factor`` is
``weight`` squared. For a point receptor ``factor`` is 1.

Written over pairs of levels, the sum is

.. math::

   \sigma^2 = \sum_{i,j} w_i w_j R_{ij}\,
              \frac{\mathrm{dvar}_i + \mathrm{dvar}_j}{2}

so a pair of levels counts with the mean of its two ``dvar``. ``R`` is
taken between the particles, not between the levels' mean heights, so
``levels`` only sets how finely ``dvar`` is resolved in height.

By default every particle counts, as in Lin and Gerbig. ``percentile=0.99``
reproduces X-STILT's trimming
(`Wu et al., 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_). It drops
the top 1 % of particles in each level before taking the variance, which
damps the few particles that pass over a point source at the cost of a
small bias. The :doc:`X-STILT migration page <migration/x_stilt>` lists
where the rest of the calculation differs.

PYSTILT uses each level's ``dvar`` as it is, with its sign, and the column
sum is linear in it. The noise in ``dvar`` therefore averages out over many
receptors, for a column as for a point receptor.

A pair of levels could instead count with the product of its two standard
deviations, :math:`\sqrt{\mathrm{dvar}_i\, \mathrm{dvar}_j}`. The two
forms agree when both levels have the same error. Where the error changes
with height within a correlation length, the mean is the larger: by 2 % for
an error that falls by a factor of e in 1000 m, 7 % in 500 m, and 20 % in
250 m. PYSTILT takes the mean because the square root of a noisy ``dvar`` is
too small on average, by the same amount at every receptor, so an average
over receptors stays too small.

Fitting a line through only the levels where ``dvar`` is positive would
bias the result upward: it keeps the levels whose variance happened to
rise, so under pure sampling noise it reports a positive error at every
level.

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

.. _wind-error-statistics:

Wind-error statistics
---------------------

A transport-error run needs four numbers that describe the meteorology's
wind errors: the standard deviation of the error, and how long, how high
and how far it stays correlated. `Lin and Gerbig (2005)
<https://doi.org/10.1029/2004GL021127>`_ derive them by comparing the
analyzed wind with observed winds, and this section does the same for your
meteorology and region. Derive your own values rather than copying them
from another analysis or place, because the correlation scales decide
whether the perturbation has any effect at all.

What the numbers are
~~~~~~~~~~~~~~~~~~~~

For each wind component, the error at an observation is the analyzed wind
minus the observed wind. Its standard deviation is ``siguverr``.

The correlation scales come from the variogram. At a separation ``h``, the
variogram is half the mean squared difference between the errors of pairs
of observations ``h`` apart. For an error with an exponential correlation
it is

.. math::

   \gamma(h) = \sigma^2 \left(1 - e^{-h/l}\right)

where σ is the standard deviation and ``l`` the correlation scale. Fitting
this curve with σ fixed gives ``l``. Pairs of heights give ``zcoruverr``,
pairs of times give ``tluverr``, and pairs of stations give
``horcoruverr``.

Each scale needs data that resolves it. Radiosonde profiles resolve height,
but launches are twelve hours apart, so they cannot measure a time scale of
a few hours. Hourly surface stations resolve time and horizontal distance,
but they only sample the bottom of the layer. The recipe below takes the
standard deviation and the vertical scale from radiosondes, and the time
and horizontal scales from surface stations.

Step 1: the errors
~~~~~~~~~~~~~~~~~~

Sample the meteorology at the observations with
`arlmet <https://github.com/jmineau/arl-met>`_. Winds in ARL files on
projected grids are stored relative to the grid, so ask for earth-relative
components (``earth_relative=True``). Radiosondes
report geopotential height, so sample in meters above sea level:

.. code-block:: python

   import arlmet
   import pandas as pd

   sondes = pd.read_csv("slc_2024_sondes.csv")   # time, lon, lat, height (m MSL), elevation, u, v
   points = sondes.rename(columns={"height": "z"})
   met = arlmet.sample_points(hrrr_files, points, ["UWND", "VWND"],
                              z_kind="msl", earth_relative=True)
   upper = pd.DataFrame({
       "time": sondes["time"],
       "height": sondes["height"] - sondes["elevation"],   # m above ground
       "u_err": met["UWND"] - sondes["u"],
       "v_err": met["VWND"] - sondes["v"],
   })

Surface stations report wind at a height above ground, 10 m for a standard
anemometer. Sample the 10 m wind, or the lowest model level:

.. code-block:: python

   # stations: time, site, lon, lat, u, v
   points = stations.assign(z=10.0)
   met = arlmet.sample_points(hrrr_files, points, ["U10M", "V10M"],
                              z_kind="agl", earth_relative=True)
   surface = pd.DataFrame({
       "time": stations["time"], "site": stations["site"],
       "lon": stations["lon"], "lat": stations["lat"],
       "u_err": met["U10M"] - stations["u"],
       "v_err": met["V10M"] - stations["v"],
   })

Reading the observations is up to you. The Salt Lake Valley numbers below
used the sonde reader in `lair <https://github.com/jmineau/lair>`_.

For radiosondes anywhere in the world, the usual source is NOAA's
Integrated Global Radiosonde Archive (IGRA2).
`siphon <https://unidata.github.io/siphon/>`_ reads it
(``pip install siphon``). IGRA2 names a WMO station by its two-letter
country code, ``M``, and the WMO number padded to eight digits. Salt Lake
City, WMO 72572, is ``USM00072572``. The full list is
``igra2-station-list.txt`` in the NCEI archive. This builds the ``sondes``
table above:

.. code-block:: python

   import datetime as dt
   from siphon.simplewebservice.igra2 import IGRAUpperAir

   levels, launches = IGRAUpperAir.request_data(
       [dt.datetime(2024, 1, 1), dt.datetime(2024, 12, 31, 23)], "USM00072572"
   )
   launches = launches.drop_duplicates("date")[["date", "latitude", "longitude"]]
   levels = levels.merge(launches, on="date")
   ground = levels["lvltyp2"] == 1                  # IGRA2's surface level
   elevation = levels["height"].where(ground).groupby(levels["date"]).transform("max")
   sondes = pd.DataFrame({
       "time": levels["date"],
       "lon": levels["longitude"], "lat": levels["latitude"],
       "height": levels["height"],                  # geopotential height, m
       "elevation": elevation,
       "u": levels["u_wind"], "v": levels["v_wind"],
   }).dropna()

Each call downloads the station's whole record and keeps the dates you ask
for, so ask for the full period once, not one launch at a time. ``time`` is
the nominal launch hour, 00 or 12 UTC. The balloon goes up as much as an
hour earlier. The last line drops levels without a height, which IGRA2 has
on some significant levels. It also drops launches with no surface level,
since their heights above ground are unknown.

Step 2: the scales
~~~~~~~~~~~~~~~~~~

:func:`~stilt.meteorology.variogram` builds the empirical variogram. It
takes the errors, the coordinate the separation is measured in, and a
``group`` label. Only points with the same label are paired. The coordinate
can also be two columns of longitude and latitude, and the separation is
then the great-circle distance in kilometers.
:func:`~stilt.meteorology.fit_variogram` fits the exponential model with
σ fixed.

.. code-block:: python

   import numpy as np
   from stilt.meteorology import fit_variogram, variogram

   def scale(errors, lag, group, bins):
       table = variogram(errors, lag, group=group, bins=bins)
       return fit_variogram(table["lag"], table["gamma"], sigma=errors.std()).length

   layer = upper[upper["height"].between(0, 3000)]        # m above ground
   minutes = (surface["time"] - pd.Timestamp("2000-01-01")) / pd.Timedelta("1min")
   components = ("u_err", "v_err")

   siguverr = np.mean([layer[c].std() for c in components])
   zcoruverr = np.mean([scale(layer[c], layer["height"], layer["time"],
                              range(0, 3100, 100)) for c in components])
   tluverr = np.mean([scale(surface[c], minutes, surface["site"],
                            range(0, 14401, 60)) for c in components])
   horcoruverr = np.mean([scale(surface[c], surface[["lon", "lat"]], surface["time"],
                                range(0, 51)) for c in components])

The recipe makes these choices, and you can change any of them:

- ``layer`` is the height range whose errors matter for your transport.
  Use the boundary layer and a little above for a surface receptor, and
  more for a column.
- The vertical variogram pairs levels within one launch (``group`` is the
  launch time), in 100 m bins up to 3 km.
- The time variogram pairs hours at one station, out to ten days.
- The horizontal variogram pairs stations at one time, out to 50 km.
- Each fit fixes σ at the standard deviation of the errors it was built
  from.
- The fit has no nugget, and it is an unweighted least squares on the bins.
  Noise in the observed winds, such as instrument error or a difference
  between what a station and a 3 km grid cell represent, adds a variance
  that stays at small separations. The model cannot hold it, so the fit
  shortens the scale instead. In a test with a true scale of 450 m, noise
  that was 20% of the variance gave a fitted scale of about 330 m, and noise
  of 40% gave about 260 m. A scale that is too short makes the perturbation
  average out (`Choosing the settings`_), so a scale fitted to noisy winds is
  likely too short. In the plot below, a variogram that is well above zero
  at its smallest lags is the sign.
- u and v are done separately and averaged.
- The mean error (the bias) is ignored, as in Lin and Gerbig. The
  perturbation does not represent it.

The four values go into ``config.yaml`` under the same names, or into
:class:`~stilt.transport.hysplit.HysplitConfig`. Look at the fits before you trust
them:

.. code-block:: python

   import matplotlib.pyplot as plt

   table = variogram(layer["u_err"], layer["height"], group=layer["time"],
                     bins=range(0, 3100, 100))
   fit = fit_variogram(table["lag"], table["gamma"], sigma=layer["u_err"].std())
   plt.scatter(table["lag"], table["gamma"], s=8)
   plt.plot(table["lag"], fit(table["lag"]))

Seasonal and diurnal values are worth a look. Filter both tables and run
the recipe again. For the Salt Lake Valley the horizontal scale ranged
from 9 km in summer to 20 km in winter.

Without surface stations, the time scale can only come from pairs of
launches at the same height. That tells you whether the errors are still
correlated twelve hours later, and little else. ``horcoruverr`` then has to
come from another source.

Salt Lake Valley, HRRR, 2024
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Radiosondes from Salt Lake City and 20 surface stations in the valley,
compared with the 3 km HRRR analysis, gave:

.. list-table::
   :header-rows: 1

   * - Component
     - σ [m/s]
     - l_z [m]
     - l_x [km]
   * - u
     - 2.37
     - 365
     - 12.1
   * - v
     - 2.87
     - 548
     - 16.5

These became the values in the above example:
``siguverr: 2.6``, ``zcoruverr: 450``, ``horcoruverr: 14`` and
``tluverr: 260``. The hourly stations gave time scales of 207 min for u
and 221 min for v, and the sondes, extrapolated, 212 to 312 min. The recipe
above gives 2.5 m/s, 480 m, 214 min and 14 km. For comparison, Lin and
Gerbig found about 120 km, 4 hours and 900 m for an 80 km analysis over the
eastern United States.

Notes
-----

- Lin and Gerbig derived their scales from variograms of analyzed minus
  radiosonde winds, and got about 120 km, 4 hours, and 900 m for an 80 km
  analysis. For a 3 km model such as HRRR the horizontal scale is much
  shorter (:ref:`wind-error-statistics`).
- With ``krand: 4`` the clock gives only about 5000 distinct seeds, so two
  realizations can be identical. With a few realizations this is very
  unlikely. With a hundred it is more likely than not.
- With ``krand: 2``, realization 0 uses ``seed`` itself, as STILT-R's error
  run does.
- ``noise`` is a standard error. The variance of ``n`` values has the
  sampling variance ``(m4 - var²) / n``, where ``m4`` is their fourth
  central moment. A level's ``noise`` adds this for the unperturbed run and
  for the perturbed run, the second divided by the number of realizations
  ``N``, and ``result.noise`` is ``sqrt(((factor * noise) ** 2).sum())``.
  For two runs that spread alike, the noise with ``N`` realizations is
  ``sqrt((1 + 1/N) / 2)`` times that of a single run, which never falls
  below ``1/sqrt(2)``. With ``percentile`` below 1 the moments are those of
  the particles kept.
