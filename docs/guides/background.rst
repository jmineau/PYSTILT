Background
==========

A footprint gives the enhancement, the amount the fluxes inside the domain
add to a measurement. The measurement is that enhancement on top of the
background, the mole fraction of the air before it entered the domain. To
compare a modelled value with an observed one, or to subtract the
background from observations for an inversion, you need both.

Each back-trajectory ends where its air came from. Sample a mole-fraction
field at every particle's endpoint, average over the particles, and you
have the background the receptor saw. :func:`~stilt.observations.background`
does this with any field you give it. X-STILT does the same with
CarbonTracker in ``endpts.trajfoot``. For a satellite swath you can also
take the background from the soundings that a city's plume did not reach
(:doc:`plume_background`).

The field
---------

The field is an :class:`xarray.DataArray` with ``lat`` and ``lon``
dimensions, and optionally ``time`` and one vertical dimension. Name the
vertical dimension after the particle column it should be matched against.
Use ``pres`` for pressure levels in hPa or ``zagl`` for height above ground
in metres. Both are in the particle table by default. You can also add a
column of your own, such as height above sea level from ``zagl + zsfc``
(add ``zsfc`` to ``varsiwant`` to get it).

.. code-block:: python

   import xarray as xr

   cams = xr.open_dataset("cams_ch4_2023-07.nc")["ch4"]     # (time, level, lat, lon), ppb
   field = cams.rename(level="pres")                        # levels are in hPa

For each endpoint PYSTILT uses the nearest level, the nearest time, and the
nearest grid cell. An endpoint above the top level takes the top level. An
endpoint outside the field's grid gets no value and is left out of the
average. Longitudes from 0 to 360, as in many global models, are handled.

PYSTILT does not read model files. If your model's levels vary in space, as
CarbonTracker's do, sample it yourself and pass one value per particle
instead of a field, as a Series indexed by particle number (``particle``).
``sim.particles.stilt.endpoints()`` gives the endpoints as a table for this,
and lair's ``CarbonTracker.sample`` takes that table directly.

The background at a receptor
----------------------------

.. code-block:: python

   from stilt.observations import background

   rows = []
   for rid in project.receptors.receptor:
       sim = project.simulation(rid, "hrrr")
       bg = background(
           sim.particles,
           field,
           transforms=sim.variant.footprint.transforms,
           receptor=sim.receptor,
           directory=project.directory,
       )
       enhancement = float(sim.footprint.stilt.enhancement(flux).sum())
       rows.append({"receptor": sim.receptor.id, "background": bg.value,
                    "enhancement": enhancement, "modelled": bg.value + enhancement})

Pass the footprint's transforms, the receptor, and the project directory, as for
:func:`~stilt.observations.transport_error`. The background is then
weighted the way the footprint is, with the averaging kernel (including one
from a per-receptor table), pressure weighting, and any lifetime decay, and
the background and the enhancement add. For a tower receptor there is
nothing to pass, and the value is the plain mean over the particles.

``bg.per_particle`` is the field at each endpoint, and ``bg.weights`` is
each particle's share of the average. Both are indexed by particle. The
spread of ``per_particle`` shows how uniform the background was over the
air the receptor sampled.

What a column's weights cover
-----------------------------

With pressure weighting, the weights sum to the fraction of the column's
air mass that the particles cover, ``(p_sfc - p_top) / p_sfc``. The
enhancement covers the same fraction, which is why the two add. It also
means ``bg.value`` is the background over the receptor's levels only. The
air above the receptor top saw no fluxes, so its share is the field alone.
Take it from the same model with the same pressure weights and averaging
kernel, and add it in your own code.

With transport error
--------------------

Wind errors move the endpoints as well as the path near the surface. Where
the background field has gradients, this adds to the transport error. Pass
the field to :func:`~stilt.observations.transport_error`, and it works with
each particle's modelled mole fraction, enhancement plus background:

.. code-block:: python

   err = project.simulation(sim.receptor.id, "hrrr-err")
   result = transport_error(
       sim.particles, err.particles, flux,
       transforms=sim.variant.footprint.transforms,
       receptor=sim.receptor, directory=project.directory,
       background=field,
   )
   result.enhancement - result.background   # the enhancement alone

Choices made here
-----------------

- The endpoint is each particle's row farthest in time from its release.
  A particle that left the domain early ends where it left, which is where
  the background should be sampled.
- Values come from the nearest cell in every dimension, without
  interpolation. Before the first or after the last time or level, the end
  value is used. Outside the grid there is no value.
- A lifetime transform decays the background by the endpoint's age, as it
  decays the enhancement along the trajectory.
- Particles without a value are left out, and the others carry their
  weight. This assumes the missing particles have the same mean as the
  rest. Check ``bg.per_particle`` when a regional field might not cover
  every endpoint.
