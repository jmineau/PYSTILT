Background
==========

A footprint gives the enhancement: how much the fluxes inside the domain
add to a measurement. The measurement itself is that enhancement on top of
the background, the mole fraction of the air before it entered the domain.
To compare a modelled value with an observed one, or to subtract a
background from observations for an inversion, you need both.

Each back-trajectory ends where its air came from, so a mole-fraction
field sampled at every particle's endpoint and averaged over the particles
is the background the receptor saw. X-STILT does this against CarbonTracker
(``endpts.trajfoot``); PYSTILT does it with
:func:`~stilt.observations.background` against any field you give it.

The field
---------

The field is an :class:`xarray.DataArray` with ``lat`` and ``lon``
dimensions, optionally ``time``, and optionally one vertical dimension.
Name the vertical dimension after the particle column it is matched
against: ``pres`` for pressure levels in hPa or ``zagl`` for height above
ground in metres (both are default particle variables), or a column you
add yourself, such as height above sea level from ``zagl + zsfc``:

.. code-block:: python

   import xarray as xr

   cams = xr.open_dataset("cams_ch4_2023-07.nc")["ch4"]     # (time, level, lat, lon), ppb
   field = cams.rename(level="pres")                        # levels are in hPa

The field is looked up at the cell nearest each endpoint in every
dimension: the nearest level (an endpoint above the top level takes the top
level), the nearest time, and the nearest grid cell. An endpoint outside
the field's grid gets no value and is left out of the average. A global
model's ``0..360`` longitudes are handled.

Readers stay outside PYSTILT. If your model's levels vary in space
(CarbonTracker's do), sample it yourself and pass one value per particle
instead of a field; ``sim.trajectories.endpoints()`` gives the endpoints
as a table for that, and lair's ``CarbonTracker.sample`` takes it directly.

The background at a receptor
----------------------------

.. code-block:: python

   from stilt.observations import background

   config = model.config.footprints["column"]
   rows = []
   for sim_id in model.simulations.ids(footprint="column"):
       sim = model.simulations[sim_id]
       bg = background(
           sim.trajectories.data,
           field,
           transforms=config.transforms,
           context=sim.transform_context("column"),
       )
       enhancement = float(sim.get_footprint("column").enhancement(flux).sum())
       rows.append({"receptor": sim.receptor.id, "background": bg.value,
                    "enhancement": enhancement, "modelled": bg.value + enhancement})

Pass the footprint's transforms and the simulation's context, as for
:func:`~stilt.observations.transport_error`, so the background is weighted
the way the footprint is: the averaging kernel (including one from a
per-receptor table), pressure weighting, and any lifetime decay. Then the
background and the enhancement add. For a tower receptor there is nothing
to pass and the value is the plain mean over particles.

``bg.per_particle`` is the field at each endpoint and ``bg.weights`` each
particle's share of the average, both indexed by particle. The spread of
``per_particle`` says how uniform the background was over the air the
receptor sampled.

What a column's weights cover
-----------------------------

With pressure weighting the weights sum to the fraction of the column's
air mass the particles cover, ``(p_sfc - p_top) / p_sfc``, not to one.
That is the same fraction the enhancement covers, which is why the two
add, but it means ``bg.value`` is the background over the receptor's
levels only. The column above the receptor top saw no fluxes, so its
contribution is the field alone: take it from the same model with the
same pressure weights and averaging kernel, and add it in your own code.

With transport error
--------------------

Wind errors move the endpoints as well as the surface contact, so where
the background field has gradients they add to the transport error.
Give :func:`~stilt.observations.transport_error` the field and its
statistics become those of the modelled mole fraction, enhancement plus
background per particle:

.. code-block:: python

   result = transport_error(
       sim.trajectories.data, sim.error_trajectories.data, flux,
       transforms=config.transforms, context=sim.transform_context("column"),
       background=field,
   )
   result.enhancement - result.background   # the enhancement alone

Choices made here
-----------------

- Endpoints are the row farthest in time from release. A particle that
  left the domain early ends where it left, which is where the background
  should be sampled.
- Nearest neighbour in every dimension, with no interpolation. Held at the
  ends in time and in the vertical, missing outside the grid.
- A lifetime transform decays the background by the endpoint age, as it
  does the enhancement along the trajectory.
- Missing particles are left out and the rest carry their weight, as if
  the missing ones had the same mean. Check ``bg.per_particle`` when a
  regional field might not cover every endpoint.
