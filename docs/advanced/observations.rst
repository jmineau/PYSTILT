Observations And Scenes
=======================

``stilt.observations`` is the science-facing layer above :class:`stilt.Receptor`
for measurements and retrievals that are not already STILT receptors: tower
records with metadata, TCCON columns, satellite soundings with viewing
geometry and averaging kernels. It ports X-STILT's observation handling as a
small set of objects and functions, each of which you can replace with your
own.

The four steps
--------------

An observation-driven run is four steps. PYSTILT provides the objects and the
built-in functions; your code provides the reader and any custom pieces.

1. **Read** your product into :class:`~stilt.observations.Observation` objects.
   Product readers live outside PYSTILT.
2. **Select** which observations to run:
   :func:`~stilt.observations.filter_observations`,
   :func:`~stilt.observations.select_observations_spatial` (X-STILT's
   near-field plus background sampling), or
   :func:`~stilt.observations.jitter_observation` for extra receptors inside
   a large pixel. Group them into a :class:`~stilt.observations.Scene` per
   overpass with :func:`~stilt.observations.group_by_overpass`.
3. **Build receptors** from each observation's coordinates with the
   :class:`~stilt.Receptor` classes directly, or with
   :func:`~stilt.observations.build_slant_receptor` for a slant path.
4. **Weight** with particle transforms (:doc:`transforms`), per footprint in
   ``config.yaml`` or per observation through ``observation.transforms``.

.. code-block:: python

   from stilt import ColumnReceptor
   from stilt.observations import (
       Observation,
       group_by_overpass,
       select_observations_spatial,
   )

   observations = [Observation(sensor="oco2", species="xco2", time=t, latitude=lat,
                               longitude=lon, value=x, uncertainty=s)
                   for t, lat, lon, x, s in my_reader(path)]

   for scene in group_by_overpass(observations):
       chosen = select_observations_spatial(
           scene.observations,
           site_longitude=-111.85, site_latitude=40.77,
           near_field_dlon=0.3, near_field_dlat=0.3,
           near_field_cols=20, near_field_rows=20,
           background_cols=10, background_rows=10,
           domain_lon_range=(-113.5, -110.5), domain_lat_range=(39.5, 42.0),
       )
       receptors = [ColumnReceptor(o.time, o.longitude, o.latitude, 0, 3000) for o in chosen]
       model.register(receptors=receptors)

   model.run()

Observation
-----------

One normalized record: ``sensor``, ``species``, ``time``, ``latitude``,
``longitude``, optional ``value`` / ``units`` / ``uncertainty``, an
``observation_id``, an ``altitude`` with its reference (the station or
surface altitude, where a slant path is anchored), and two optional geometry
records for column and satellite products:

- :class:`~stilt.observations.HorizontalGeometry` — the pixel footprint
  (corners, ellipse, or center plus resolution) used by jitter and selection.
- :class:`~stilt.observations.ViewingGeometry` — the zenith and azimuth
  angles of the line of sight, used by the slant builder.

``transforms`` holds per-observation particle transforms, typically the
retrieval's own :class:`~stilt.transforms.AveragingKernel`. Anything the core
does not model goes in ``quality`` or ``metadata``.

Scene
-----

A :class:`~stilt.observations.Scene` is a named, time-ordered group of
observations with shared metadata: one overpass, one flight leg, one day at a
tower. :func:`~stilt.observations.group_by_overpass` splits a list wherever
consecutive times differ by more than ``max_gap``, which is how X-STILT finds
overpasses; :func:`~stilt.observations.group_observations` groups by any key.
``scene.receptors(build)`` maps any observation-to-receptor callable over the
members, and ``scene.time_range`` is what you pass to
``model.simulations.ids(time_range=...)`` to find the scene's simulations
later. Scenes carry no durable state.

Adding your own instrument
--------------------------

Nothing is registered or subclassed. A new instrument is a reader, possibly a
receptor builder, and possibly a transform.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from stilt.observations import Observation, ViewingGeometry, build_slant_receptor
   from stilt.receptors import Receptor
   from stilt.transforms import AveragingKernel


   def read_my_product(path) -> list[Observation]:
       """Reader: product file -> observations. Species validation belongs here."""
       df = pd.read_parquet(path)
       return [
           Observation(
               sensor="mysat",
               species="xch4",
               time=row.time,
               latitude=row.lat,
               longitude=row.lon,
               value=row.xch4,
               uncertainty=row.xch4_sigma,
               observation_id=row.sounding_id,
               altitude=row.surface_elevation,
               altitude_ref="msl",
               viewing=ViewingGeometry(zenith_angle=row.vza, azimuth_angle=row.vaa),
               transforms=[AveragingKernel(levels=row.ak_levels, values=row.ak)],
               metadata={"orbit": row.orbit},
           )
           for row in df.itertuples()
       ]


   def my_receptor(obs: Observation) -> Receptor:
       """Builder: 20 points up the slant from the surface to 3 km above it."""
       return build_slant_receptor(obs, np.linspace(obs.altitude, obs.altitude + 3000, 20))


   pairs = []  # (observation, receptor) across every overpass
   for scene in group_by_overpass(read_my_product(path)):
       receptors = scene.receptors(my_receptor)
       model.register(receptors=receptors)
       pairs.extend(zip(scene, receptors))
   model.run()

   # apply each sounding's own averaging kernel when generating its footprint
   config = model.config.footprints["column"]
   for obs, receptor in pairs:
       sim = model.simulations[f"hrrr_{receptor.id}"]
       sim.generate_footprint("column", config, transforms=obs.transforms, write=True)

The reader is where the product's conventions live: unit conversions,
quality flags, which column holds the kernel. The builder is where geometry
decisions live. The transform is where weighting lives. Each is a plain
function or class you own; PYSTILT only fixes the shape of ``Observation``
and ``Receptor`` between them.

What this layer does not do
---------------------------

It ships no product readers, no background estimation, and no propagation of
transport error to retrieved columns. Error trajectories and
``FootprintConfig.error`` provide the transport-error building block; the
rest is application code.
