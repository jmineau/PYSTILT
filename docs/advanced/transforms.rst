Particle Weighting (Transforms)
===============================

A footprint is the mean surface influence of a simulation's particles. A
*transform* changes how much each particle counts before that mean is taken,
by rescaling the particle table's ``foot`` column. Column instruments need
this (an averaging kernel and a pressure weighting), and so does any species
that decays during transport.

A transform is any object with one method::

   def apply(self, particles: pd.DataFrame, context: TransformContext) -> pd.DataFrame

It receives the particle table and returns a new one. Transforms are listed
per footprint in ``config.yaml`` or passed to
:meth:`stilt.Simulation.generate_footprint`, run once in order on the
unweighted particles, and are recorded in the footprint's netCDF so a stored
footprint knows how it was weighted.

Built-in transforms
-------------------

.. code-block:: yaml

   footprints:
     column:
       grid: slv
       transforms:
         - kind: averaging_kernel
           levels: [0, 500, 1000, 2000]
           values: [1.0, 0.95, 0.8, 0.5]
         - kind: pressure_weighting
         - kind: first_order_lifetime
           lifetime_hours: 4.0

``averaging_kernel`` — :class:`stilt.transforms.AveragingKernel`
   Multiplies each particle's ``foot`` by the kernel interpolated at the
   particle's release coordinate. ``levels`` are release heights AGL in
   metres by default; set ``coordinate: pres`` when the kernel is on
   pressure levels (hPa). Fold instrument-specific factors (for example
   TCCON's wet-air scaling) into ``values``. Adds an ``ak_weight`` column.

``pressure_weighting`` — :class:`stilt.transforms.PressureWeighting`
   Weights each particle by the fraction of the column's air mass it
   represents. A column instrument averages over air mass, but HYSPLIT
   releases column particles uniformly in *height*, so a plain particle mean
   over-weights the thin upper layers. Following X-STILT, the pressure
   weighting function is derived from the particles themselves: a
   hypsometric curve is fit to their first-step heights and pressures and
   evaluated at each release height, and each particle carries the slab of
   air centred on it. Nothing needs to be supplied; pass
   ``surface_pressure`` (hPa) to reference the profile to the retrieval's
   surface pressure instead of the fitted value. Requires ``pres`` and
   ``zagl`` in ``varsiwant`` (both are defaults). Adds ``xpres`` (release
   pressure) and ``pwf`` columns.

   Two things are worth knowing about the result. The weights sum to the
   fraction of the atmosphere's mass the column covers (about 0.3 for a
   0–3 km column), not to one: air above the column top cannot be reached
   by surface fluxes within the back-trajectory, so the footprint is
   complete and the remaining fraction belongs to the prior profile. And
   the weighted footprint's magnitude does not depend on ``numpar``. A
   column whose bottom sits above the ground still measures the air beneath
   it, so the lowest particle carries that whole sub-column; start the
   receptor at or near the surface unless you intend that.

``first_order_lifetime`` — :class:`stilt.transforms.FirstOrderLifetime`
   Decays ``foot`` by ``exp(-age / lifetime)``, where ``age`` is the
   particle's transport time from ``time_column`` (minutes by default).

The usual satellite or TCCON column weighting is ``averaging_kernel``
followed by ``pressure_weighting``. Listing nothing leaves every particle
with equal weight, which is standard STILT behaviour.

In Python
---------

The same classes work directly on a simulation:

.. code-block:: python

   from stilt.transforms import AveragingKernel, PressureWeighting

   foot = sim.generate_footprint(
       "column",
       config,
       transforms=[AveragingKernel(levels=ak.z, values=ak.values), PressureWeighting()],
   )

``transforms=`` runs after ``config.transforms``. To inspect what a transform
did, apply it to the particle table yourself:

.. code-block:: python

   weighted = PressureWeighting().apply(sim.trajectories.data)
   weighted.drop_duplicates("indx")[["xhgt", "xpres", "pwf"]]

Writing your own transform
--------------------------

Write a pydantic model with the fields you want in YAML and an ``apply``
method. Return a new frame; never modify the one you are given.

.. code-block:: python

   # mypkg/transforms.py
   import numpy as np
   import pandas as pd
   from pydantic import BaseModel

   from stilt import TransformContext
   from stilt.transforms import release_coordinate


   class BoundaryLayerOnly(BaseModel):
       """Drop influence from particles released above ``max_height`` m AGL."""

       max_height: float = 1500.0

       def apply(self, particles: pd.DataFrame, context: TransformContext) -> pd.DataFrame:
           z = release_coordinate(particles, "xhgt")           # one value per particle
           keep = z.reindex(particles["indx"].to_numpy()) <= self.max_height
           out = particles.copy()
           out["foot"] = np.where(keep.to_numpy(), out["foot"], 0.0)
           return out

Reference it from ``config.yaml`` by its import path; the remaining keys are
the model's fields:

.. code-block:: yaml

   footprints:
     bl:
       grid: slv
       transforms:
         - kind: mypkg.transforms.BoundaryLayerOnly
           max_height: 1200

Three rules keep this predictable:

- **Return a copy.** ``Trajectories.data`` must still be the unweighted table
  after a footprint is generated. The built-ins all ``particles.copy()`` first.
- **Be importable wherever it runs.** Workers rebuild the model from
  ``config.yaml`` alone, so ``mypkg`` must be installed on the Slurm node or
  in the container. A footprint whose transform cannot be imported can still
  be *read* (the entry becomes an :class:`~stilt.transforms.UnresolvedTransform`),
  but a project config naming one fails validation with the import error.
- **Use the context for anything outside the table.** ``context.receptor``
  gives the release time and location, ``context.is_error`` says whether
  this is the error trajectory, and ``context.observation`` carries the
  observation when the caller attached one. The built-ins ignore it.

A plain class works too (``kind`` imports it and calls ``cls(**keys)``), but
it cannot be written back to config or into a footprint's netCDF, so prefer
pydantic for anything declared in YAML. Any object with ``apply`` can be
passed through the Python ``transforms=`` argument.

The science helpers the built-ins are made of are public in
:mod:`stilt.transforms`: :func:`~stilt.transforms.release_coordinate`,
:func:`~stilt.transforms.particle_pwf` and :func:`~stilt.transforms.ak_weights`.
Composing them is usually shorter than re-deriving the weighting.
