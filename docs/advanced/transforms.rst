Particle Weighting (Transforms)
===============================

A footprint is the mean surface influence of a simulation's particles. A
*transform* changes how much each particle counts in that mean, by scaling
the ``foot`` column of the particle table. Column instruments need an
averaging kernel and a pressure weighting. A species that decays during
transport needs a lifetime.

A transform is any object with an ``apply`` method::

   def apply(self, particles: pd.DataFrame, context: TransformContext) -> pd.DataFrame

It takes the particle table and returns a new one. List transforms in
``config.yaml``, as a default or per variant, or pass them to
:meth:`stilt.Simulation.generate_footprint`. They run once, in order,
starting from the unweighted particles. The transforms listed in the
footprint config are recorded in the footprint's netCDF file.

Built-in transforms
-------------------

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       levels: [0, 500, 1000, 2000]
       values: [1.0, 0.95, 0.8, 0.5]
     - kind: pressure_weighting
     - kind: first_order_lifetime
       lifetime_hours: 4.0

``averaging_kernel`` (:class:`stilt.transforms.AveragingKernel`)
   Multiplies each particle's ``foot`` by the kernel, interpolated to the
   particle's release height ``xhgt``. The ``levels`` are in metres in the
   receptor's vertical reference: above ground, or above sea level for a
   receptor built with ``altitude_ref="msl"``. For a kernel on pressure
   levels (hPa), set ``coordinate: pres``. Outside ``levels`` the kernel
   keeps its end values. Fold any instrument-specific factor, such as
   TCCON's wet-air scaling, into ``values``. Adds an ``ak_weight`` column.

``pressure_weighting`` (:class:`stilt.transforms.PressureWeighting`)
   Weights each particle by the share of the column's air mass it stands
   for. A column instrument averages over air mass. HYSPLIT releases
   column particles evenly in height, so an unweighted mean gives too much
   weight to the thin air near the column top.

   The weights come from the particles themselves, as in X-STILT. PYSTILT
   fits a hypsometric curve, :math:`\ln p = b + a z`, to the particles'
   heights and pressures at their first output step. It evaluates the curve
   at each release height to get the release pressure. Each release height
   then stands for the slab of air centred on it, with the ground closing
   the lowest slab. X-STILT gives each particle the layer below it instead,
   which leaves a particle released at the surface almost no weight.

   A multipoint receptor releases several particles from each point. They
   share one release height, so the point's slab is split evenly among
   them. The weighted footprint does not depend on how many particles each
   point released.

   You do not need to supply anything. Pass ``surface_pressure`` (hPa) to
   use the retrieval's surface pressure in place of the fitted one. The
   transform needs ``pres`` and ``zagl`` in ``varsiwant``, and both are
   there by default. A receptor with ``altitude_ref="msl"`` also needs
   ``zsfc``, the terrain height, so the fit can be made against height
   above sea level. It adds ``xpres`` (release pressure, hPa) and ``pwf``
   columns.

   The weights add up to the fraction of the atmosphere's mass the column
   covers, about 0.3 for a 0 to 3 km column. The rest of the atmosphere is
   above the column top. Surface fluxes do not reach it within the
   back-trajectory, and it belongs with the prior profile. The weighted
   footprint does not change size with ``numpar``.

   The lowest particle's slab reaches down to the surface. A column that
   starts above the ground still measures the air below it, so that
   particle carries the whole sub-column. Start the receptor at or near
   the surface unless that is what you want.

``first_order_lifetime`` (:class:`stilt.transforms.FirstOrderLifetime`)
   Scales ``foot`` by :math:`\exp(-\text{age} / \tau)`, where
   :math:`\tau` is ``lifetime_hours``. The age is the particle's transport
   time, from its ``time`` column.

For satellite and TCCON columns, list ``averaging_kernel`` and then
``pressure_weighting``. With no transforms every particle counts equally,
which is the standard STILT footprint.

One kernel per receptor
-----------------------

A satellite product gives each sounding its own averaging kernel, and an
EM27/SUN kernel changes with the solar zenith angle. For these, name a
table in the project in place of inline ``levels`` and ``values``:

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       table: kernels.parquet
       coordinate: pres
     - kind: pressure_weighting

The table has a ``receptor`` column with the receptor id, and one row per
kernel point with its ``level`` and ``value``. It can be Parquet or CSV.
Build it with :func:`~stilt.transforms.averaging_kernel_table` from the
receptors you added and the kernels in your product, and write it next
to ``receptors.csv``:

.. code-block:: python

   from stilt.transforms import averaging_kernel_table

   table = averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak)
   table.to_parquet(project.directory / "kernels.parquet")

Pass ``levels`` as one array per receptor, or as a single array when all
kernels share one grid. When the footprint is made, the transform looks up
the receptor's id in the table. The ``table`` path is relative to the
project root, so this works the same in a notebook, with ``stilt run``, and
on Slurm workers. A receptor with no rows in the table
raises an error.

In Python
---------

The same classes work directly on a simulation:

.. code-block:: python

   from stilt.transforms import AveragingKernel, PressureWeighting

   foot = sim.generate_footprint(
       transforms=[
           AveragingKernel(levels=[0, 1000, 2000, 3000], values=[1.0, 0.95, 0.85, 0.7]),
           PressureWeighting(),
       ],
   )

These run after the transforms in the variant's own footprint config, and
the footprint records all of them. To try other footprint settings without
changing the project, pass
``config=sim.footprint_config.model_copy(update={...})``. Without a
simulation, ``traj.footprint(config)`` applies ``config.transforms`` the
same way.

To see what a transform did, apply it to the particle table yourself:

.. code-block:: python

   weighted = PressureWeighting().apply(sim.trajectories.data)
   weighted.drop_duplicates("indx")[["xhgt", "xpres", "pwf"]]

Writing your own transform
--------------------------

Write a pydantic model with an ``apply`` method. Its fields are the keys
you set in YAML. Return a new table and leave the one you are given
unchanged.

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

In ``config.yaml``, set ``kind`` to the class's import path. The other keys
are the model's fields:

.. code-block:: yaml

   transforms:
     - kind: mypkg.transforms.BoundaryLayerOnly
       max_height: 1200

A few rules keep custom transforms working everywhere:

- **Return a copy.** ``Trajectories.data`` must still hold the unweighted
  particles after a footprint is made. The built-ins all call
  ``particles.copy()`` first.
- **Make it importable wherever it runs.** Workers open the project from
  ``config.yaml``, so ``mypkg`` must be installed on the Slurm nodes or in
  the container. A project config that names a transform PYSTILT cannot
  import fails to load, with the import error. A stored footprint that
  names one can still be read, with a warning. That entry of
  ``foot.config.transforms`` is left as its settings mapping.
- **Get anything outside the table from the context.**
  ``context.receptor`` is the receptor, and its ``id`` is the key for any
  per-receptor input file. ``context.variant`` is the variant name.
  ``context.store`` is the project store. Its ``local_path(key)`` gives a
  readable local file for a path relative to the project root, wherever
  the worker runs. The ``averaging_kernel`` table is found this way.

A transform named in ``config.yaml`` must be a pydantic model, because
PYSTILT writes it back out to the variants record and the footprint's
netCDF file. A config that names a plain class fails to load. The Python
``transforms=`` argument takes any object with ``apply``.

The functions the built-ins use are public in :mod:`stilt.transforms`:
:func:`~stilt.transforms.release_coordinate`,
:func:`~stilt.transforms.particle_pwf`, and
:func:`~stilt.transforms.ak_weights`. Build on them before writing your own
weighting from scratch.
