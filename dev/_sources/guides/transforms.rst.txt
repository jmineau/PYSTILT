Particle Weighting (Transforms)
===============================

A footprint is the mean surface influence of a simulation's particles. A
*transform* changes how much each particle counts in that mean, by scaling
the ``foot`` column of the particle table. Column instruments need an
averaging kernel and a pressure weighting. A species that decays during
transport needs a lifetime, one that reacts with OH a rate constant, and
one that deposits to the ground a deposition velocity.

A transform is any object with an ``apply`` method::

   def apply(self, particles, receptor=None, directory=None) -> pd.DataFrame

It takes the particle table, the receptor the particles were released
from, and the project directory, and returns a new table. List transforms in
``config.yaml``, as a default or per variant, or pass them to
:meth:`stilt.Simulation.calc_footprint`. They run once, in order,
starting from the unweighted particles. Each footprint file records the
transforms it was made with.

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
     - kind: dry_deposition
       velocity: 0.01
     - kind: oxidation
       a: 2.45e-12
       e_over_r: 1775
       oxidant: oh.nc

``averaging_kernel`` (:class:`stilt.transforms.AveragingKernel`)
   Multiplies each particle's ``foot`` by the kernel, interpolated to the
   particle's release height (``release_height``). The ``levels`` are in
   meters in the receptor's vertical reference: above ground, or above sea level for a
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
   then stands for the slab of air centered on it, with the ground closing
   the lowest slab. HYSPLIT places each particle at random inside its own
   slab of the column, so the slab around the release height is the air the
   particle stands for. The :doc:`X-STILT migration page
   <migration/x_stilt>` lists how this differs from X-STILT's weights.

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
   :math:`\tau` is ``lifetime_hours``. The age is the particle's time since
   release, its ``age`` column.

``dry_deposition`` (:class:`stilt.transforms.DryDeposition`)
   Scales ``foot`` by the fraction of a depositing tracer that survives
   from the particle's position to the receptor. Use it for a soluble gas
   or a fine aerosol. While a particle is below ``surface_layer_depth``
   (default 50 m), the tracer is lost at the rate ``velocity`` (m/s)
   divided by that depth, HYSPLIT's own dry deposition rule. HYSPLIT does
   not deposit in a STILT run, so one run serves every species.

   The depth is the one number to choose. HYSPLIT takes its lowest
   internal level at or below 75 m, which is 50 m on HRRR. On a stable
   night with particles released at 10 m, 30 m gave a surviving fraction
   of 0.53 after 6 h and 75 m gave 0.62. With 50 m the transform came within
   0.5 % of HYSPLIT's own mass loss in a forward and two backward runs.

   The transform needs ``zagl`` at every one-minute time step, so keep
   ``outdt: 0``, the default. It refuses hourly particles, which do not say
   how long a particle stayed near the ground.
   :func:`~stilt.transforms.dry_deposition_survival` gives the surviving
   fraction for each row of the particle table, if you want the particles
   rather than a footprint. Wet removal and
   gravitational settling are not included.

``oxidation`` (:class:`stilt.transforms.Oxidation`)
   Scales ``foot`` by the fraction of the tracer that survives a reaction
   with an oxidant on its way from the particle's position to the
   receptor. Use it for a species lost mainly to OH, such as CO or ethane.
   At each row the loss rate is :math:`k(T) [X]`: the rate constant
   :math:`k = a \exp(-E/R \, / \, T)`, with ``a`` and ``e_over_r`` as the
   JPL and IUPAC tables give them, times the oxidant's number density
   ``oxidant`` in molecules cm⁻³. Give the oxidant as one number or as a
   netCDF file of a gridded field (an OH climatology or a chemistry model's
   output) on ``lat``/``lon``, with pressure levels in hPa and ``time`` if
   it has them. Each row takes the value of the cell it is in; a row
   outside the field has no loss there. ``stilt check`` reports a missing
   file or variable before the run.

   The temperature is the particles' ``temz`` when ``varsiwant`` has it.
   Otherwise it comes from ``pres`` and ``dens``, which are there by
   default, so particles you already ran need no rerun. That is within
   about 1 K of HYSPLIT's ``temz`` in dry air, about 2 % in the rate for
   methane and OH.

   For methane at regional scales you do not need it: its lifetime
   against OH is about nine years, and over a few days of trajectory the
   footprint changes by less than 0.1 %.

Any loss whose rate you can compute at every row works the same way, as
long as the rate does not depend on how much tracer the particle carries.
:func:`~stilt.transforms.survival` takes a rate per row, in s⁻¹, and
returns the fraction that reaches the receptor; the built-in lifetime,
deposition and oxidation transforms are special cases of it. Use it in your
own transform for rain-out, photolysis, or a lifetime from a chemistry
model. A loss that depends on the tracer itself, as NOx's lifetime depends
on NOx, has no footprint: it needs the emissions and a chemistry model
along each trajectory.

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
       table: kernels
       coordinate: pres
     - kind: pressure_weighting

The table has a ``receptor`` column with the receptor id, and one row per
kernel point with its ``level`` and ``value``. Build it with
:func:`~stilt.transforms.averaging_kernel_table` from the receptors you
added and the kernels in your product, and add it to the project:

.. code-block:: python

   from stilt.transforms import averaging_kernel_table

   table = averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak)
   project.add_table("kernels", table)     # tables/kernels.parquet

:func:`~stilt.observations.receptors_from_soundings` returns this table
for you. Pass ``levels`` as one array per receptor, or as a single array
when all kernels share one grid. When the footprint is made, the
transform looks up the receptor's id in the table. The table is part of
the project, so this works the same in a notebook, with ``stilt run``, and
on Slurm workers. A receptor with no rows in the table raises an error, and
``stilt check`` lists those receptors before a run, with the table's ids that
match no receptor (:doc:`checking`). ``table`` also takes the path of a
Parquet or CSV file, relative to the project root.

In Python
---------

The same classes work directly on a simulation:

.. code-block:: python

   from stilt.transforms import AveragingKernel, PressureWeighting

   foot = sim.calc_footprint(
       transforms=[
           AveragingKernel(levels=[0, 1000, 2000, 3000], values=[1.0, 0.95, 0.85, 0.7]),
           PressureWeighting(),
       ],
   )

These replace the transforms in the variant's own footprint settings, and
the footprint records them. The other settings (``grid``,
``smooth_factor``, ``tres``) are replaced the
same way, and nothing in the project changes. Without a simulation,
:func:`stilt.calc_footprint` takes the same ``transforms``.

To see what a transform did, apply it to the particle table yourself:

.. code-block:: python

   weighted = PressureWeighting().apply(sim.particles)
   weighted.drop_duplicates("particle")[["release_height", "xpres", "pwf"]]

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

   from stilt.transforms import release_coordinate


   class BoundaryLayerOnly(BaseModel):
       """Drop influence from particles released above ``max_height`` m AGL."""

       max_height: float = 1500.0

       def apply(self, particles, receptor=None, directory=None):
           z = release_coordinate(particles, "release_height") # one value per particle
           keep = z.reindex(particles["particle"].to_numpy()) <= self.max_height
           out = particles.copy()
           out["foot"] = np.where(keep.to_numpy(), out["foot"], 0.0)
           return out

In ``config.yaml``, set ``kind`` to the class's import path. The other keys
are the model's fields:

.. code-block:: yaml

   transforms:
     - kind: mypkg.transforms.BoundaryLayerOnly
       max_height: 1200

What a transform may do
~~~~~~~~~~~~~~~~~~~~~~~

The footprint is gridded from ``foot``, so a transform changes the
footprint through ``foot`` alone. It may add columns as well, such as a
chemistry step's per-particle concentrations, and leave ``foot`` as it was.
The built-in transforms only scale ``foot``. ``sim.background`` and
``sim.transport_error`` rely on that: they apply the transforms to
particles whose ``foot`` is 1 and read each particle's weight from the
result. A transform that does more than scale ``foot`` makes those weights
something other than a weight.

A few rules keep custom transforms working everywhere:

- **Return a copy.** ``sim.particles`` must still hold the unweighted
  particles after a footprint is made. The built-ins all call
  ``particles.copy()`` first.
- **Make it importable wherever it runs.** Workers open the project from
  ``config.yaml``, so ``mypkg`` must be installed on the Slurm nodes or in
  the container. A project config that names a transform PYSTILT cannot
  import fails to load, with the import error. A stored footprint that
  names one can still be read, with a warning. That entry of
  ``foot.stilt.config.transforms`` is left as its settings mapping.
- **Get anything outside the table from the receptor or the directory.**
  ``receptor.id`` is the key for any per-receptor input file, and
  ``receptor.altitude_ref`` says whether heights are above ground or sea
  level. ``directory`` is the project directory. Read a file named in your
  transform's settings relative to it, as the ``averaging_kernel`` table
  is read. Both can be ``None`` when a transform is applied to particles
  alone, so raise a clear error if yours needs them.

- **Let** ``stilt check`` **see its inputs** (optional). A ``check(self,
  receptors, directory)`` method gets the project's receptor ids and
  directory and returns what is wrong, as ``(check, receptor id or None,
  message)`` tuples, so a missing input file shows up before any run. The
  ``averaging_kernel`` checks its table this way. Set ``reads_zsfc = True``
  on the class if it reads ``zsfc`` for a receptor above sea level, as
  ``pressure_weighting`` does, and ``stilt check`` says when ``varsiwant``
  lacks it.

A transform named in ``config.yaml`` must be a pydantic model, because
PYSTILT writes it back out to the footprint folder's ``_settings.yaml``
and to each footprint file. A config that names a plain class fails to
load. The Python ``transforms=`` argument takes any object with ``apply``.

The functions the built-ins use are public in :mod:`stilt.transforms`:
:func:`~stilt.transforms.release_coordinate`,
:func:`~stilt.transforms.particle_pwf`, and
:func:`~stilt.transforms.ak_weights`. Build on them before writing your own
weighting from scratch.
