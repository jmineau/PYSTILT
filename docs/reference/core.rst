Core Objects
============

.. currentmodule:: stilt

The project, its releases, the simulations it runs, and their outputs.
The particle runs are `runtraj <https://jmineau.github.io/runtraj/>`_'s, and
its reference has their classes and functions; this page lists what
``stilt`` adds and the names it gives you.

One release, without a project
-------------------------------

:func:`run_trajectories` runs the transport model for one release and
returns its particles, and :func:`calc_footprint` makes a footprint from
them. :func:`footprint.footprint_particles` gives the particles the
``foot`` a footprint's settings ask for, as a project's footprint stage
does. A project runs the same two steps for each of its releases and
stores the results.

.. code-block:: python

   particles = stilt.run_trajectories(release, met, model="hysplit", n_hours=-24, numpar=200)
   foot = stilt.calc_footprint(particles, release, grid)

.. autosummary::
   :toctree: _api
   :nosignatures:

   run_trajectories
   calc_footprint
   footprint.footprint_particles

Project interface
-----------------

:class:`Project` is what you open a project with. It is runtraj's
:class:`runtraj.Project`, which runs the transport model and reads the
particles, with the reads of what STILT makes from the particles added: the
footprints, the Jacobian, and the transport error of a selection. Both open
the same directory; the workers use runtraj's.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Project

Release objects
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Release
   PointRelease
   ColumnRelease
   MultiPointRelease
   read_releases

Simulation objects
------------------

A :class:`~runtraj.config.Variant` is one variant of a project, resolved: its configs, its
met, the transport model build, and the hashes that find its results.
``project.variants`` holds them, and :meth:`ProjectConfig.resolve` makes
them from a config.

:class:`Simulation`, which ``project.simulation(release_id, variant)``
returns, adds the footprint, the enhancement, the background and the
transport error to runtraj's :class:`runtraj.Simulation`, which the workers
and the stages are given.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulation

Settings and their hashes
-------------------------

:mod:`runtraj.identity` records what a run was made with, as its folder's
``_settings.yaml`` holds it, and hashes it. A stage records its own settings
(:meth:`runtraj.stages.Stage.settings`; the footprint's are
:func:`~stilt.footprint.config.footprint_settings`), which are hashed with
the particles' hash and the hash of each stage it reads. A stored record is
read back through the current config classes, so a setting added since, with
a default, still matches.

.. autosummary::
   :toctree: _api
   :nosignatures:

   footprint.config.footprint_settings
   footprint.config.read_footprint_settings

Particles and footprints
------------------------

A simulation's particles are a pandas DataFrame and its footprint an xarray
DataArray. PYSTILT's methods on them are under ``.stilt``:
:class:`~stilt.particles.ParticlesAccessor` (runtraj's
:class:`~runtraj.particles.ParticlesAccessor` with ``enhancement`` added) and
:class:`~stilt.footprint.FootprintAccessor`. The functions below read and
write particle and footprint files without a project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   read_particles
   read_footprint
   footprint.write_footprint
   footprint.write_empty_footprint
   footprint.open_footprints
   particles.ParticlesAccessor
   footprint.FootprintAccessor

Output of STILT-R and X-STILT
-----------------------------

:mod:`stilt.migration` reads the ``out/by-id`` folder of a STILT-R or
X-STILT project as releases, particle tables, and footprints, and imports
it into an output directory, where a ``stilt-r`` variant reads it
(:doc:`/guides/migration/stilt_r`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   migration.read_stilt_r
   migration.import_stilt_r

What the particles give beyond the footprint
--------------------------------------------

The background mole fraction at a release, from a field sampled where
the particles end (:doc:`/guides/background`), and the transport error of
the modeled enhancement, from a wind-error variant
(:doc:`/guides/transport_error`). With a project they are
``sim.background(field)`` and ``sim.transport_error(error_sim, flux)``.
Both weight the particles the way the footprint does, as
``sim.enhancement(flux)`` does for the enhancement of each particle.

.. autosummary::
   :toctree: _api
   :nosignatures:

   background
   Background
   transport_error
   TransportError

Simulation tables
-----------------

``project.releases`` is a pandas DataFrame with one row per release, and
``project.simulations`` one with one row per simulation. Select rows as in
pandas, then pass the selection to the project's :meth:`~Project.status`,
:meth:`~Project.incomplete`, :meth:`~Project.particles`,
:meth:`~Project.footprints`, :meth:`~Project.jacobian`, or
:meth:`~Project.transport_error`.

.. code-block:: python

   sims = project.simulations
   tower = sims[(sims.variant == "hrrr") & (sims.site == "tower_a")]
   feet = project.footprints(tower)

.. currentmodule:: stilt

Spatial geometries
------------------

The cells ``foot.stilt.aggregate`` sums a footprint into, such as
polygons, hexagons, windows around point sources, or groups of grid cells.
A :class:`Grid` works as well. It is documented under :doc:`configuration`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Mesh
   Zones
   footprint.Geometry

:mod:`stilt.footprint` also has the helpers that compute how much of each
footprint cell falls in each target cell. These overlap weights are cached,
so each geometry's weights are computed once per footprint grid.

.. autosummary::
   :toctree: _api
   :nosignatures:

   footprint.overlap_weights
   footprint.check_resolution

Sampling fields
---------------

:func:`runtraj.sampling.sample_field` looks up a gridded field at points:
a surface flux under a footprint or along particle paths
(``foot.stilt.enhancement``, ``particles.stilt.enhancement``), or a
mole-fraction field at particle endpoints (:func:`stilt.background`); it
is listed with the transforms (:doc:`transforms`).
:func:`runtraj.sampling.sample_met` reads the met at particle
positions, for the particle columns a transport model can write from the
met, and :func:`runtraj.sampling.calc_pbl_height` finds the mixed-layer
height from its profile. Both are runtraj's, and both take ARL met files or
met in another format as one dataset (:meth:`runtraj.models.MetConfig.open`).
