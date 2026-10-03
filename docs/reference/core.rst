Core Objects
============

.. currentmodule:: stilt

The project, its receptors, the simulations it runs, and their outputs.

Project interface
-----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Project

Receptor objects
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Receptor
   PointReceptor
   ColumnReceptor
   MultiPointReceptor
   read_receptors
   receptors.write_receptors
   receptors.receptors_to_frame
   receptors.receptors_from_frame
   receptors.receptor_rows
   receptors.receptors_from_rows
   receptors.read_receptor_frame
   receptors.parse_receptor_id

Simulation objects
------------------

A :class:`Variant` is one variant of a project, resolved: its configs, its
met, the transport model build, and the hashes that find its results.
``project.variants`` holds them, and :func:`stilt.variants.resolve` makes
them from a config.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulation
   SimID
   Variant
   variants.resolve

Settings and their hashes
-------------------------

:mod:`stilt.identity` records what a result was made with, as each output
folder's ``_settings.yaml`` holds it, and hashes it. A stored record is read
back through the current config classes, so a setting added since, with a
default, still matches.

.. autosummary::
   :toctree: _api
   :nosignatures:

   identity.run_settings
   identity.read_run_settings
   identity.transport_from_settings
   identity.footprint_settings
   identity.read_footprint_settings
   identity.footprint_hash
   identity.settings_hash
   transport.ModelInfo

Particles and footprints
------------------------

A simulation's particles are a pandas DataFrame and its footprint an xarray
DataArray. PYSTILT's methods on them are under ``.stilt``
(:class:`~stilt.particles.ParticlesAccessor`,
:class:`~stilt.footprint.FootprintAccessor`). The functions below read and
write particle and footprint files without a project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   read_particles
   particles_metadata
   write_particles
   read_footprint
   particles.ParticlesAccessor
   footprint.calculate
   footprint.FootprintAccessor

Simulation tables
-----------------

``project.receptors`` is a pandas DataFrame with one row per receptor.
``project.simulations`` is a :class:`Simulations`, one row per simulation.
Select its rows as in pandas, then ask the selection for its status or
results.

.. code-block:: python

   sims = project.simulations
   wbb = sims[(sims.variant == "hrrr") & (sims.site == "WBB")]
   feet = wbb.load_footprints()

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulations

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
   Geometry

:mod:`stilt.footprint` also has the helpers that compute how much of each
footprint cell falls in each target cell. These overlap weights are cached,
so each geometry's weights are computed once per footprint grid.
:mod:`stilt.spatial` has the CRS helpers.

.. autosummary::
   :toctree: _api
   :nosignatures:

   footprint.overlap_weights
   footprint.check_resolution
   spatial.same_crs
   spatial.is_longlat
   spatial.horizontal_dims

Sampling fields
---------------

:mod:`stilt.sampling` looks up a gridded field at points: a surface flux
under a footprint or along particle paths (``foot.stilt.enhancement``,
``particles.stilt.enhancement``), or a mole-fraction field at particle
endpoints (:func:`stilt.observations.background`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   sampling.sample_field
   sampling.vertical_dim
   sampling.nearest_cell
