Transforms
==========

Particle transforms multiply each particle row's ``weight``, and the
footprint is made from ``foot`` times ``weight`` (see
:doc:`/guides/transforms`). They are runtraj's (:mod:`runtraj.transforms`),
listed here because PYSTILT's footprint, background and transport error use
them. A transform is any object with an ``apply(particles, release=None,
directory=None)`` method. The built-ins are pydantic models, and their
fields are their ``config.yaml`` keys.

Built-in transforms
-------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   runtraj.transforms.AveragingKernel
   runtraj.transforms.PressureWeighting
   runtraj.transforms.FirstOrderLifetime
   runtraj.transforms.DryDeposition
   runtraj.transforms.Oxidation

Averaging-kernel tables
-----------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   runtraj.transforms.averaging_kernel_table

Science helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   runtraj.transforms.weigh
   runtraj.transforms.release_coordinate
   runtraj.transforms.particle_pwf
   runtraj.transforms.ak_weights
   runtraj.transforms.survival
   runtraj.transforms.dry_deposition_survival

Loading and dumping
-------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   runtraj.transforms.load_transform
   runtraj.transforms.dump_transform
   runtraj.transforms.apply_transforms
   runtraj.transforms.transform_kind

Sampling a field at particles
-----------------------------

:func:`~runtraj.sampling.sample_field` reads a gridded field in the cell
nearest each point: an oxidant along each trajectory, a mole fraction where
each particle ends (:func:`stilt.background`), or a surface flux under each
particle (``particles.stilt.enhancement``).

.. autosummary::
   :toctree: _api
   :nosignatures:

   runtraj.sampling.sample_field
   runtraj.sampling.vertical_dim
   runtraj.sampling.nearest_cell
