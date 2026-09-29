Transforms
==========

Particle transforms scale each particle's ``foot`` before the footprint is
made (see :doc:`/advanced/transforms`). A transform is any object with an
``apply(particles, context)`` method. The built-ins are pydantic models,
and their fields are their ``config.yaml`` keys.

Interface
---------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.ParticleTransform
   stilt.TransformContext

Built-in transforms
-------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transforms.AveragingKernel
   stilt.transforms.PressureWeighting
   stilt.transforms.FirstOrderLifetime
   stilt.transforms.UnresolvedTransform

Averaging-kernel tables
-----------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transforms.averaging_kernel_table

Science helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transforms.release_coordinate
   stilt.transforms.particle_pwf
   stilt.transforms.ak_weights

Loading and dumping
-------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transforms.load_transform
   stilt.transforms.dump_transform
   stilt.transforms.apply_transforms
