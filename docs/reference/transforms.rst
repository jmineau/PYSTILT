Transforms
==========

Particle transforms rescale each particle's ``foot`` before a footprint is
rasterized (see :doc:`/advanced/transforms`). A transform is any object with
``apply(particles, context)``; the built-ins are pydantic models whose fields
are their ``config.yaml`` keys.

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
