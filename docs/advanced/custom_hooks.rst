.. _custom_hooks:

Particle Transforms
===================

PYSTILT applies typed particle transforms after trajectories are available and
before a footprint is rasterized. This is the core interface used for observation
weighting and lightweight chemistry adjustments.

Two entry points are supported:

- declarative built-in transforms in :class:`~stilt.config.FootprintConfig`
- runtime transforms passed to
  :meth:`~stilt.simulation.Simulation.generate_footprint`

Unlike the old ``before_footprint`` callback idea, these transforms are typed,
validated, and easier to document and serialize.

Declarative transforms
----------------------

For ``config.yaml`` and ordinary Python configuration, attach built-in
transform specs directly to a footprint product:

.. code-block:: python

   from stilt.config import FootprintConfig, VerticalOperatorTransformSpec

   footprint = FootprintConfig(
       grid=grid,
       transforms=[
           VerticalOperatorTransformSpec(
               kind="vertical_operator",
               mode="ak_pwf",
               levels=[0.0, 1000.0, 2000.0],
               values=[0.2, 0.5, 0.3],
               coordinate="xhgt",
           ),
       ],
   )

Built-ins are intentionally narrow:

- ``vertical_operator``
- ``first_order_lifetime``

These specs are validated when config is loaded and are stored with footprint
metadata for provenance.

Runtime transforms
------------------

Advanced Python workflows can pass typed transforms directly at runtime.
This is useful when transform state comes from an in-memory observation object
or other dynamic context.

.. code-block:: python

   from stilt.observations import VerticalOperator, VerticalOperatorWeighting
   from stilt.transforms import ParticleTransformContext

   operator = VerticalOperator(
       mode="ak_pwf",
       levels=[0.0, 1000.0, 2000.0],
       values=[0.2, 0.5, 0.3],
   )
   transform = VerticalOperatorWeighting(operator)

   sim.generate_footprint(
       "column",
       transforms=[transform],
       transform_context=ParticleTransformContext(
           receptor=sim.receptor,
           footprint_name="column",
           footprint_config=sim.config.footprints["column"],
       ),
   )

The runtime transform interface is still deliberately small: transforms operate
on the particle DataFrame and receive a
:class:`~stilt.transforms.ParticleTransformContext`.
