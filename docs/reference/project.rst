Project And Output
==================

A project is a folder holding ``config.yaml`` and ``receptors.csv``. Its
results go to the output directory that ``config.yaml`` names, one tree per
kind of result with a folder per set of settings (see
:doc:`../guides/project_layout` and :doc:`../advanced/output_state`). Most
users never use these classes directly. :class:`stilt.Model` uses them for
you; ``model.output`` is the :class:`~stilt.Output`.

Project
-------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.project.Project
   stilt.project.project_slug

Output directory
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.output.Output
   stilt.output.Run
   stilt.output.Footprints
   stilt.output.Jacobian
   stilt.output.convert_project
   stilt.simulation.VariantOutput
