Project And Output
==================

A project is a folder holding ``config.yaml`` and ``receptors.csv``. Its
results go to the output directory that ``config.yaml`` names, one tree per
kind of result with a folder per set of settings (see
:doc:`../guides/project_layout` and :doc:`../advanced/output_state`). Most
users only need :class:`stilt.Project`. ``project.output`` is the
:class:`~stilt.output.Output`.

Project
-------

:class:`stilt.Project` is documented with the :doc:`core`.

.. autosummary::
   :toctree: _api
   :nosignatures:


Output directory
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.output.Output
   stilt.output.completed
   stilt.footprint.Jacobian
