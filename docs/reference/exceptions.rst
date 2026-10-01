Exceptions
==========

Every exception PYSTILT raises is a :class:`stilt.StiltError`, so one
``except`` clause catches them all:

.. code-block:: python

   import stilt

   try:
       project.run()
   except stilt.StiltError as error:
       print(f"PYSTILT stopped: {error}")

Each one also subclasses the builtin that describes it. A
:class:`~stilt.exceptions.SimulationError` is a ``RuntimeError`` and a
:class:`~stilt.exceptions.HYSPLITNotFoundError` is a ``FileNotFoundError``,
so code that catches the builtin keeps working. Plain input checks, such as
a negative particle count, raise ``ValueError``.

Base class
----------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.StiltError

Failed runs
-----------

A worker records these as a failed simulation. Its log says why, and
:class:`stilt.transport.hysplit.FailureReason` names the HYSPLIT messages it knows.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.SimulationError
   stilt.exceptions.MeteorologyError
   stilt.exceptions.HYSPLITTimeoutError
   stilt.exceptions.HYSPLITFailureError
   stilt.exceptions.NoParticleOutputError
   stilt.exceptions.EmptyTrajectoryError

Setup
-----

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.HYSPLITNotFoundError

Empty footprints
----------------

:class:`~stilt.exceptions.EmptyFootprint` is a finished result, not a
failure. :meth:`stilt.Trajectories.footprint` raises it when no particle
reaches the grid, and its ``reason`` says why. A run stores the reason in
an empty footprint file.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.EmptyFootprint
