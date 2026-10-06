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

A worker records a :class:`~stilt.exceptions.SimulationError` as a failed
simulation (:attr:`stilt.Simulation.failure`). Its ``reason`` is a short
name for the cause: for HYSPLIT, one of
:class:`stilt.transport.hysplit.FailureReason`, such as ``MET_COVERAGE``,
``MET_TRUNCATED``, ``NO_PARTICLE_DATA``, or ``TIMEOUT``.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.SimulationError
   stilt.exceptions.MeteorologyError

Setup
-----

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.HYSPLITNotFoundError

Empty footprints
----------------

:class:`~stilt.exceptions.EmptyFootprint` is a finished result, not a
failure. :func:`stilt.calc_footprint` raises it when no particle
reaches the grid, and its ``reason`` says why. A run stores the reason in
an empty footprint file.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.exceptions.EmptyFootprint
