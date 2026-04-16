User Guide
==========

The user guide is organized around the full simulation lifecycle:

1. define project configuration and receptors,
2. point PYSTILT to ARL meteorology,
3. run with local, Slurm, or queue-worker execution patterns,
4. inspect trajectories, footprints, and run status.

Two optional higher-level layers now sit on top of that same runtime:

- ``stilt.service`` for queue/service-oriented Python workflows
- ``stilt.observations`` for science-facing observation and receptor-building workflows

Execution semantics in this guide reflect current alpha behavior:

- at-least-once processing,
- explicit per-footprint terminal states (including ``complete-empty``),
- skip-or-rerun control through ``skip_existing``.

.. toctree::
   :maxdepth: 2

   project_setup
   meteorology
   running
   execution
   observations
   service
   results
