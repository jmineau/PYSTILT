Development
===========

Contributions are welcome: bug reports, documentation fixes, and code. See
`CONTRIBUTING.md <https://github.com/jmineau/PYSTILT/blob/main/CONTRIBUTING.md>`_
for conventions and for how to write the docs, and open issues on
`GitHub <https://github.com/jmineau/PYSTILT/issues>`_.

:doc:`architecture`
   How the code is organized, how a run goes from a project to result files,
   and how PYSTILT knows which work is finished.

:doc:`stilt_r_parity`
   What "matches STILT-R" covers, and how to move to a newer STILT-R commit.

:doc:`roadmap`
   What is implemented and what is planned.

Use of AI coding agents
-----------------------

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.

If you contribute with an agent, ``AGENTS.md`` at the repository root is the
orientation file it should read.

Set up a development environment
--------------------------------

PYSTILT uses `uv <https://docs.astral.sh/uv/>`_ and
`just <https://github.com/casey/just>`_. From a clone of the repository:

.. code-block:: bash

   uv sync
   uv run pre-commit install

Common tasks:

.. code-block:: bash

   just test             # run the unit tests
   just quality-check    # lint, type check, import contracts, docstrings, unit tests
   just build-docs       # build this documentation into docs/_build/html
   just docs-serve       # preview it at http://127.0.0.1:8000, rebuilt on every save

``just`` with no arguments lists the rest.

.. toctree::
   :hidden:

   architecture
   stilt_r_parity
   roadmap
