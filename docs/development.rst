Development
===========

Contributions are welcome: bug reports, documentation fixes, and code. See
`CONTRIBUTING.md <https://github.com/jmineau/PYSTILT/blob/main/CONTRIBUTING.md>`_
for conventions, and open issues on
`GitHub <https://github.com/jmineau/PYSTILT/issues>`_.

Set up a development environment
--------------------------------

PYSTILT uses `uv <https://docs.astral.sh/uv/>`_ and
`just <https://github.com/casey/just>`_. From a clone of the repository:

.. code-block:: bash

   uv sync --group dev

Common tasks:

.. code-block:: bash

   just test             # run the test suite
   just quality-check    # lint, type-check, and test
   just build-docs       # build this documentation into docs/_build/html
