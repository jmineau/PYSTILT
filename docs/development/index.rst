Development
===========

Contributions are welcome: bug reports, documentation fixes, and code. See
`CONTRIBUTING.md <https://github.com/jmineau/PYSTILT/blob/main/CONTRIBUTING.md>`_
for conventions, and open issues on
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

Writing documentation
---------------------

The sources are in ``docs/``: ``getting_started/`` (install, quickstart,
concepts), ``guides/`` (task-oriented how-tos, one idea per page),
``tutorials/`` (end-to-end worked examples), ``guides/migration/`` (from
STILT-R, X-STILT, stiltctl, Jena STILT), ``reference/`` (API pages built from
docstrings, and the table of every ``config.yaml`` key, made by the
``config-model`` directive in ``docs/_ext/``), and ``development/``
(architecture, STILT-R parity, roadmap). A user-facing change needs a
guide or reference update, and ``just build-docs`` must pass: warnings are
errors.

Voice
^^^^^

Docs, docstrings, config field descriptions, and CLI help all follow this
voice. Most readers are scientists who want to run the model and trust the
result; write for them. Good examples are the
`STILT-R docs <https://uataq.github.io/stilt/>`_ and the xarray, pandas, and
MetPy user guides.

- **Start with what it is for.** Open a page or docstring with what the thing
  does, in plain words. Then show an example. Internals and edge cases go
  last, or in :doc:`architecture`.
- **Keep sentences short.** One idea per sentence. If a sentence needs a
  colon, a semicolon, and a parenthesis, split it.
- **Use the reader's words.** Receptor, particles, footprint, meteorology.
  Internal names (store key, publish, resolve) belong only on pages about
  internals. Describe a config's fields (its parameters) in plain English,
  naming no class or module. ``AGENTS.md`` has the table of words to use.
- **Show it.** A short code block, with its output when that helps, is
  clearer than a paragraph describing it.
- **Say it plainly.** No metaphors ("a dial worth turning"), no selling
  ("powerful", "seamless"), no filler ("note that", "it's worth noting").
  Use American spelling (meters, modeled, center).

Some habits make text read as machine-written. Avoid them:

- Colon reveals: "PYSTILT does one thing: it follows the air." Write
  "PYSTILT follows the air."
- "X, not Y" contrasts when nobody suggested Y.
- A bold sentence at the start of a paragraph that states its point.
- Dashes joining clauses. Use a period or parentheses.
- Lists of three out of habit, and closing sentences that restate the
  paragraph.
- Asides about what the code does "deliberately" or used to do.
- Validation statistics in a user guide. Give the reader the setting to use
  and cite the study.
- Labels invented on one line and referred to later ("the first case").

Docstrings follow the NumPy style. The summary line says what the object is
or what the function returns. Parameter descriptions give the meaning and
units, not the type again. Check every example and claim against the code
when you write it.

API pages
^^^^^^^^^

Each class has a page with tables of its attributes and methods, and each
member a page of its own; a subclass lists what it defines and links to what
it inherits. ``docs/_templates/autosummary/class.rst`` is PYSTILT's own (it
renders config models through ``config-model``). ``docs/_ext/api_pages.py``,
from python-template, decides which members get a row (not pydantic's, so no
list of them is kept) and what is inherited.

A property or attribute's page is where its See Also and Examples go, as in
pandas. Give each public one a docstring that says what it is, and add See
Also or Examples where they help; one with no docstring shows an empty row.
Napoleon reads a first line of the form ``text: more text`` of a property, an
attribute, or module data as ``type: description``, so keep a colon out of
the middle of that line.

Module data such as ``PARTICLE_SCHEMA`` is listed under the package that
exports it (``stilt.particles``). autodoc looks for its ``#:`` comment in that
module, where there is none, so ``reexported_data_comment`` in
``docs/conf.py`` takes the comment from the submodule that defines it.
Without the hook, a page shows the docstring of the object's type
("Built-in immutable sequence.").

Examples in the docs
^^^^^^^^^^^^^^^^^^^^

Examples run when the docs build, so they show real output, and the build
fails when one breaks.

- **A figure in a docstring:** put a ``.. plot::`` directive in the Examples
  section. Its code runs (doctest ``>>>`` lines work), and the figure appears
  under it.

  .. code-block:: rst

     Examples
     --------
     .. plot::

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.plot([0, 1, 2], [0, 1, 4])

- **Code in an .rst page:** an ``.. ipython:: python`` block runs and shows
  each line with its output, as in an IPython session. Changing a
  ``.. code-block:: python`` to ``.. ipython:: python`` is enough to make it
  run. For a figure, put ``@savefig name.png`` on the line above the plotting
  call.

  .. code-block:: rst

     .. ipython:: python

        import matplotlib.pyplot as plt
        import stilt

        stilt.__version__

        @savefig squares.png width=5in
        plt.plot([0, 1, 2], [0, 1, 4]);

  A block that raises or warns fails the build; ``:okexcept:`` or
  ``:okwarning:`` under the directive allows one that is meant to. All .rst
  pages share one session, so each page imports and defines what it uses.
- **When they run:** ``just build-docs`` runs everything, as CI does.
  ``just docs-serve`` reruns a page's blocks whenever that page is rebuilt.
- **Notebooks, and code cells in Markdown pages,** are off in this project.
  ``copier update --data docs_notebooks=true`` turns them on; it adds myst-nb,
  and Jupyter with it, to the dev tools.
- **Data:** the docs build on GitHub Actions, so an example uses synthetic
  data, a small file in the repository, or a public download, never a
  machine-specific path. Seed random numbers so the figures don't change from
  build to build.
- **The Getting Started page runs a real simulation** on the met files the
  tests cache (``tests/met_cache``, 14 and 15 January 2021 around Salt Lake),
  in a temporary folder. ``just build-docs`` needs them: set
  ``STILT_TEST_MET_DIR`` to that folder, or fetch them once with

  .. code-block:: bash

     STILT_TEST_FETCH_MET=1 uv run pytest \
       tests/transport/hysplit/test_release_assignment.py::test_hysplit_multipoint_release_points_follow_control_order \
       --setup-only -q

  A receptor on the page must stay inside that time window. ``docs.yml``
  restores the same cache the tests use.
- **The guides that open a project start from a sample project**
  (``docs/_ext/docs_examples.py``): four receptors an hour apart on 15 January
  2021, run once per docs build (about 40 s) and copied into a temporary
  folder for each page. A page starts it from a hidden block
  (``docs_examples.temp_workdir(sample=True)``) and goes back with
  ``os.chdir(docs_cwd)`` at its end, so the page's files land in the
  temporary folder.
- **Optional dependencies:** the ``dev`` group installs
  ``pystilt[complete]``, so the plotting extra (``visualization``) is there in
  CI. If an example needs another extra, add ``pystilt[extra]`` the same way.

.. toctree::
   :hidden:

   architecture
   stilt_r_parity
   roadmap
