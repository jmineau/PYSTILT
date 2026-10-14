Internals
=========

How PYSTILT works inside. You don't need these pages to use PYSTILT. They
help when you run at scale, share an output directory between projects,
or build on PYSTILT.

:doc:`output_state`
   How a result is found, why a changed setting makes a new folder, and
   how two workers never step on each other.

:doc:`architecture`
   How the code is organized, how a run goes from a project to result
   files, and the implementation details that are easy to break.

.. toctree::
   :maxdepth: 1
   :hidden:

   output_state
   architecture
