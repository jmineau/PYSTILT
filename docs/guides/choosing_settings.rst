Choosing Settings
=================

Four settings decide most of a study's results: which meteorology, how
long the particles run (``n_hours``), how many there are (``numpar``), and
the footprint grid. Each has a default that gets a project started, but the
right value depends on your site and your question. Here is how we choose
each one, and a quick check that tells you whether the choice was good
enough.

.. This page runs when the docs are built, on the sample project of
   docs/_ext/docs_examples.py (four receptors on 15 January 2021).

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir(sample=True)

Meteorology
-----------

Use the finest product that covers your place and years, because the
valleys, coastlines and mixed-layer depth that steer air near a city are
smoothed away on a coarse grid. In the contiguous US since mid-2015 that is
HRRR at 3 km. :doc:`meteorology` has a table of the products and when to use
each. To see how much the choice matters for your site, run the same
receptors with a second met as a second variant and compare the footprints.

How long to run
---------------

``n_hours`` should be long enough for the air to leave the region whose
fluxes you care about, because a flux the particles never reached is
missing from the modeled enhancement. We start with 24 hours for a tower in
a city and 72 hours or more for a regional network or a remote site. The
check is the particles themselves: how long they ran, and how many are
still inside the footprint grid at the end.

.. ipython:: python

   import stilt

   project = stilt.Project("./my_project")
   sim = project.simulation("202101150800_-111.848_40.766_10", "hrrr")
   ends = sim.particles.stilt.endpoints()
   print(f"the longest particle ran {ends['age'].abs().max() / 60:.1f} of 24 hours")

These particles stopped after 13.5 of the 24 hours, because they left the
small crop of met files these pages run on. The met's domain, not
``n_hours``, limited this footprint, so a study would widen the crop
(:doc:`meteorology`). When the particles do run the whole time and many of
them end inside the grid, a longer run would add footprint, so raise
``n_hours``.

How many particles
------------------

More particles give a smoother footprint, and the far part of a footprint,
where few particles reach, needs the most. The default of 200 shows the
shape of a footprint. For results we use 1000, STILT-R's default. To check
a value for your site, run a few receptors with three times as many
particles as a second variant and compare:

.. code-block:: yaml

   variants:
     hrrr: {}
     hrrr-np3k: {numpar: 3000}

If the enhancements you model from the two differ by much less than the
other errors in your study, such as the transport error
(:doc:`transport_error`), the lower count is enough.

The footprint grid
------------------

The grid must cover where the air comes from over the whole run, because
particles outside it add nothing to the footprint. Plot the particles of a
few receptors from different seasons and times of day, and draw the grid
around all of them.

Make the cells as fine as the finest flux inventory you will multiply the
footprints by, so that every inventory is as coarse as the footprint or
coarser (:doc:`aggregation`, "Footprints times a flux"). We use 0.01°
(about 1 km) for a city and 0.1° for a region.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

What else to set
----------------

Most projects leave the other settings at their defaults, which follow
STILT-R. Three are worth knowing:

- ``krand: 2`` with a ``seed`` makes the particles the same on every run, so
  a rerun or a colleague's run gives the same footprint
  (:doc:`../getting_started/quickstart`).
- ``hnf_plume``, the near-field correction, is on by default and matters for
  a receptor near strong sources (:doc:`checking`).
- A transport-error variant estimates how much the wind errors of the
  meteorology change the modeled enhancement (:doc:`transport_error`).
