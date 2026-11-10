Meteorology
===========

Each transport model reads its meteorology from a met config of its own.
HYSPLIT's, :class:`~stilt.transport.hysplit.MetConfig`, holds the settings
for one met in ``config.yaml``: ARL files in a directory, or an archive to
download. :class:`~stilt.transport.hysplit.Met` finds the files a
simulation needs, downloads them when ``download`` is set, and crops them
when asked. See :doc:`../guides/meteorology` for how to set it up.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.MetConfig
   stilt.transport.hysplit.Met
   stilt.meteorology.run_window

How a met finds its files
-------------------------

These are the exact rules behind :doc:`../guides/meteorology`.

Which files a run reads
~~~~~~~~~~~~~~~~~~~~~~~

Each ARL file's first index record gives its
first time, its grid and the size of a time step, and the second gives the
spacing, so PYSTILT chooses files by their headers and never by their names.
Files are found anywhere under ``directory``, and links are followed. Lock
files, backup copies ending in ``~``, names ending in ``.partial`` or
``.bad``, and anything that is not an ARL file (such as a README) are
skipped.

HYSPLIT interpolates between the met steps on either side of each of its
times, so a run needs the files from the step at or before its start to the
step at or after its end: the release of a backward run, the end of a
forward one (:meth:`Met.files_for <stilt.transport.hysplit.Met.files_for>`).
With 6-hour HRRR files of hourly steps, a backward run released at 23:30
needs the next day's first file, and one released at 23:00 does not. With
daily NAM12 files of 3-hourly steps (00:00 to 21:00), a run released at
22:00 needs the next day's file. A 24-hour backward run on 6-hour HRRR files
needs 5 or 6 files, depending on the receptor hour. Downloads follow the
same rule, with the archive's own step (arlmet's ``Archive.step``).

The catalog
~~~~~~~~~~~

What the files hold is kept in a table per folder, in ``catalogs/`` inside
``subgrid_dir``, or under ``$XDG_CACHE_HOME`` (else ``~/.cache``) without
one. ``stilt check`` brings it up to date; otherwise the first simulation builds it while the
others wait. A simulation reads the table, and looks in the folder again
only when the table has no file for one of its hours or a file it chose has
changed. Files of two products or two grids that hold the same time fail
the run, naming both. A file that holds part of what another file of the
same product holds, such as a short copy, is passed over for the longer one.

Files that fail a run
~~~~~~~~~~~~~~~~~~~~~

These fail a run before HYSPLIT starts, as ``MET_COVERAGE``: an hour
that no file holds, a file whose first index record cannot be read, a file
with one time period, and a damaged file (cut short, an hour written
partway, or records lost to null bytes), found by
:meth:`arlmet.File.check`. Each file is checked once in a process, which
reads every record's header: a few seconds for a 1 GB HRRR file. After the
run, a run that stopped early with a particle more than 5 grid cells inside
the met's grid fails the same way (:doc:`hysplit`, "Failure reasons").

Crops
~~~~~

Each crop gets a folder inside ``subgrid_dir``, named by a hash of
``subgrid_bounds`` and ``subgrid_levels``
(:attr:`Met.crop_dir <stilt.transport.hysplit.Met.crop_dir>`), so a new crop
box starts a new folder and projects with the same box share one. Old
folders are not deleted. The crops have a catalog of their own, so a crop
serves a run by the hours its header holds, whatever it is named and
whether or not the archive still has its source. A crop that holds fewer
hours than the archive file it shares a name with is made again when a run
needs the hours it lacks. A run records the files it was given: the
archive's, or the crops that held its hours.

What a run records
~~~~~~~~~~~~~~~~~~

The weather product is read from the header of the
first file in ``directory`` (a download's comes from its archive) when the
project's settings folders are first needed, to run or to read results. So
``directory`` must hold the files on the machine that opens the project
(:meth:`MetConfig.settings <stilt.transport.hysplit.MetConfig.settings>`).

Wind-error statistics
---------------------

Variograms of analysis-minus-observation winds. Fit them to get the
wind-error settings for a transport-error run (:ref:`wind-error-statistics`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.variogram
   stilt.meteorology.fit_variogram
   stilt.meteorology.VariogramFit
