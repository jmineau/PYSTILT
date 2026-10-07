"""
The :class:`Project`: a directory of receptors and settings, and a view of their results.

A project is a directory holding what the user wrote::

    config.yaml       the settings (written once, by Project.init or stilt init)
    receptors.csv     the receptors (only appended to)

Results go to the output directory ``config.yaml`` names
(:class:`stilt.output.Output`), ``./output`` by default. The simulations a
project defines are its receptors crossed with the variants in
``config.yaml``. A project reads; the workers in :mod:`stilt.execution` are
the only code that writes results.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
import yaml

from stilt._paths import absolute, location, write_parquet
from stilt.config import STARTER_CONFIG, ProjectConfig, Variant
from stilt.execution.config import ExecutionConfig
from stilt.footprint import Geometry, Jacobian
from stilt.footprint.aggregation import _jacobian
from stilt.footprint.io import open_footprints
from stilt.output import Kind, Output, completed
from stilt.particles import particles_from_table
from stilt.receptors import Receptor, read_receptors
from stilt.receptors.table import (
    COLUMNS,
    ROW_COLUMNS,
    append_receptors_csv,
    check_distinct_ids,
    read_receptor_frame,
    receptor_rows,
    receptors_from_rows,
    receptors_to_csv,
)
from stilt.simulation import Simulation

if TYPE_CHECKING:
    from stilt.visualization import ProjectPlotAccessor


#: The columns of :attr:`Project.receptors` before the label columns.
RECEPTOR_COLUMNS = ("receptor", "time", "kind", "location")

#: Where a simulation stands, the ``state`` column of :meth:`Project.status`.
STATES = ("complete", "failed", "interrupted", "pending")


def _as_receptors(
    receptors: Receptor | Iterable[Receptor] | str | Path,
) -> list[Receptor]:
    """Return receptors given as one, several, or the path of a receptors CSV."""
    if isinstance(receptors, (str, Path)):
        return read_receptors(absolute(receptors))
    if isinstance(receptors, Receptor):
        return [receptors]
    items = list(receptors)
    if not all(isinstance(item, Receptor) for item in items):
        raise TypeError(
            "Receptors must be a Receptor, an iterable of Receptors, or the path "
            "of a receptors CSV."
        )
    return items


#: A variant and a realization, how the bulk methods group a selection.
_Key = tuple[str, int | None]
#: The receptors with particles, and with a footprint (``None`` when the variant makes none).
_Present = tuple[frozenset[str], frozenset[str] | None]


def _realization(value: Any) -> int | None:
    """Return a ``realization`` cell as an int, or ``None`` for a single run."""
    return None if pd.isna(value) else int(value)


def _rows(frame: pd.DataFrame) -> list[tuple[str, str, int | None]]:
    """Return each row's ``(receptor, variant, realization)``."""
    return [
        (str(r), str(v), _realization(k))
        for r, v, k in zip(
            frame["receptor"], frame["variant"], frame["realization"], strict=True
        )
    ]


def _groups(frame: pd.DataFrame) -> Iterator[tuple[str, int | None, pd.DataFrame]]:
    """Yield the rows of each variant and realization in *frame*."""
    for key, rows in frame.groupby(
        ["variant", "realization"], sort=False, dropna=False
    ):
        assert isinstance(key, tuple)  # grouped by two columns
        name, k = key
        yield str(name), _realization(k), rows


def _one_group(frame: pd.DataFrame, what: str) -> tuple[str, int | None]:
    """Return the one ``(variant, realization)`` of a selection, or raise."""
    groups = list(dict.fromkeys((v, k) for _, v, k in _rows(frame)))
    if len(groups) != 1:
        raise ValueError(
            f"{what} is made from one variant (and one realization of an "
            f"ensemble); this selection has {groups}. Select one first, as in "
            "sims[sims.variant == 'hrrr']."
        )
    return groups[0]


def _is_mask(sel: Any) -> bool:
    """Return whether *sel* is a boolean mask (a Series, an array, or a list of booleans)."""
    if isinstance(sel, (pd.Series, np.ndarray)):
        return sel.dtype == bool or str(sel.dtype) == "boolean"
    return isinstance(sel, list) and bool(sel) and all(isinstance(v, bool) for v in sel)


def _column(table: Any, name: str) -> list[str]:
    """Return one column of a pandas, polars, or pyarrow table as a list of strings."""
    column = table[name]
    for convert in ("to_pylist", "to_list", "tolist"):  # pyarrow, polars, pandas
        if hasattr(column, convert):
            return [str(v) for v in getattr(column, convert)()]
    return [str(v) for v in column]


class Project:
    """
    A STILT project: receptors, settings, and the simulations they define.

    ``Project(path)`` opens a project directory and reads it. It is every
    receptor under every variant, and a view of their results: list the
    simulations, see which have finished, load particles and footprints.
    The results are in the output directory ``config.yaml`` names
    (``./output`` by default), which several projects can share.

    Make a new project with :meth:`init` (or ``stilt init``). After that,
    ``config.yaml`` is yours to edit; PYSTILT never rewrites it.

    Parameters
    ----------
    path : str or Path
        Project directory.
    output : str or Path, optional
        Read and write results here instead of the output directory
        ``config.yaml`` names: a path, or a URL such as
        ``s3://bucket/output``. For a container whose output is somewhere
        else than the project's.

    Attributes
    ----------
    directory : Path
        The project directory, absolute.

    Examples
    --------
    Make a project, run it, and load a footprint:

    >>> import stilt
    >>> receptor = stilt.PointReceptor(
    ...     time="2023-07-15 18:00", longitude=-111.848, latitude=40.766, altitude=10
    ... )
    >>> met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
    >>> project = stilt.Project.init(
    ...     "./my_project",
    ...     receptors=[receptor],
    ...     mets={"hrrr": met},
    ...     variants={"hrrr": {}},
    ...     n_hours=-24,
    ...     grid={
    ...         "xmin": -113,
    ...         "xmax": -110.5,
    ...         "ymin": 40,
    ...         "ymax": 42,
    ...         "xres": 0.01,
    ...         "yres": 0.01,
    ...     },
    ... )
    >>> project.run()
    >>> foot = project.simulation(receptor.id, "hrrr").footprint

    Open it again later and check a selection of simulations:

    >>> project = stilt.Project("./my_project")
    >>> sims = project.simulations
    >>> project.status(sims[sims.variant == "hrrr"])
    """

    def __init__(self, path: str | Path, *, output: str | Path | None = None) -> None:
        self.directory = absolute(path)
        self._output = output

    @classmethod
    def init(
        cls,
        path: str | Path,
        receptors: Receptor | Iterable[Receptor] | str | Path | None = None,
        *,
        config: ProjectConfig | None = None,
        starter: bool = False,
        **settings: Any,
    ) -> Project:
        """
        Make a new project and return it.

        Writes ``config.yaml``, with only the settings given, and
        ``receptors.csv`` when receptors are given. With ``starter=True`` it
        writes the commented starter config instead, to edit by hand, as
        ``stilt init`` does.

        Parameters
        ----------
        path : str or Path
            Project directory. It is created if needed.
        receptors : Receptor, iterable of Receptor, str or Path, optional
            Receptors, or the path of a receptors CSV to copy them from.
        config : ProjectConfig, optional
            The settings. Or give them as keywords instead.
        starter : bool, default False
            Write the commented starter ``config.yaml``
            (:data:`stilt.config.STARTER_CONFIG`) in place of a config.
            Without receptors, ``receptors.csv`` gets only its header.
        **settings
            Settings for :class:`~stilt.ProjectConfig`, such as ``mets``, ``variants``,
            ``n_hours``, ``numpar``, and ``grid``.

        Raises
        ------
        FileExistsError
            If the directory already has a ``config.yaml``. Open it with
            ``Project(path)`` instead, and edit the file to change settings.
        TypeError
            If both *config* and keyword settings are given, or either is
            given with ``starter=True``.
        ValueError
            If a variant's settings are invalid. Nothing is written.
        """
        if config is not None and settings:
            raise TypeError("Give a ProjectConfig or keyword settings, not both.")
        if starter and (config is not None or settings):
            raise TypeError(
                "A starter config takes no settings; edit its config.yaml instead."
            )
        if starter:
            config = ProjectConfig.model_validate(yaml.safe_load(STARTER_CONFIG))
        elif config is None:
            config = ProjectConfig(**settings)
        project = cls(path)
        # The config checks a variant's transport settings only when it is
        # resolved; do it now, so a bad config is never written.
        config.resolve(project.directory)
        if project.config_path.exists():
            raise FileExistsError(
                f"{project.directory} already has a config.yaml. Open it with "
                "Project(path), and edit config.yaml to change its settings."
            )
        project.directory.mkdir(parents=True, exist_ok=True)
        if starter:
            project.config_path.write_text(STARTER_CONFIG)
        else:
            config.to_yaml(project.config_path)
        if receptors is not None:
            project.add_receptors(receptors)
        elif starter:
            # Header only: every other line of a receptors file is a receptor.
            project.receptors_path.write_text("time,longitude,latitude,altitude\n")
        return project

    def __repr__(self) -> str:
        return f"Project({str(self.directory)!r})"

    # -- inputs ----------------------------------------------------------------

    @property
    def name(self) -> str:
        """The project's name: the directory name."""
        return self.directory.name

    @property
    def config_path(self) -> Path:
        """The project's ``config.yaml``."""
        return self.directory / "config.yaml"

    @property
    def receptors_path(self) -> Path:
        """The project's ``receptors.csv``."""
        return self.directory / "receptors.csv"

    @cached_property
    def config(self) -> ProjectConfig:
        """
        The settings, read from ``config.yaml``.

        Raises
        ------
        FileNotFoundError
            If the project has no ``config.yaml``.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml in {self.directory}. Make a project with "
                "stilt.Project.init(...) or `stilt init`."
            )
        return ProjectConfig.from_yaml(self.config_path)

    @cached_property
    def output(self) -> Output:
        """
        The output directory.

        The ``output`` the project was opened with, or else ``config.output``
        (``./output`` by default), relative to the project.
        """
        if self._output is not None:
            return Output(location(self._output))
        return Output(location(self.config.output, self.directory))

    @cached_property
    def variants(self) -> dict[str, Variant]:
        """
        Each variant, resolved, by name, in config order.

        An ensemble (``realizations: N``) is one variant; its realizations
        are the ``realization`` column of :attr:`simulations`. A footprint given by a geometry gets its grid
        here, so the geometry is read on first use (:meth:`stilt.ProjectConfig.resolve`),
        and a relative geometry file starts from the project directory.
        """
        return self.config.resolve(self.directory)

    @cached_property
    def _rows(self) -> pd.DataFrame:
        """
        The rows of ``receptors.csv``, checked, with each row's receptor id.

        No receptor is built here; :meth:`receptor` builds one when asked.
        """
        if not self.receptors_path.exists():
            empty = pd.DataFrame(columns=["time", "longitude", "latitude", "altitude"])
            return receptor_rows(empty)
        return receptor_rows(read_receptor_frame(self.receptors_path))

    @cached_property
    def _positions(self) -> dict[str, np.ndarray]:
        """Where each receptor's rows are in :attr:`_rows`, by id, so :meth:`receptor` need not search."""
        groups = self._rows.groupby("receptor", sort=False).indices
        return {str(rid): np.asarray(positions) for rid, positions in groups.items()}

    @cached_property
    def receptors(self) -> pd.DataFrame:
        """
        The receptors, one row per receptor, in file order.

        The columns are ``receptor`` (the id), ``time``, ``kind``
        (``point``, ``column``, or ``multipoint``), ``location`` (the
        location id), and one column per label from ``receptors.csv``.
        :meth:`receptor` gives the receptor itself.
        """
        rows = self._rows
        labels = [c for c in rows.columns if c not in COLUMNS and c not in ROW_COLUMNS]
        first = rows.drop_duplicates("receptor")
        frame = first.loc[:, [*RECEPTOR_COLUMNS, *labels]].reset_index(drop=True)
        return frame.astype({"time": "datetime64[ns]"})

    def receptor(self, receptor_id: str) -> Receptor:
        """
        Return one receptor by id, built from its rows.

        Raises
        ------
        KeyError
            If the project has no receptor with that id.
        """
        return self._receptors([receptor_id])[receptor_id]

    def _receptors(self, receptor_ids: Iterable[str]) -> dict[str, Receptor]:
        """
        Return receptors by id, built together from their rows.

        Selecting a receptor's rows from a large table is most of the cost
        of building it, so a selection builds all of its receptors in one
        pass: on a project of millions of rows that is about 0.05 ms a
        receptor, against several ms one at a time.

        Raises
        ------
        KeyError
            If the project has no receptor with one of the ids.
        """
        ids = list(dict.fromkeys(receptor_ids))
        for rid in ids:
            if rid not in self._positions:
                raise KeyError(f"No receptor {rid!r} in {self.directory}.")
        if not ids:
            return {}
        rows = self._rows.iloc[np.concatenate([self._positions[r] for r in ids])]
        # receptors_from_rows builds them in the order the ids first appear.
        return dict(zip(ids, receptors_from_rows(rows), strict=True))

    def add_receptors(
        self, receptors: Receptor | Iterable[Receptor] | str | Path
    ) -> list[str]:
        """
        Add receptors to ``receptors.csv`` and return the ids of the new ones.

        The file is created if needed, and otherwise only appended to, in its
        own columns (:func:`stilt.receptors.table.append_receptors_csv`). Receptors
        it already holds are skipped.

        Parameters
        ----------
        receptors : Receptor, iterable of Receptor, str or Path
            Receptors, or the path of a receptors CSV to add them from.

        Raises
        ------
        ValueError
            If a new receptor has the id of a different receptor, since an
            id names a receptor's result files.
        """
        batch = _as_receptors(receptors)
        known = [self.receptor(r.id) for r in batch if r.id in self._positions]
        check_distinct_ids([*known, *batch])
        new = list({r.id: r for r in batch if r.id not in self._positions}.values())
        if new:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.receptors_path.exists():
                text = self.receptors_path.read_text()
                self.receptors_path.write_text(append_receptors_csv(text, new))
            else:
                self.receptors_path.write_text(receptors_to_csv(new))
            # These were read from the file that just changed.
            for name in ("_rows", "_positions", "receptors", "_simulations"):
                vars(self).pop(name, None)
        return [r.id for r in new]

    def table_path(self, name: str) -> Path:
        """Return where the project's table *name* is kept: ``tables/<name>.parquet``."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                f"A table name is letters, digits, '_', and '-'; got {name!r}."
            )
        return self.directory / "tables" / f"{name}.parquet"

    def add_table(self, name: str, table: pd.DataFrame) -> Path:
        """
        Add rows to one of the project's input tables, and return its path.

        A table is an input, like ``receptors.csv``: the averaging kernels of
        satellite soundings, for one, which a transform names as ``table:
        kernels``. It is kept as ``tables/<name>.parquet``, created if
        needed and otherwise added to. When the table has a ``receptor``
        column, rows of a receptor it already holds are left out, so adding
        the same kernels again changes nothing.

        Parameters
        ----------
        name : str
            The table's name: letters, digits, ``_``, and ``-``.
        table : pandas.DataFrame
            Rows to add, in the columns of the table.

        Raises
        ------
        ValueError
            If *table* has other columns than the table it adds to.

        Examples
        --------
        >>> receptors, kernels = receptors_from_soundings(df, "slant", top=3000)
        >>> project.add_receptors(receptors)
        >>> project.add_table("kernels", kernels)
        """
        path = self.table_path(name)
        rows = table.reset_index(drop=True)
        if path.exists():
            held = pd.read_parquet(path)
            if list(held.columns) != list(rows.columns):
                raise ValueError(
                    f"Table {name!r} has the columns {list(held.columns)}, "
                    f"not {list(rows.columns)}."
                )
            if "receptor" in rows.columns:
                rows = rows[~rows["receptor"].isin(held["receptor"])]
            rows = pd.concat([held, rows], ignore_index=True)
        write_parquet(pa.Table.from_pandas(rows, preserve_index=False), path)
        return path

    # -- simulations -----------------------------------------------------------

    @cached_property
    def _simulations(self) -> pd.DataFrame:
        """The table :attr:`simulations` copies, built once."""
        variants = pd.DataFrame(
            [
                (name, k, v.model.name)
                for name, v in self.variants.items()
                for k in v.realization_numbers
            ],
            columns=["variant", "realization", "model"],
        ).astype({"realization": "Int64"})
        frame = self.receptors.merge(variants, how="cross")
        first = ["receptor", "variant", "realization", "model"]
        return frame.loc[:, first + [c for c in frame.columns if c not in first]]

    @property
    def simulations(self) -> pd.DataFrame:
        """
        Every receptor under every variant, one row per simulation.

        A pandas DataFrame with the columns ``receptor``, ``variant``,
        ``realization`` (``0`` to ``N - 1`` for a variant with
        ``realizations: N``, empty for one that runs once), ``model`` (its
        transport model), then the other columns of :attr:`receptors`. Rows
        run receptor by receptor, with variants in config order. Select rows as in pandas
        and pass the selection to :meth:`status`, :meth:`incomplete`,
        :meth:`particles`, :meth:`footprints`, or :meth:`jacobian`.

        Examples
        --------
        >>> sims = project.simulations
        >>> july = sims[(sims.variant == "hrrr") & (sims.site == "WBB")]
        >>> project.status(july)
        >>> project.footprints(july)
        """
        return self._simulations.copy()

    def simulation(
        self, receptor_id: str, variant: str, realization: int | None = None
    ) -> Simulation:
        """
        Return one simulation: a receptor under a variant, and a realization of an ensemble.

        A simulation is a value built from its receptor, its variant, and the
        output directory, so it is cheap. It knows where its results are and
        loads them.

        Raises
        ------
        KeyError
            If the project has no such receptor or variant.
        ValueError
            If *realization* is not one the variant runs as: ``0`` to
            ``N - 1`` for ``realizations: N``, ``None`` otherwise.
        """
        if variant not in self.variants:
            raise KeyError(
                f"No variant {variant!r}; the variants are {list(self.variants)}."
            )
        return Simulation(
            self.receptor(receptor_id),
            self.variants[variant],
            self.output,
            self.directory,
            realization,
        )

    def folders(self) -> pd.DataFrame:
        """
        Return every settings folder in the output directory, and which variants use it.

        A settings folder holds the results of one set of settings, named
        after the variant that first made them and a short hash of the
        settings (``settings=hrrr-b2399e``). A folder no variant uses comes
        from changed settings, a dropped variant, or another project sharing
        the output directory; PYSTILT never deletes one. ``stilt status``
        prints this table.

        Returns
        -------
        pandas.DataFrame
            ``kind`` (``particles`` or ``footprints``), ``folder`` (the
            ``settings=`` value), ``name`` (the variant that made it),
            ``files`` (the result files it holds), ``variant`` (the variants
            here that use it, empty when none does), and ``differs`` (how an
            unused folder's settings differ from the variant of its name).

        Examples
        --------
        >>> folders = project.folders()
        >>> folders[folders.variant == ""]  # results no variant here uses
        """
        return self.output.folders(self.variants)

    @cached_property
    def plot(self) -> ProjectPlotAccessor:
        """Plotting methods, such as ``project.plot.availability()``."""
        from stilt.visualization import ProjectPlotAccessor

        return ProjectPlotAccessor(self)

    # -- the simulations' results ----------------------------------------------

    def _selected(self, sel: Any = None) -> pd.DataFrame:
        """
        Return the rows of :attr:`simulations` that *sel* selects.

        *sel* is ``None`` (every simulation), a pandas table with
        ``receptor`` and ``variant`` columns (returned as it is, extra
        columns and all), a boolean mask over :attr:`simulations`, or a
        polars or pyarrow table with those two columns. A table without a
        ``realization`` column selects every realization of an ensemble.

        Raises
        ------
        ValueError
            If the table lacks the ``receptor`` or ``variant`` column.
        KeyError
            If it names a variant the project does not have.
        """
        if sel is None:
            return self.simulations
        if isinstance(sel, pd.DataFrame):
            frame = sel
        elif _is_mask(sel):
            frame = self.simulations.loc[np.asarray(sel, dtype=bool)]
        else:
            try:
                frame = pd.DataFrame(
                    {name: _column(sel, name) for name in ("receptor", "variant")}
                )
            except (KeyError, TypeError, ValueError, IndexError):
                frame = pd.DataFrame()
        missing = [c for c in ("receptor", "variant") if c not in frame.columns]
        if missing:
            raise ValueError(
                "A selection of simulations is a table with 'receptor' and "
                f"'variant' columns, or a mask over project.simulations; this "
                f"one has no {missing}."
            )
        unknown = sorted(set(frame["variant"]) - set(self.variants))
        if unknown:
            raise KeyError(
                f"No variant {unknown}; the variants are {list(self.variants)}."
            )
        if "realization" not in frame.columns:
            # Every realization of an ensemble, and the rest of each row.
            sims = self.simulations
            keep = [c for c in sims.columns if c not in frame.columns]
            frame = frame.merge(
                sims[["receptor", "variant", *keep]],
                on=["receptor", "variant"],
                how="left",
            )
        return frame

    def _present(self, frame: pd.DataFrame) -> dict[_Key, _Present]:
        """
        Return, per variant and realization, the selected receptors with particles and with a footprint.

        Read from a listing of the date folders the selection falls in
        (:meth:`stilt.output.Output.present`), in place of a file check per
        simulation: on a large project the checks are the slow part. The
        footprint set is ``None`` for a variant that makes no footprint.
        """
        present: dict[_Key, _Present] = {}
        for name, k, rows in _groups(frame):
            variant = self.variants[name]
            among = set(rows["receptor"])
            particles = self.output.present("particles", variant, among, k)
            footprints = None
            if variant.footprint is not None:
                footprints = self.output.present("footprints", variant, among, k)
            present[(name, k)] = (particles, footprints)
        return present

    @staticmethod
    def _complete(frame: pd.DataFrame, present: dict[_Key, _Present]) -> np.ndarray:
        """Return whether each row is complete (:func:`stilt.output.completed`), from :meth:`_present`."""
        done = {key: completed(*sets) for key, sets in present.items()}
        return np.array([r in done[(v, k)] for r, v, k in _rows(frame)], dtype=bool)

    def status(self, sel: Any = None) -> pd.DataFrame:
        """
        Return the selected simulations with what each has, and where it stands.

        Whether a result exists comes from a listing of the date folders, so
        no result file is opened; only the failure records of failed
        simulations are read.

        Parameters
        ----------
        sel : DataFrame or mask, optional
            The simulations: rows of :attr:`simulations`, any table with
            ``receptor`` and ``variant`` columns (pandas, polars, pyarrow),
            or a boolean mask over :attr:`simulations`. All of them by
            default.

        Returns
        -------
        pandas.DataFrame
            The selection with six more columns. ``particles`` and
            ``footprint`` are ``True`` when the result exists, ``False``
            when it is missing, and ``NA`` when the variant does not make
            it. ``state`` is ``complete`` when every expected result exists
            (:attr:`stilt.Simulation.is_complete`), ``failed`` when the last
            run failed, ``interrupted`` when a run of its particles started
            and stopped before it finished (a time limit, preemption, or a
            killed process), and ``pending`` when nothing has run. ``step``, ``reason``, and ``message`` say
            why a failed simulation failed (:attr:`stilt.Simulation.failure`),
            and are ``NA`` for the others.

        Notes
        -----
        An empty footprint (no particle reached the grid) is complete: its
        ``sim.footprint`` is ``None`` while ``sim.has_footprint`` is true,
        and :meth:`jacobian` lists the empty ones.

        Examples
        --------
        >>> st = project.status()
        >>> st.state.value_counts()
        >>> st[st.state == "failed"][["receptor", "variant", "reason"]]
        """
        frame = self._selected(sel)
        present = self._present(frame)
        rows = _rows(frame)
        particles = [r in present[(v, k)][0] for r, v, k in rows]
        feet = [
            None if (have := present[(v, k)][1]) is None else r in have
            for r, v, k in rows
        ]
        complete = self._complete(frame, present)
        found = self._failure_records(frame, present)
        records = [found.get(row, {}) for row in rows]
        started = self._started(frame, present, found)
        state = [
            "complete"
            if done
            else "failed"
            if record
            else "interrupted"
            if row in started
            else "pending"
            for done, record, row in zip(complete, records, rows, strict=True)
        ]
        return frame.assign(
            particles=pd.array(particles, dtype="boolean"),
            footprint=pd.array(feet, dtype="boolean"),
            state=pd.Categorical(state, categories=STATES),
            **{
                column: pd.array([r.get(column) for r in records], dtype="string")
                for column in ("step", "reason", "message")
            },
        )

    def _failure_records(
        self, frame: pd.DataFrame, present: dict[_Key, _Present]
    ) -> dict[tuple[str, str, int | None], dict[str, Any]]:
        """
        Return the failure record of each selected simulation missing a result, by ``(receptor, variant, realization)``.

        A simulation without particles is explained by its particles'
        record, one with particles but no footprint by its footprint's
        (:attr:`stilt.Simulation.failure`). Only those receptors' date
        folders are listed, and only the records found are read.
        """
        found: dict[tuple[str, str, int | None], dict[str, Any]] = {}
        for name, k, rows in _groups(frame):
            variant = self.variants[name]
            particles, feet = present[(name, k)]
            receptors = set(rows["receptor"])
            no_particles = receptors - particles
            no_footprint = set() if feet is None else (receptors & particles) - feet
            missing: list[tuple[Kind, set[str]]] = [
                ("particles", no_particles),
                ("footprints", no_footprint),
            ]
            for kind, among in missing:
                if among:
                    failures = self.output.failures(kind, variant, among, k)
                    for rid, record in failures.items():
                        found[(rid, name, k)] = record
        return found

    def _started(
        self,
        frame: pd.DataFrame,
        present: dict[_Key, _Present],
        failed: dict[tuple[str, str, int | None], dict[str, Any]],
    ) -> set[tuple[str, str, int | None]]:
        """
        Return the selected simulations whose particles run started and did not finish.

        That is a log with no particles and no failure record. Only the
        date folders of receptors without particles are listed.
        """
        started: set[tuple[str, str, int | None]] = set()
        for name, k, rows in _groups(frame):
            missing = set(rows["receptor"]) - present[(name, k)][0]
            if not missing:
                continue
            logged = self.output.logged(self.variants[name], missing, k)
            started |= {
                (rid, name, k) for rid in logged if (rid, name, k) not in failed
            }
        return started

    def incomplete(self, sel: Any = None) -> pd.DataFrame:
        """
        Return the selected simulations that are missing an expected result.

        The rows :meth:`status` does not mark ``complete``. It reads no
        failure record, so it is the quickest way to see what is left.
        *sel* is as for :meth:`status`.
        """
        frame = self._selected(sel)
        return frame.loc[~self._complete(frame, self._present(frame))]

    def particles(self, sel: Any = None) -> pd.DataFrame:
        """
        Load the particles of every selected simulation that has them, as one table.

        Parameters
        ----------
        sel : DataFrame or mask, optional
            The simulations, as for :meth:`status`. All of them by default.

        Returns
        -------
        pandas.DataFrame
            One row per particle per output step per simulation, with
            ``receptor``, ``variant``, and ``realization`` columns first. Simulations without
            particles are left out. Variants that share a run each get their
            own copy of its rows.

        Notes
        -----
        A run of 1,000 particles over 24 hours is about 1.4 million rows,
        some 250 MB in memory, so this suits a selection of tens. For a
        whole large project, read the ``particles/`` tree of the output
        directory with pyarrow, DuckDB, or polars instead.
        """
        frame = self._selected(sel)
        parts = []
        for name, k, rows in _groups(frame):
            variant = self.variants[name]
            present = self.output.present("particles", variant, rows["receptor"], k)
            table = self.output.table("particles", variant, present, k)
            if table.num_rows:
                part = particles_from_table(table).assign(variant=name)
                parts.append(
                    part.assign(realization=pd.array([k] * len(part), "Int64"))
                )
        if not parts:
            return pd.DataFrame(
                columns=pd.Index(["receptor", "variant", "realization"])
            )
        particles = pd.concat(parts, ignore_index=True)
        first = ["receptor", "variant", "realization"]
        return particles.loc[
            :, first + [c for c in particles.columns if c not in first]
        ]

    def footprints(self, sel: Any = None) -> xr.Dataset:
        """
        Open the footprints of the selected simulations as one dataset.

        Footprints are stacked by their hour from the receptor time, so
        receptors at any time share one ``hour`` axis; the ``time``
        coordinate says when each layer starts. Only the files' metadata is
        read here. The values load with dask when they are used, in blocks
        of receptors from one date folder. The selection must hold one
        variant, as for :meth:`jacobian`.

        Parameters
        ----------
        sel : DataFrame or mask, optional
            The simulations, as for :meth:`status`, all of one variant (and
            one realization of an ensemble). All of them by default.

        Returns
        -------
        xarray.Dataset
            ``foot`` with dims ``(receptor, hour, lat, lon)`` (``y`` and
            ``x`` on a projected grid), in selection order. A receptor with
            an empty footprint has no row and is listed in
            ``attrs["empty"]``; one not run yet is listed in
            ``attrs["missing"]``.

        Raises
        ------
        ValueError
            If the selection holds more or fewer than one variant, the
            variant has no grid, or no selected simulation has a footprint
            file yet.

        Notes
        -----
        A footprint is about 35 MB of values on a 350,000-cell grid over 24
        hours, so a computation that loads every receptor at once suits a
        selection of hundreds. Sums over ``hour`` or the grid run block by
        block. :meth:`jacobian` sums any number of footprints onto a target.

        Examples
        --------
        >>> ds = project.footprints(sims[sims.variant == "hrrr"])
        >>> ds.foot.sum("hour").mean("receptor").plot()
        """
        frame = self._selected(sel)
        name, k = _one_group(frame, "A footprint dataset")
        variant = self.variants[name]
        if variant.footprint is None:
            raise ValueError(f"Variant {name!r} makes no footprints (no grid).")
        requested = list(dict.fromkeys(frame["receptor"]))
        found = self.output.present("footprints", variant, requested, k)
        paths = [
            path
            for r in requested
            if r in found
            and (path := self.output.path("footprints", variant, r, k)) is not None
        ]
        if not paths:
            raise ValueError(
                f"None of the {len(requested)} selected simulations of {name!r} "
                "has a footprint yet."
            )
        ds = open_footprints(paths)
        ds.attrs["missing"] = [r for r in requested if r not in found]
        return ds

    def jacobian(
        self,
        sel: Any,
        target: Geometry,
        time_bins: pd.IntervalIndex,
        *,
        workers: int | None = None,
        batch: int = 64,
    ) -> Jacobian:
        """
        Sum the selected footprints onto a target, per time bin, as one sparse matrix.

        The same operation as ``foot.stilt.aggregate``, for every selected
        footprint at once. Each footprint cell is split among the target
        cells it overlaps in proportion to area, time layers are summed
        within each of *time_bins*, and cells or layers outside the target
        or the bins are dropped. The selection must hold one variant.

        Footprints are read and summed *batch* receptors at a time, in
        *workers* threads. Each thread holds one batch, so memory peaks at
        about *workers* times a batch whatever the selection's size; the
        result itself is sparse.

        Parameters
        ----------
        sel : DataFrame or mask
            The simulations, as for :meth:`status`, all of one variant.
        target : Geometry
            Where the fluxes are: a grid, mesh, or set of zones.
        time_bins : pandas.IntervalIndex
            Flux time bins, such as a flux inventory's steps. They must be
            closed on the left (``closed="left"``): each bin holds the
            footprint hours that start in it.
        workers : int, optional
            Threads that read and sum batches. Defaults to the CPUs this
            process may use (in a Slurm job, the job's), at most 8: more
            threads add memory and stop adding speed.
        batch : int, default 64
            Receptors read and summed together. At about 350,000 cells a
            footprint, a batch of 64 takes some 2 GB at its peak, so 8
            threads take some 16 GB.

        Returns
        -------
        Jacobian
            Rows are the selected receptors with a non-empty footprint, in
            selection order; columns are ``(time bin, target cell)``.
            Receptors with an empty footprint are listed in ``empty``, and
            those not run yet in ``missing``.

        Raises
        ------
        ValueError
            If the selection holds more or fewer than one variant, the
            variant has no grid or no footprints yet, or ``time_bins`` is not
            closed on the left.
        """
        frame = self._selected(sel)
        name, k = _one_group(frame, "A Jacobian")
        variant = self.variants[name]
        if variant.footprint is None:
            raise ValueError(f"Variant {name!r} makes no footprints (no grid).")
        if self.output.folder("footprints", variant) is None:
            raise ValueError(f"Variant {name!r} has no footprints yet.")
        requested = list(dict.fromkeys(frame["receptor"]))
        found = self.output.present("footprints", variant, requested, k)
        present = [r for r in requested if r in found]
        batches = [present[i : i + batch] for i in range(0, len(present), batch)]
        return _jacobian(
            lambda rows: self.output.table("footprints", variant, rows, k),
            batches,
            variant.footprint,
            target,
            time_bins,
            missing=[r for r in requested if r not in found],
            geometry_hash=variant.geometry_hash,
            workers=workers,
        )

    # -- running ---------------------------------------------------------------

    def run(
        self,
        skip_existing: bool = True,
        compute_root: str | Path | None = None,
        execution: ExecutionConfig | None = None,
        *,
        receptors: Iterable[str] | None = None,
        task: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        Run every simulation that has not finished, and wait for it.

        Shorthand for :func:`stilt.execution.run`, which documents the
        parameters. Use :meth:`submit` to submit to Slurm and return at once.
        """
        from stilt.execution import run

        return run(
            self,
            receptors=receptors,
            task=task,
            execution=execution,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )

    def submit(
        self,
        skip_existing: bool = True,
        compute_root: str | Path | None = None,
        execution: ExecutionConfig | None = None,
        *,
        receptors: Iterable[str] | None = None,
    ) -> str | None:
        """
        Submit every simulation that has not finished to Slurm, and return at once.

        Shorthand for :func:`stilt.execution.submit`, which documents the
        parameters. Returns the Slurm job id, or ``None`` when nothing needs
        to run.
        """
        from stilt.execution import submit

        return submit(
            self,
            receptors=receptors,
            execution=execution,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )


__all__ = ["Project"]
