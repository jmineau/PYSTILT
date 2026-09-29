"""
Collections of a model's receptors and simulations.

``model.simulations`` is a :class:`SimulationCollection` of every receptor
under every variant. :meth:`SimulationCollection.sel` narrows it, and its
``trajectories`` and ``footprint`` properties give an
:class:`OutputCollection` for loading one output across the selection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

from stilt.footprint import Footprint
from stilt.receptors import Receptor, read_receptors
from stilt.simulation import SimID, Simulation
from stilt.trajectory import Trajectories

if TYPE_CHECKING:
    from stilt.model import Model
    from stilt.project import Project

#: The two outputs a simulation can produce, as ``status()`` columns.
TRAJECTORY = "trajectory"
FOOTPRINT = "footprint"


class ReceptorCollection:
    """
    A model's receptors, by position or by receptor id.

    Index by position (``receptors[0]``, ``receptors[:3]``) or by id
    (``receptors["202307151800_-111.848_40.766_10"]``). :meth:`sel` filters.
    The receptors are loaded on first use.

    Parameters
    ----------
    receptors : Receptor, iterable of Receptor, str, Path or None
        The receptors, the path of a receptors CSV, or ``None`` to use the
        project's ``receptors.csv``.
    project : Project
        Project the receptors belong to. A relative CSV path is relative to
        its directory.
    """

    def __init__(
        self,
        receptors: Receptor | Iterable | str | Path | None,
        *,
        project: Project,
    ):
        self._items, self._source_path = self._normalize(receptors)
        self._project = project
        self._by_id: dict[str, Receptor] | None = None

    @staticmethod
    def _normalize(
        receptors: Receptor | Iterable | str | Path | None,
    ) -> tuple[list[Receptor] | None, Path | None]:
        """Split the constructor input into a receptor list or a CSV path."""
        if receptors is None:
            return None, None
        if isinstance(receptors, (str, Path)):
            return None, Path(receptors)
        if isinstance(receptors, Receptor):
            return [receptors], None
        if isinstance(receptors, Iterable):
            items = list(receptors)
            if all(isinstance(item, Receptor) for item in items):
                return items, None
        raise TypeError(
            "Receptors must be a Receptor, an iterable of Receptors, or a path to "
            "a receptors CSV."
        )

    @property
    def source_path(self) -> Path | None:
        """Absolute path of the receptors CSV given to the constructor, or ``None``."""
        if self._source_path is None:
            return None
        if self._source_path.is_absolute() or self._project.is_cloud:
            return self._source_path.resolve()
        return self._project.directory / self._source_path

    def _load(self) -> list[Receptor]:
        """Return the receptors, reading them on first use."""
        if self._items is not None:
            return self._items
        if self.source_path is not None:
            self._items = read_receptors(self.source_path)
            return self._items
        loaded = self._project.load_receptors()
        if loaded is None:
            raise FileNotFoundError(
                "No receptors available: no explicit receptors, no source path, "
                f"and no receptors.csv in {self._project.root}."
            )
        self._items = loaded
        return self._items

    @property
    def _data(self) -> dict[str, Receptor]:
        """Return a cached ``{receptor.id: receptor}`` mapping."""
        if self._by_id is None:
            self._by_id = {r.id: r for r in self._load()}
        return self._by_id

    def sel(
        self,
        *,
        time: slice | tuple | str | pd.Timestamp | None = None,
        location: str | Iterable[str] | None = None,
        where: Callable[[Receptor], bool] | None = None,
    ) -> ReceptorCollection:
        """
        Return the receptors that match every given filter.

        Parameters
        ----------
        time : slice, tuple, str or Timestamp, optional
            ``slice(start, stop)`` or ``(start, stop)`` with inclusive bounds
            (either may be ``None``), or a single time.
        location : str or iterable of str, optional
            One or more location ids.
        where : callable, optional
            Function that takes a :class:`~stilt.Receptor` and returns
            ``True`` to keep it.

        Returns
        -------
        ReceptorCollection
        """
        items = list(self._load())
        if time is not None:
            start, stop = _time_bounds(time)
            items = [r for r in items if start <= pd.Timestamp(r.time) <= stop]
        if location is not None:
            wanted = {location} if isinstance(location, str) else set(location)
            items = [r for r in items if r.location_id in wanted]
        if where is not None:
            items = [r for r in items if where(r)]
        return ReceptorCollection(items, project=self._project)

    @overload
    def __getitem__(self, item: int | str) -> Receptor: ...

    @overload
    def __getitem__(self, item: slice) -> list[Receptor]: ...

    def __getitem__(self, item: int | slice | str) -> Receptor | list[Receptor]:
        if isinstance(item, str):
            try:
                return self._data[item]
            except KeyError:
                raise KeyError(item) from None
        return self._load()[item]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._data
        return item in self._load()

    def __iter__(self) -> Iterator[Receptor]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


def _timestamp(value: object) -> pd.Timestamp:
    """Parse one time-selector bound; NaT is not a bound."""
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if not isinstance(ts, pd.Timestamp):
        raise ValueError(f"Not a time: {value!r}")
    return ts


def _time_bounds(
    time: slice | tuple | str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a time selector as inclusive ``(start, stop)`` timestamps."""
    if isinstance(time, slice):
        start, stop = time.start, time.stop
    elif isinstance(time, tuple):
        start, stop = time
    else:
        start = stop = time
    lo = _timestamp(start) if start is not None else pd.Timestamp.min
    hi = _timestamp(stop) if stop is not None else pd.Timestamp.max
    return lo, hi


class SimulationCollection:
    """
    An ordered selection of a model's simulations.

    ``model.simulations`` holds every receptor under every variant, receptor
    by receptor with variants in config order. Index it by
    ``(receptor_id, variant)`` or ``"<receptor_id>/<variant>"``. Iterating
    gives :class:`~stilt.Simulation` objects. :meth:`sel`, :meth:`incomplete`
    and :meth:`OutputCollection.missing` return smaller collections, so
    filters can be chained.

    Examples
    --------
    >>> sims = model.simulations.sel(variant="hrrr", time=("2023-07-01", "2023-07-31"))
    >>> sims.status()
    >>> feet = sims.footprint.load()
    """

    def __init__(self, model: Model, keys: list[SimID] | None = None):
        self._model = model
        self._keys = keys
        self._key_set: frozenset[SimID] | None = None

    # -- registered set --------------------------------------------------------

    def _all(self) -> list[SimID]:
        """Return the selected ids, receptor by receptor with variants in config order."""
        if self._keys is None:
            self._keys = [
                SimID(receptor.id, variant)
                for receptor in self._model.receptors
                for variant in self._model.variants
            ]
        return self._keys

    def _members(self) -> frozenset[SimID]:
        if self._key_set is None:
            self._key_set = frozenset(self._all())
        return self._key_set

    def keys(self) -> list[SimID]:
        """Return the selected simulation ids, receptor by receptor with variants in config order."""
        return list(self._all())

    def __iter__(self) -> Iterator[Simulation]:
        return (self._model.simulation(key) for key in self._all())

    def __len__(self) -> int:
        return len(self._all())

    def __contains__(self, key: object) -> bool:
        try:
            sid = SimID.parse(key)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return False
        return sid in self._members()

    def __getitem__(self, key: str | SimID | tuple[str, str]) -> Simulation:
        sid = SimID.parse(key)
        if sid not in self._members():
            raise KeyError(str(sid))
        return self._model.simulation(sid)

    @property
    def receptors(self) -> list[str]:
        """Receptor ids in the selection, in order, each once."""
        return list(dict.fromkeys(key.receptor for key in self._all()))

    @property
    def variants(self) -> list[str]:
        """Variant names in the selection, in order, each once."""
        return list(dict.fromkeys(key.variant for key in self._all()))

    # -- selection -------------------------------------------------------------

    def sel(
        self,
        *,
        receptor: str | Iterable[str] | None = None,
        variant: str | Iterable[str] | None = None,
        time: slice | tuple | str | pd.Timestamp | None = None,
        location: str | Iterable[str] | None = None,
        where: Callable[[Receptor], bool] | None = None,
    ) -> SimulationCollection:
        """
        Return the simulations that match every given filter.

        Parameters
        ----------
        receptor : str or iterable of str, optional
            One or more receptor ids.
        variant : str or iterable of str, optional
            One or more variant names. A realization group's name
            (``hrrr-err``) selects all its realizations.
        time, location, where
            Receptor filters, as in :meth:`ReceptorCollection.sel`.

        Returns
        -------
        SimulationCollection

        Raises
        ------
        KeyError
            If a receptor id or variant name is not in the model. The other
            filters raise nothing when they match no simulation.
        """
        keys = self._all()
        if receptor is not None:
            wanted = {receptor} if isinstance(receptor, str) else set(receptor)
            unknown = wanted - {r.id for r in self._model.receptors}
            if unknown:
                raise KeyError(f"Unknown receptor id(s): {sorted(unknown)}")
            keys = [k for k in keys if k.receptor in wanted]
        if variant is not None:
            wanted = {variant} if isinstance(variant, str) else set(variant)
            groups = {name: v.group for name, v in self._model.variants.items()}
            unknown = wanted - set(groups) - set(groups.values())
            if unknown:
                raise KeyError(f"Unknown variant(s): {sorted(unknown)}")
            keys = [
                k
                for k in keys
                if k.variant in wanted or groups.get(k.variant) in wanted
            ]
        if time is not None or location is not None or where is not None:
            ids = {
                r.id
                for r in self._model.receptors.sel(
                    time=time, location=location, where=where
                )
            }
            keys = [k for k in keys if k.receptor in ids]
        return SimulationCollection(self._model, keys)

    # -- completion ------------------------------------------------------------

    def incomplete(self) -> SimulationCollection:
        """Return the simulations that are missing an expected output."""
        return SimulationCollection(
            self._model, [sim.id for sim in self if not sim.is_complete()]
        )

    def status(self) -> pd.DataFrame:
        """
        Return one row per simulation saying which outputs exist.

        Returns
        -------
        pandas.DataFrame
            Columns ``receptor``, ``variant``, ``trajectory``, ``footprint``,
            and ``complete``. ``trajectory`` and ``footprint`` are ``True``
            when the output exists, ``False`` when it is missing, and ``NA``
            when the simulation does not make it. ``complete`` is
            :meth:`~stilt.Simulation.is_complete`.
        """
        rows = [
            {
                "receptor": str(sim.id.receptor),
                "variant": sim.variant,
                TRAJECTORY: sim.has_trajectory if not sim.is_derived else pd.NA,
                FOOTPRINT: sim.has_footprint if sim.makes_footprint else pd.NA,
                "complete": sim.is_complete(),
            }
            for sim in self
        ]
        columns = ["receptor", "variant", TRAJECTORY, FOOTPRINT, "complete"]
        return pd.DataFrame(rows, columns=pd.Index(columns)).astype(
            {TRAJECTORY: "boolean", FOOTPRINT: "boolean", "complete": "bool"}
        )

    # -- outputs ---------------------------------------------------------------

    @property
    def trajectories(self) -> OutputCollection:
        """Trajectories of the selected simulations that run HYSPLIT."""
        return OutputCollection(self, TRAJECTORY)

    @property
    def footprint(self) -> OutputCollection:
        """Footprints of the selected simulations that make one."""
        return OutputCollection(self, FOOTPRINT)


class OutputCollection:
    """
    One output, ``trajectory`` or ``footprint``, across a selection of simulations.

    Only simulations that make the output are included. A derived variant
    has no trajectory of its own, and a variant without a grid has no
    footprint. Get one from ``model.simulations.trajectories`` or
    ``model.simulations.footprint``.
    """

    def __init__(self, simulations: SimulationCollection, output: str):
        if output not in (TRAJECTORY, FOOTPRINT):
            raise ValueError(f"Unknown output {output!r}")
        self._sims = simulations
        self.output = output

    def _producers(self) -> list[Simulation]:
        """Return the simulations that make this output."""
        if self.output == TRAJECTORY:
            return [sim for sim in self._sims if not sim.is_derived]
        return [sim for sim in self._sims if sim.makes_footprint]

    def _exists(self, sim: Simulation) -> bool:
        return sim.has_trajectory if self.output == TRAJECTORY else sim.has_footprint

    def _path(self, sim: Simulation) -> Path:
        return (
            sim.trajectories_path if self.output == TRAJECTORY else sim.footprint_path
        )

    def paths(self) -> dict[SimID, Path]:
        """
        Return local paths of the output files that exist, by simulation id.

        Files in a remote store are downloaded first.
        """
        found = ((sim.id, sim.resolve(self._path(sim))) for sim in self._producers())
        return {sid: path for sid, path in found if path is not None}

    def load(self) -> dict[SimID, Trajectories] | dict[SimID, Footprint]:
        """
        Load every output that exists, by simulation id.

        Returns
        -------
        dict
            :class:`~stilt.Trajectories` or :class:`~stilt.Footprint`
            objects keyed by :class:`~stilt.SimID`.
        """
        if self.output == TRAJECTORY:
            return {
                sid: Trajectories.from_parquet(p) for sid, p in self.paths().items()
            }
        return {sid: Footprint.from_netcdf(p) for sid, p in self.paths().items()}

    def missing(self) -> SimulationCollection:
        """Return the simulations whose output does not exist yet."""
        return SimulationCollection(
            self._sims._model,
            [sim.id for sim in self._producers() if not self._exists(sim)],
        )

    def __len__(self) -> int:
        return len(self._producers())


__all__ = ["OutputCollection", "ReceptorCollection", "SimulationCollection"]
