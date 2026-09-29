"""
Science-facing collection objects for STILT models.

``SimulationCollection`` is the one query surface: an ordered selection over
the registered set ``receptors × variants``, narrowed by :meth:`sel`. Every
cross-simulation question (which are complete, which outputs exist, load
them all) is answered by asking each :class:`~stilt.simulation.Simulation`
handle. ``OutputCollection`` is the view of one output over a selection.
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
    Sequence of receptors with positional and receptor-id access.

    Access by position (``receptors[0]``, ``receptors[:3]``) or by receptor
    identifier (``receptors[sim_id.receptor]``). :meth:`sel` narrows.

    Parameters
    ----------
    receptors
        A receptor, an iterable of receptors or ``(time, lon, lat, alt)``
        tuples, a path to a receptors CSV, or ``None`` to load the project's
        ``receptors.csv`` lazily.
    project
        Project the receptors belong to; used to resolve relative paths and
        to load ``receptors.csv`` when nothing explicit was given.
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
        """Normalize receptor inputs to either an in-memory list or a source path."""
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
        """Return the constructor-supplied receptors path, resolved, if any."""
        if self._source_path is None:
            return None
        if self._source_path.is_absolute() or self._project.is_cloud:
            return self._source_path.resolve()
        return self._project.directory / self._source_path

    def _load(self) -> list[Receptor]:
        """Load and cache receptors from the best available source."""
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
        Return the receptors matching every given filter.

        Parameters
        ----------
        time
            A ``slice(start, stop)`` or ``(start, stop)`` pair of inclusive
            bounds, or one timestamp.
        location
            One location id or several.
        where
            Predicate on the :class:`~stilt.Receptor`.
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
    """Normalise a time selector into inclusive ``(start, stop)`` timestamps."""
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
    An ordered selection of a model's simulations (``receptors × variants``).

    ``model.simulations`` is the whole registered set; :meth:`sel`,
    :meth:`incomplete` and :meth:`OutputCollection.missing` return narrower
    collections, so filters compose. Handles are built lazily and cached on
    the model; building one has no side effects on disk.
    """

    def __init__(self, model: Model, keys: list[SimID] | None = None):
        self._model = model
        self._keys = keys
        self._key_set: frozenset[SimID] | None = None

    # -- registered set --------------------------------------------------------

    def _all(self) -> list[SimID]:
        """The selected ids, built once: receptor-major, variants in config order."""
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
        """The selected simulation ids, receptor-major, variants in config order."""
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
        """Receptor ids in the selection, in order, without repeats."""
        return list(dict.fromkeys(key.receptor for key in self._all()))

    @property
    def variants(self) -> list[str]:
        """Variant names in the selection, in order, without repeats."""
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
        Narrow the selection.

        Parameters
        ----------
        receptor
            One receptor id or several.
        variant
            One variant name or several. A realization group's name
            (``hrrr-err``) selects every realization.
        time, location, where
            Receptor filters, as :meth:`ReceptorCollection.sel`.

        Raises
        ------
        KeyError
            For a receptor id or variant name the model does not define. The
            other filters may legitimately select nothing.
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
        """The simulations that have not produced every expected output."""
        return SimulationCollection(
            self._model, [sim.id for sim in self if not sim.is_complete()]
        )

    def status(self) -> pd.DataFrame:
        """
        One row per simulation with a column per output.

        ``trajectory`` and ``footprint`` are ``True`` when the output exists,
        ``False`` when it is expected and missing, and ``NA`` when the
        simulation does not produce it. ``complete`` is the completion rule.
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
        """The trajectories of the selected simulations that run HYSPLIT."""
        return OutputCollection(self, TRAJECTORY)

    @property
    def footprint(self) -> OutputCollection:
        """The footprints of the selected simulations that produce one."""
        return OutputCollection(self, FOOTPRINT)


class OutputCollection:
    """
    One output (``trajectory`` or ``footprint``) over a simulation selection.

    Covers only the simulations that produce the output: derived variants
    have no trajectory of their own, and a variant without a grid has no
    footprint.
    """

    def __init__(self, simulations: SimulationCollection, output: str):
        if output not in (TRAJECTORY, FOOTPRINT):
            raise ValueError(f"Unknown output {output!r}")
        self._sims = simulations
        self.output = output

    def _producers(self) -> list[Simulation]:
        """Simulations for which this output is expected."""
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
        Local paths of the outputs that exist, keyed by simulation id.

        Empty footprints have no file, so they are not listed here; they still
        count as complete (see :meth:`missing`).
        """
        found = ((sim.id, sim.resolve(self._path(sim))) for sim in self._producers())
        return {sid: path for sid, path in found if path is not None}

    def load(self) -> dict[SimID, Trajectories] | dict[SimID, Footprint]:
        """Load every existing output, keyed by simulation id."""
        if self.output == TRAJECTORY:
            return {
                sid: Trajectories.from_parquet(p) for sid, p in self.paths().items()
            }
        return {sid: Footprint.from_netcdf(p) for sid, p in self.paths().items()}

    def missing(self) -> SimulationCollection:
        """The producing simulations whose output does not exist yet."""
        return SimulationCollection(
            self._sims._model,
            [sim.id for sim in self._producers() if not self._exists(sim)],
        )

    def __len__(self) -> int:
        return len(self._producers())


__all__ = ["OutputCollection", "ReceptorCollection", "SimulationCollection"]
