"""Science-facing collection objects for STILT models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, overload

from stilt.config import STILTParams
from stilt.footprint import Footprint
from stilt.index import OutputSummary, SimulationIndex
from stilt.meteorology import MetSource
from stilt.receptor import Receptor, read_receptors
from stilt.selection import (
    filter_ids,
    matching_ids,
    missing_ids,
    output_paths,
    resolve_mets,
)
from stilt.simulation import SimID, Simulation
from stilt.storage import (
    ProjectFiles,
    Storage,
    Store,
)
from stilt.trajectory import Trajectories

if TYPE_CHECKING:
    from stilt.model import Model

TOutput = TypeVar("TOutput", covariant=True)


class ReceptorCollection:
    """Sequence of receptors with positional and receptor-id access.

    Access by position (``receptors[0]``, ``receptors[:3]``) or by
    receptor identifier (``receptors[sim_id.receptor_id]``), symmetric with
    the ``mets`` mapping.
    """

    def __init__(
        self,
        receptors: Receptor | Iterable | str | Path | None,
        *,
        storage: Storage,
    ):
        self._items, self._source_path = self._normalize(receptors)
        self._storage = storage
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
            if not items:
                return [], None
            if all(isinstance(item, Receptor) for item in items):
                return list(items), None
            if len(items) == 4 and not any(
                isinstance(item, Iterable) and not isinstance(item, (str, bytes))
                for item in items
            ):
                return [Receptor(*items)], None
            if all(
                isinstance(item, Iterable) and not isinstance(item, (str, bytes))
                for item in items
            ):
                return [
                    item if isinstance(item, Receptor) else Receptor(*item)
                    for item in items
                ], None
            raise TypeError(
                "Receptors must be a Receptor, a path, or an iterable of Receptor "
                "instances / (time, longitude, latitude, altitude) tuples."
            )
        raise TypeError(
            "Receptors must be a Receptor, a path, or an iterable of Receptor "
            "instances / (time, longitude, latitude, altitude) tuples."
        )

    def _load(self) -> list[Receptor]:
        """Load and cache receptors from the best available durable source."""
        if self._items is not None:
            return self._items
        if self.source_path is not None:
            self._items = read_receptors(self.source_path)
            return self._items
        loaded = self._storage.load_receptors()
        if loaded is None:
            raise FileNotFoundError(
                "No receptors available: no explicit receptors, no source path, "
                "and no receptors.csv in the project root or durable store."
            )
        self._items = loaded
        return self._items

    @property
    def source_path(self) -> Path | None:
        """Return the original constructor-supplied receptors path, if any."""
        if self._source_path is None:
            return None
        if self._source_path.is_absolute():
            return self._source_path
        if self._storage.is_cloud_project:
            return self._source_path.resolve()
        return self._storage.project_dir / self._source_path

    @property
    def _data(self) -> dict[str, Receptor]:
        """Return a cached ``{receptor.id: receptor}`` mapping."""
        if self._by_id is None:
            self._by_id = {r.id: r for r in self._load()}
        return self._by_id

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


class SimulationCollection:
    """Lazy simulation collection backed by the durable project index.

    This object is the science-facing boundary for simulation identity,
    selection, and lazy handle construction.
    """

    def __init__(
        self,
        output_dir: Path,
        params: STILTParams,
        mets: dict[str, MetSource],
        receptors: ReceptorCollection,
        footprint_names: list[str],
        index: SimulationIndex,
        store: Store,
    ):
        self._output_dir = output_dir
        self._params = params
        self._mets = mets
        self._receptors = receptors
        self._footprint_names = footprint_names
        self._index = index
        self._store = store
        self._cache: dict[str, Simulation] = {}

    def __getitem__(self, sim_id: str) -> Simulation:
        if sim_id not in self._cache:
            self._cache[sim_id] = self._build(SimID(sim_id))
        return self._cache[sim_id]

    def __contains__(self, sim_id: str) -> bool:
        return self._index.has(sim_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index.sim_ids())

    def __len__(self) -> int:
        return self._index.count()

    def keys(self) -> list[str]:
        """Return all registered simulation identifiers."""
        return self._index.sim_ids()

    def items(self) -> Iterator[tuple[str, Simulation]]:
        """Yield ``(sim_id, Simulation)`` pairs for all registered simulations."""
        return ((sid, self[sid]) for sid in self)

    def values(self) -> Iterator[Simulation]:
        """Yield :class:`Simulation` objects for all registered simulations."""
        return (self[sid] for sid in self)

    def ids(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return simulation identifiers matching the given filters."""
        filtered = filter_ids(
            self._index.sim_ids(),
            mets=self._resolve_mets(mets),
            time_range=time_range,
            location_ids=location_ids,
        )

        if footprint is None or not filtered:
            return filtered

        found = self._index.summaries(filtered)
        return [
            sim_id
            for sim_id in filtered
            if found.get(sim_id, OutputSummary()).footprints_complete([footprint])
        ]

    def select(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Simulation]:
        """Return simulation handles matching the given filters."""
        return [
            self[sim_id]
            for sim_id in self.ids(
                mets=mets,
                footprint=footprint,
                time_range=time_range,
                location_ids=location_ids,
            )
        ]

    def incomplete(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return simulation IDs not fully complete for the current model config."""
        candidate_ids = matching_ids(
            self._index,
            receptors=self._receptors,
            configured_mets=self._mets,
            registered=False,
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )
        found = self._index.summaries(candidate_ids)
        return [
            sim_id
            for sim_id in candidate_ids
            if found.get(sim_id, OutputSummary()).needs_work(
                self._footprint_names,
                skip_existing=True,
            )
        ]

    def _resolve_mets(self, mets: str | list[str] | None) -> set[str]:
        """Resolve an optional met-name filter against configured streams."""
        return resolve_mets(self._mets, mets)

    def _build(self, sim_id: SimID) -> Simulation:
        receptor = self._receptors[sim_id.receptor]
        sim_dir = ProjectFiles(self._output_dir).simulation(str(sim_id)).directory
        return Simulation(
            directory=sim_dir,
            receptor=receptor,
            params=self._params,
            meteorology=self._mets[sim_id.met],
            store=self._store,
        )


class _OutputSpec(Protocol[TOutput]):
    """Typed contract for one durable simulation output family."""

    def present(self, summary: OutputSummary) -> bool:
        """Return whether this output is complete in one index summary."""
        ...

    def local_path(self, model: Model, sim_id: str) -> Path:
        """Return this output's project-local path for one simulation id."""
        ...

    def load_one(self, path: Path) -> TOutput:
        """Load one resolved local output path."""
        ...


class _TrajectoryOutputSpec:
    """Output spec for main or error trajectory parquet files."""

    def __init__(self, *, error: bool = False):
        self.error = error

    def present(self, summary: OutputSummary) -> bool:
        return summary.error_traj_present if self.error else summary.traj_present

    def local_path(self, model: Model, sim_id: str) -> Path:
        sim_files = ProjectFiles(model.layout.output_dir).simulation(sim_id)
        return (
            sim_files.error_trajectory_path if self.error else sim_files.trajectory_path
        )

    def load_one(self, path: Path) -> Trajectories:
        return Trajectories.from_parquet(path)


class _NamedFootprintOutputSpec:
    """Output spec for one named footprint netCDF file."""

    def __init__(self, name: str):
        self.name = name

    def present(self, summary: OutputSummary) -> bool:
        return summary.footprint_complete(self.name)

    def local_path(self, model: Model, sim_id: str) -> Path:
        return (
            ProjectFiles(model.layout.output_dir)
            .simulation(sim_id)
            .footprint_path(self.name)
        )

    def load_one(self, path: Path) -> Footprint:
        return Footprint.from_netcdf(path)


class _OutputAccessor(Generic[TOutput]):
    """Shared implementation of cross-simulation output accessors.

    Subclasses/factories supply a typed output spec and inherit ``paths()`` /
    ``load()`` / ``missing()`` with consistent filter semantics.
    """

    def __init__(
        self,
        model: Model,
        spec: _OutputSpec[TOutput],
    ):
        self._model = model
        self._spec = spec

    def _configured_mets(self) -> Iterable[str] | None:
        return self._model.config.mets if self._model._config is not None else None

    def _matching_ids(
        self,
        *,
        registered: bool,
        mets: str | list[str] | None,
        time_range: tuple | None,
        location_ids: set[str] | None,
    ) -> list[str]:
        return matching_ids(
            self._model.index,
            receptors=self._model.receptors,
            configured_mets=self._configured_mets(),
            registered=registered,
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )

    def paths(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Path]:
        return output_paths(
            self._model.storage,
            self._model.index,
            self._matching_ids(
                registered=True,
                mets=mets,
                time_range=time_range,
                location_ids=location_ids,
            ),
            present=self._spec.present,
            local_path=lambda sim_id: self._spec.local_path(self._model, sim_id),
        )

    def load(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[TOutput]:
        return [
            self._spec.load_one(path)
            for path in self.paths(
                mets=mets, time_range=time_range, location_ids=location_ids
            )
        ]

    def missing(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        return missing_ids(
            self._model.index,
            self._matching_ids(
                registered=False,
                mets=mets,
                time_range=time_range,
                location_ids=location_ids,
            ),
            present=self._spec.present,
        )


class TrajectoryCollection:
    """Science-facing accessor for trajectory outputs across simulations."""

    def __init__(self, model: Model):
        self._model = model
        self._main = _OutputAccessor(model, _TrajectoryOutputSpec())
        self._error = _OutputAccessor(model, _TrajectoryOutputSpec(error=True))

    def _accessor(self, error: bool) -> _OutputAccessor[Trajectories]:
        return self._error if error else self._main

    def paths(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Path]:
        """Return local-accessible paths for matching trajectory outputs."""
        return self._accessor(error).paths(
            mets=mets, time_range=time_range, location_ids=location_ids
        )

    def load(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Trajectories]:
        """Load trajectories across matching simulations."""
        return self._accessor(error).load(
            mets=mets, time_range=time_range, location_ids=location_ids
        )

    def missing(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return simulation IDs missing completed trajectory output."""
        return self._main.missing(
            mets=mets, time_range=time_range, location_ids=location_ids
        )


class NamedFootprintCollection(_OutputAccessor[Footprint]):
    """Science-facing accessor for one named footprint output across simulations."""

    def __init__(self, model: Model, name: str):
        self.name = name
        super().__init__(model, _NamedFootprintOutputSpec(name))

    def load(  # type: ignore[override]
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Footprint]:
        """Load this named footprint across matching simulations."""
        return super().load(mets=mets, time_range=time_range, location_ids=location_ids)


class FootprintCollection:
    """Science-facing namespace for named footprint outputs."""

    def __init__(self, model: Model):
        self._model = model
        self._cache: dict[str, NamedFootprintCollection] = {}

    def __getitem__(self, name: str) -> NamedFootprintCollection:
        if name not in self._cache:
            self._cache[name] = NamedFootprintCollection(self._model, name)
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def names(self) -> list[str]:
        """Return configured footprint output names for this model."""
        try:
            return list(self._model.config.footprints)
        except FileNotFoundError:
            found = self._model.index.summaries()
            return sorted(
                {name for summary in found.values() for name in summary.footprints}
            )
