"""
Science-facing collection objects for STILT models.

``SimulationCollection`` is the one query surface: the registered set is
``receptors × mets``, every filter lives here, and every cross-simulation
question (which are complete, which paths exist, load them all) is answered
by asking each :class:`~stilt.simulation.Simulation` handle. The trajectory
and footprint collections are thin views over it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import pandas as pd

from stilt.errors import ConfigValidationError
from stilt.footprint import Footprint
from stilt.receptors import PointReceptor, Receptor, read_receptors
from stilt.simulation import ERROR_TRAJECTORY, TRAJECTORY, SimID, Simulation
from stilt.trajectory import Trajectories

if TYPE_CHECKING:
    from stilt.model import Model
    from stilt.project import Project


class ReceptorCollection:
    """
    Sequence of receptors with positional and receptor-id access.

    Access by position (``receptors[0]``, ``receptors[:3]``) or by receptor
    identifier (``receptors[sim_id.receptor]``).

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
            if not items:
                return [], None
            if all(isinstance(item, Receptor) for item in items):
                return list(items), None
            if len(items) == 4 and not any(
                isinstance(item, Iterable) and not isinstance(item, (str, bytes))
                for item in items
            ):
                return [PointReceptor(*items)], None
            if all(
                isinstance(item, Iterable) and not isinstance(item, (str, bytes))
                for item in items
            ):
                return [
                    item if isinstance(item, Receptor) else PointReceptor(*item)
                    for item in items
                ], None
        raise TypeError(
            "Receptors must be a receptor, a path, or an iterable of receptor "
            "instances / (time, longitude, latitude, altitude) tuples."
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


def _time_bounds(
    time_range: tuple | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Normalise a time range into a pair of timestamps."""
    if time_range is None:
        return None
    return cast(pd.Timestamp, pd.Timestamp(time_range[0])), cast(
        pd.Timestamp, pd.Timestamp(time_range[1])
    )


class SimulationCollection:
    """
    The simulations a model defines: its receptors crossed with its met streams.

    Mapping-like over simulation ids. Handles are built lazily and cached, and
    building one has no side effects on disk.
    """

    def __init__(self, model: Model):
        self._model = model
        self._cache: dict[str, Simulation] = {}

    # -- registered set --------------------------------------------------------

    def _pairs(self) -> Iterator[tuple[SimID, Receptor]]:
        """Yield every (simulation id, receptor) pair, receptors times mets."""
        for met in self._model.mets:
            for receptor in self._model.receptors:
                yield SimID.from_parts(met, receptor), receptor

    def keys(self) -> list[str]:
        """Return every simulation id (receptors × mets), sorted."""
        return sorted(str(sim_id) for sim_id, _ in self._pairs())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._model.mets) * len(self._model.receptors)

    def __contains__(self, sim_id: object) -> bool:
        if not isinstance(sim_id, str):
            return False
        try:
            sid = SimID(sim_id)
        except ValueError:
            return False
        return sid.met in self._model.mets and sid.receptor in self._model.receptors

    def __getitem__(self, sim_id: str) -> Simulation:
        if sim_id not in self._cache:
            self._cache[sim_id] = self._model.simulation(sim_id)
        return self._cache[sim_id]

    def items(self) -> Iterator[tuple[str, Simulation]]:
        """Yield ``(sim_id, Simulation)`` pairs."""
        return ((sid, self[sid]) for sid in self)

    def values(self) -> Iterator[Simulation]:
        """Yield :class:`Simulation` handles."""
        return (self[sid] for sid in self)

    # -- filtering -------------------------------------------------------------

    def _resolve_mets(self, mets: str | list[str] | None) -> set[str]:
        """Resolve a met-name filter to a set of configured met streams."""
        available = set(self._model.mets)
        if mets is None:
            return available
        requested = {mets} if isinstance(mets, str) else set(mets)
        missing = sorted(requested - available)
        if missing:
            raise ConfigValidationError(f"Unknown met name(s): {missing}")
        return requested

    def ids(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Return simulation ids matching the filters.

        Parameters
        ----------
        mets
            Met stream name(s) to include. All configured streams by default.
        footprint
            When given, keep only simulations whose named footprint is complete.
        time_range
            ``(start, end)`` receptor-time bounds, inclusive.
        location_ids
            Receptor location ids to include.
        """
        wanted_mets = self._resolve_mets(mets)
        bounds = _time_bounds(time_range)
        out: list[str] = []
        for sid, receptor in self._pairs():
            if sid.met not in wanted_mets:
                continue
            if bounds is not None:
                t = pd.Timestamp(receptor.time)
                if t < bounds[0] or t > bounds[1]:
                    continue
            if location_ids is not None and sid.location not in location_ids:
                continue
            if footprint is not None and not self[sid].has_footprint(footprint):
                continue
            out.append(str(sid))
        return sorted(out)

    def select(
        self,
        mets: str | list[str] | None = None,
        footprint: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Simulation]:
        """Return simulation handles matching the filters (see :meth:`ids`)."""
        return [
            self[sid]
            for sid in self.ids(
                mets=mets,
                footprint=footprint,
                time_range=time_range,
                location_ids=location_ids,
            )
        ]

    # -- completion ------------------------------------------------------------

    def incomplete(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Return ids of simulations that have not produced every configured output.

        Completion is decided by the outputs on disk (by key): the trajectory,
        the error trajectory when wind-error params are set, and every
        footprint in ``config.footprints``.
        """
        footprints = list(self._model.config.footprints)
        return [
            sid
            for sid in self.ids(
                mets=mets, time_range=time_range, location_ids=location_ids
            )
            if not self[sid].is_complete(footprints)
        ]

    def missing(
        self,
        output: str,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return ids of matching simulations lacking one named output."""
        return [
            sid
            for sid in self.ids(
                mets=mets, time_range=time_range, location_ids=location_ids
            )
            if not self[sid].has_output(output)
        ]

    def paths(
        self,
        output: str,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Path]:
        """
        Return local paths of one named output across matching simulations.

        Only outputs that exist are returned; for footprints, empty markers
        are skipped because there is no file to load.
        """
        out: list[Path] = []
        for sim in self.select(
            mets=mets, time_range=time_range, location_ids=location_ids
        ):
            path = sim.resolve(_output_path(sim, output))
            if path is not None:
                out.append(path)
        return out


def _output_path(sim: Simulation, output: str) -> Path:
    """Return the path of one of a simulation's named outputs."""
    if output == TRAJECTORY:
        return sim.trajectories_path
    if output == ERROR_TRAJECTORY:
        return sim.error_trajectories_path()
    return sim.footprint_path(output)


class TrajectoryCollection:
    """Cross-simulation accessor for trajectory parquet outputs."""

    def __init__(self, model: Model):
        self._sims = model.simulations

    def paths(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Path]:
        """Return local paths of existing trajectory (or error-trajectory) files."""
        return self._sims.paths(
            ERROR_TRAJECTORY if error else TRAJECTORY,
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )

    def load(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        *,
        error: bool = False,
    ) -> list[Trajectories]:
        """Load matching trajectories."""
        return [
            Trajectories.from_parquet(p)
            for p in self.paths(
                mets=mets, time_range=time_range, location_ids=location_ids, error=error
            )
        ]

    def missing(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return ids of matching simulations without a trajectory."""
        return self._sims.missing(
            TRAJECTORY, mets=mets, time_range=time_range, location_ids=location_ids
        )


class NamedFootprintCollection:
    """Cross-simulation accessor for one named footprint output."""

    def __init__(self, model: Model, name: str):
        self.name = name
        self._sims = model.simulations

    def paths(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Path]:
        """Return local paths of existing (non-empty) footprint files."""
        return self._sims.paths(
            self.name, mets=mets, time_range=time_range, location_ids=location_ids
        )

    def load(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Footprint]:
        """Load matching footprints."""
        return [
            Footprint.from_netcdf(p)
            for p in self.paths(
                mets=mets, time_range=time_range, location_ids=location_ids
            )
        ]

    def missing(
        self,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[str]:
        """Return ids of matching simulations whose footprint is not complete."""
        return self._sims.missing(
            self.name, mets=mets, time_range=time_range, location_ids=location_ids
        )


class FootprintCollection:
    """Namespace of named footprint accessors: ``model.footprints["slv"]``."""

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
        """Return the configured footprint names."""
        return list(self._model.config.footprints)


__all__ = [
    "FootprintCollection",
    "NamedFootprintCollection",
    "ReceptorCollection",
    "SimulationCollection",
    "TrajectoryCollection",
]
