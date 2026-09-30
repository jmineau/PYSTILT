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
from typing import TYPE_CHECKING, Any, overload

import pandas as pd

from stilt.footprint import Footprint
from stilt.geometry import Geometry
from stilt.output import Jacobian
from stilt.receptors import (
    Receptor,
    check_distinct_ids,
    read_receptors,
    receptors_to_frame,
)
from stilt.simulation import SimID, Simulation
from stilt.trajectory import Trajectories

if TYPE_CHECKING:
    from stilt.model import Model
    from stilt.project import Project

#: The two outputs a simulation can produce, as ``status()`` columns.
TRAJECTORY = "trajectory"
FOOTPRINT = "footprint"

#: Selections this large are checked by listing the result folders once.
_LIST_FROM = 32


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
        self._project = project
        self._items = self._normalize(receptors)
        self._from_project = receptors is None
        self._by_id: dict[str, Receptor] | None = None

    def _normalize(
        self, receptors: Receptor | Iterable | str | Path | None
    ) -> list[Receptor] | None:
        """Return the constructor input as a receptor list, reading a CSV path."""
        if receptors is None:
            return None
        if isinstance(receptors, (str, Path)):
            path = Path(receptors)
            if not path.is_absolute():
                path = self._project.directory / path
            return read_receptors(path)
        if isinstance(receptors, Receptor):
            return [receptors]
        if isinstance(receptors, Iterable):
            items = list(receptors)
            if all(isinstance(item, Receptor) for item in items):
                check_distinct_ids(items)
                return items
        raise TypeError(
            "Receptors must be a Receptor, an iterable of Receptors, or a path to "
            "a receptors CSV."
        )

    @property
    def from_project(self) -> bool:
        """Whether these are the project's own ``receptors.csv``, rather than given."""
        return self._from_project

    def _load(self) -> list[Receptor]:
        """Return the receptors, reading the project's ``receptors.csv`` on first use."""
        if self._items is not None:
            return self._items
        loaded = self._project.load_receptors()
        if loaded is None:
            raise FileNotFoundError(
                "No receptors available: none were given and there is no "
                f"receptors.csv in {self._project.directory}."
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
        **labels: Any,
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
        **labels
            Values of the receptors' ``attrs`` to match, such as
            ``site="WBB"``. A list keeps any of its values.

        Returns
        -------
        ReceptorCollection

        Examples
        --------
        >>> model.receptors.sel(time=("2024-07-01", "2024-07-02"), site="WBB")
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
        for label, value in labels.items():
            wanted = set(value) if isinstance(value, (list, tuple, set)) else {value}
            items = [r for r in items if r.attrs.get(label) in wanted]
        return ReceptorCollection(items, project=self._project)

    def to_frame(self) -> pd.DataFrame:
        """
        Return the receptors as one table with a row per release point.

        See :func:`stilt.receptors.receptors_to_frame` for the columns.
        """
        return receptors_to_frame(self._load())

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
    gives :class:`~stilt.Simulation` objects. :meth:`sel` and
    :meth:`incomplete` return smaller collections, so filters can be chained.

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
        **labels: Any,
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
        time, location, where, **labels
            Receptor filters, as in :meth:`ReceptorCollection.sel`. A
            label such as ``site="WBB"`` matches the receptors' ``attrs``.

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
        if time is not None or location is not None or where is not None or labels:
            ids = {
                r.id
                for r in self._model.receptors.sel(
                    time=time, location=location, where=where, **labels
                )
            }
            keys = [k for k in keys if k.receptor in ids]
        return SimulationCollection(self._model, keys)

    # -- completion ------------------------------------------------------------

    def _present(self) -> dict[str, tuple[frozenset[str], frozenset[str] | None]]:
        """
        Return, per selected variant, the receptors with particles and with a footprint.

        This is :meth:`stilt.Simulation.is_complete`'s rule read from one
        listing of each result folder, in place of a file check per
        simulation: on a large project the checks are the slow part. The
        footprint set is ``None`` for a variant that makes no footprint.
        """
        output = self._model.output
        present: dict[str, tuple[frozenset[str], frozenset[str] | None]] = {}
        for name in self.variants:
            variant = self._model.variants[name]
            run = output.find_run(variant.transport)
            particles = frozenset(run.receptors()) if run is not None else frozenset()
            footprints: frozenset[str] | None = None
            if variant.footprint is not None:
                feet = (
                    None
                    if run is None
                    else output.find_footprints(run.hash, variant.footprint)
                )
                footprints = (
                    frozenset(feet.receptors()) if feet is not None else frozenset()
                )
            present[name] = (particles, footprints)
        return present

    def _outputs(self) -> list[tuple[SimID, bool, bool | None]]:
        """
        Return ``(id, has particles, has footprint)`` for each selected simulation.

        The footprint entry is ``None`` when the variant makes none. A small
        selection is checked file by file, which is cheaper than listing
        whole folders for a handful of simulations.
        """
        keys = self._all()
        if len(keys) < _LIST_FROM:
            rows = []
            for sim in self:
                foot = sim.has_footprint if sim.makes_footprint else None
                rows.append((sim.id, sim.has_trajectory, foot))
            return rows
        present = self._present()
        return [
            (
                key,
                key.receptor in present[key.variant][0],
                None
                if (feet := present[key.variant][1]) is None
                else key.receptor in feet,
            )
            for key in keys
        ]

    def incomplete(self) -> SimulationCollection:
        """Return the simulations that are missing an expected output."""
        return SimulationCollection(
            self._model,
            [
                key
                for key, particles, footprint in self._outputs()
                if not (particles and footprint is not False)
            ],
        )

    def status(self) -> pd.DataFrame:
        """
        Return one row per simulation saying which outputs exist.

        Returns
        -------
        pandas.DataFrame
            Columns ``receptor``, ``variant``, ``trajectory``, ``footprint``,
            ``empty``, and ``complete``. ``trajectory`` and ``footprint`` are
            ``True`` when the output exists, ``False`` when it is missing, and
            ``NA`` when the simulation does not make it. ``empty`` is ``True``
            when the footprint is empty (no particle reached the grid).
            ``complete`` is :meth:`~stilt.Simulation.is_complete`.
        """
        rows = [
            {
                "receptor": key.receptor,
                "variant": key.variant,
                TRAJECTORY: particles,
                FOOTPRINT: pd.NA if footprint is None else footprint,
                # Emptiness is inside the file, so this opens each footprint.
                "empty": pd.NA
                if footprint is None
                else footprint and self._model.simulation(key).empty_reason is not None,
                "complete": particles and footprint is not False,
            }
            for key, particles, footprint in self._outputs()
        ]
        columns = ["receptor", "variant", TRAJECTORY, FOOTPRINT, "empty", "complete"]
        return pd.DataFrame(rows, columns=pd.Index(columns)).astype(
            {
                TRAJECTORY: "boolean",
                FOOTPRINT: "boolean",
                "empty": "boolean",
                "complete": "bool",
            }
        )

    # -- outputs ---------------------------------------------------------------

    def jacobian(self, target: Geometry, time_bins: pd.IntervalIndex) -> Jacobian:
        """
        Sum the selected footprints onto a target, per time bin, as one sparse matrix.

        The selection must hold one variant, so every footprint is on one
        grid. See :meth:`stilt.output.Footprints.jacobian`.

        Raises
        ------
        ValueError
            If the selection spans several variants, its variant has no
            grid or no footprints yet, or ``time_bins`` is not closed on the
            left.
        """
        variants = self.variants
        if len(variants) != 1:
            raise ValueError(
                f"jacobian needs one variant; select one of {variants} first."
            )
        first = self._model.simulation(self._all()[0])
        feet = first.footprints
        if feet is None:
            raise ValueError(f"Variant {variants[0]!r} has no footprints yet.")
        return feet.jacobian(target, time_bins, receptors=self.receptors)

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

    Only simulations that make the output are included: a variant without
    a grid has no footprint. Get one from ``model.simulations.trajectories``
    or ``model.simulations.footprint``.
    """

    def __init__(self, simulations: SimulationCollection, output: str):
        if output not in (TRAJECTORY, FOOTPRINT):
            raise ValueError(f"Unknown output {output!r}")
        self._sims = simulations
        self.output = output

    def _producers(self) -> list[Simulation]:
        """Return the simulations that make this output."""
        if self.output == TRAJECTORY:
            return list(self._sims)
        return [sim for sim in self._sims if sim.makes_footprint]

    def paths(self) -> dict[SimID, Path]:
        """Return the paths of the output files that exist in the output directory, by simulation id."""
        found = {}
        for sim in self._producers():
            path = (
                sim.trajectories_path
                if self.output == TRAJECTORY
                else sim.footprint_path
            )
            if path is not None and path.exists():
                found[sim.id] = path
        return found

    def load(self) -> dict[SimID, Trajectories] | dict[SimID, Footprint]:
        """
        Load every output that exists, by simulation id.

        An empty footprint is left out, as :attr:`stilt.Simulation.footprint`
        is ``None`` for it.

        Returns
        -------
        dict
            :class:`~stilt.Trajectories` or :class:`~stilt.Footprint`
            objects keyed by :class:`~stilt.SimID`.
        """
        if self.output == TRAJECTORY:
            return {
                sim.id: sim.trajectories
                for sim in self._producers()
                if sim.has_trajectory
            }
        return {
            sim.id: foot
            for sim in self._producers()
            if sim.has_footprint and (foot := sim.footprint) is not None
        }

    def __len__(self) -> int:
        return len(self._producers())


__all__ = ["OutputCollection", "ReceptorCollection", "SimulationCollection"]
