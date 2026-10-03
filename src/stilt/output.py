"""
The output directory, where particles and footprints are kept.

A project's results live in an output directory that several projects can
share. It has one tree per kind of result, and inside each a folder per set
of settings, in hive form so every tree reads as one dataset::

    <output>/
      particles/
        settings=hrrr-a3f9c2/            variant name + short hash of the transport settings
          _settings.yaml
          date=2024-07-01/<receptor>.parquet
        settings=hrrr-err-0-7be104/
      footprints/
        settings=hrrr-93278c/            variant name + short hash of transport + footprint settings
          _settings.yaml                 names the particles folder it was made from
          date=2024-07-01/<receptor>.parquet
        settings=hrrr-hexes-1b2c3d/      same particles, other footprint settings
      logs/
        settings=hrrr-a3f9c2/date=2024-07-01/<receptor>.log

A folder is found by the hash of its settings, so two projects that run the
same settings share one folder, and a changed setting lands in a new folder
beside the old one instead of overwriting it. ``pyarrow.dataset``, DuckDB,
polars, and R's arrow read ``settings`` and ``date`` as columns of the whole
tree. Particles are one Parquet file per receptor. Footprints are sparse
tables of the non-zero cells, in float32 as STILT-R writes them; an empty
footprint is a file with no rows and its reason in the metadata.

Start from :class:`Output` and a resolved variant (``project.variants``)::

    out = Output("output")
    particles = out.particles(variant)
    particles.write(receptor, frame, met_files)
    feet = out.footprints(variant)
    feet.write(footprint)
    H = feet.jacobian(target, time_bins)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import xarray as xr
import yaml

from stilt._atomic import atomic_path
from stilt.footprint import (
    FOOTPRINT_SCHEMA,
    Geometry,
    Jacobian,
    jacobian,
    read_footprint,
    write_empty_footprint,
    write_footprint,
)
from stilt.footprint.config import FootprintConfig
from stilt.identity import (
    footprint_hash,
    footprint_settings,
    read_footprint_settings,
    read_run_settings,
    settings_hash,
)
from stilt.particles import read_particles, write_particles
from stilt.receptors import Receptor, parse_receptor_id
from stilt.spatial import Grid

if TYPE_CHECKING:
    from stilt.variants import Variant

logger = logging.getLogger(__name__)

#: Underscore-prefixed, so dataset readers skip it.
SETTINGS_FILE = "_settings.yaml"
HASH_CHARS = 6

# -- identity -----------------------------------------------------------------


def _date_dir(receptor_id: str) -> str:
    """Return the ``date=YYYY-MM-DD`` folder of a receptor id, which starts with the receptor time."""
    time, _ = parse_receptor_id(receptor_id)
    return f"date={time:%Y-%m-%d}"


def _list_receptor_files(
    root: Path, suffix: str, among: Iterable[str] | None = None
) -> dict[str, Path]:
    """
    Return ``{receptor_id: path}`` for every ``date=*/<id><suffix>`` under *root*, in date order.

    With *among*, only those receptors are returned, and only their date
    folders are listed. The date folders are listed in a few threads: on a
    network filesystem the listing waits on the server, and a project can
    have thousands of date folders.
    """
    if not root.exists():
        return {}
    if among is None:
        wanted = None
        days = sorted(
            entry.path
            for entry in os.scandir(root)
            if entry.name.startswith("date=") and entry.is_dir()
        )
    else:
        wanted = set(among)
        days = [str(root / day) for day in sorted({_date_dir(r) for r in wanted})]

    def names(day: str) -> list[str]:
        try:
            return sorted(e.name for e in os.scandir(day) if e.name.endswith(suffix))
        except FileNotFoundError:  # no receptor of that day has finished
            return []

    with ThreadPoolExecutor(max_workers=16) as pool:
        listed = list(pool.map(names, days))
    found = {
        name[: -len(suffix)]: Path(day, name)
        for day, per_day in zip(days, listed, strict=True)
        for name in per_day
    }
    if wanted is not None:
        found = {rid: path for rid, path in found.items() if rid in wanted}
    return found


def _read_files(root: Path, files: dict[str, Path], empty: pa.Table) -> pa.Table:
    """
    Return the files of a results folder as one table, or *empty* when there are none.

    *files* is what :func:`_list_receptor_files` returns for *root*. The
    ``date`` column comes from the files' ``date=`` folders.
    """
    if not files:
        return empty
    dataset = pads.dataset(
        [str(p) for p in files.values()],
        format="parquet",
        partitioning=_DATE_PARTITIONING,
        partition_base_dir=str(root),
    )
    return dataset.to_table()


def _write_settings(path: Path, record: dict[str, Any]) -> None:
    """Write a ``settings.yaml`` once. An existing file with the same hash is left alone."""
    if path.exists():
        existing = yaml.safe_load(path.read_text()) or {}
        if existing.get("hash") != record["hash"]:
            raise FileExistsError(
                f"{path} holds settings with hash {existing.get('hash')}, "
                f"not {record['hash']}."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Another worker may write the same settings meanwhile; both files say
    # the same thing, so whichever rename lands last is fine.
    with atomic_path(path) as tmp:
        tmp.write_text(
            yaml.safe_dump(record, default_flow_style=False, sort_keys=False)
        )


def _pystilt_version() -> str:
    from stilt import __version__

    return __version__


def _stamp(digest: str) -> dict[bytes, bytes]:
    """Return what every result file records besides its contents: its folder's settings hash and the PYSTILT version."""
    return {
        b"stilt:hash": digest.encode(),
        b"stilt:pystilt": _pystilt_version().encode(),
    }


# -- the output directory -----------------------------------------------------


def _settings_folders(tree: Path) -> list[str]:
    """Return the ``settings=`` values under *tree* that hold a settings file, in name order."""
    if not tree.exists():
        return []
    return [
        e.name[len("settings=") :]
        for e in sorted(os.scandir(tree), key=lambda e: e.name)
        if e.is_dir()
        and e.name.startswith("settings=")
        and (Path(e.path) / SETTINGS_FILE).exists()
    ]


class Output:
    """
    An output directory: particles and footprints, by their settings.

    Parameters
    ----------
    path : str or Path
        The directory. Created on first write.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Every folder read so far, by key. Reading a folder means reading its
        # _settings.yaml and hashing it again, so each is read once; a lookup
        # that misses lists the tree again and reads only the new folders.
        self._particles: dict[str, Particles] = {}
        self._footprints: dict[str, Footprints] = {}

    def __repr__(self) -> str:
        return f"Output({str(self.path)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and other.path == self.path

    def __hash__(self) -> int:
        return hash(str(self.path))

    @property
    def particles_dir(self) -> Path:
        return self.path / Particles.tree

    @property
    def footprints_dir(self) -> Path:
        return self.path / Footprints.tree

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def scratch_dir(self) -> Path:
        return self.path / "scratch"

    # -- finding folders ----------------------------------------------------

    def _list(self, kind: type[F], known: dict[str, F]) -> dict[str, F]:
        """List *kind*'s tree and add the folders not in *known*, reading only those."""
        for key in _settings_folders(self.path / kind.tree):
            if key not in known:
                known[key] = kind(self, key)
        return known

    def _find(self, kind: type[F], known: dict[str, F], digest: str) -> F | None:
        """Return the folder whose settings hash to *digest*, listing the tree again only on a miss."""
        for folders in (known, self._list(kind, known)):
            for folder in folders.values():
                if folder.hash == digest:
                    return folder
        return None

    def _particles_folder(self, key: str) -> Particles:
        """Return the particles folder named *key*, reading it on first use."""
        if key not in self._particles:
            self._particles[key] = Particles(self, key)
        return self._particles[key]

    def particle_sets(self) -> list[Particles]:
        """Return every particles folder in the directory, in folder-name order."""
        return sorted(
            self._list(Particles, self._particles).values(), key=lambda f: f.key
        )

    def footprint_sets(self) -> list[Footprints]:
        """Return every footprint folder in the directory, in folder-name order."""
        return sorted(
            self._list(Footprints, self._footprints).values(), key=lambda f: f.key
        )

    def find_particles(self, variant: Variant) -> Particles | None:
        """
        Return the particles folder of *variant*, whatever name it carries, or ``None``.

        The folder is found by ``variant.particles_hash``. Each folder's
        stored settings are read back through the current config classes
        and hashed again (:func:`stilt.identity.read_run_settings`), so a
        setting added since the folder was written, with a default, still
        matches.
        """
        return self._find(Particles, self._particles, variant.particles_hash)

    def find_footprints(self, variant: Variant) -> Footprints | None:
        """
        Return the footprint folder of *variant*, or ``None``.

        ``None`` when the variant makes no footprints (it has no grid) or
        none have been written yet. The folder is found by
        ``variant.footprint_hash``, whatever name it carries.
        """
        if variant.footprint_hash is None:
            return None
        return self._find(Footprints, self._footprints, variant.footprint_hash)

    # -- making folders ---------------------------------------------------------

    def _create(
        self, kind: type[F], known: dict[str, F], key: str, record: dict[str, Any]
    ) -> F:
        """Write a folder's settings file and return the folder."""
        _write_settings(
            self.path / kind.tree / f"settings={key}" / SETTINGS_FILE, record
        )
        known[key] = kind(self, key)
        return known[key]

    def particles(self, variant: Variant) -> Particles:
        """
        Return the particles folder of *variant*, creating it on first use.

        The folder is ``particles/settings=<name>-<hash>``. An existing
        folder with the same settings is reused even if it was created under
        another name.
        """
        existing = self.find_particles(variant)
        if existing is not None:
            return existing
        digest = variant.particles_hash
        return self._create(
            Particles,
            self._particles,
            f"{variant.name}-{digest[:HASH_CHARS]}",
            {
                "name": variant.name,
                "hash": digest,
                "pystilt": _pystilt_version(),
                "settings": variant.run_settings,
            },
        )

    def footprints(self, variant: Variant) -> Footprints:
        """
        Return the footprint folder of *variant*, creating it, and its particles folder, on first use.

        Raises
        ------
        ValueError
            If the variant makes no footprints (it has no grid).
        """
        if variant.footprint is None:
            raise ValueError(f"Variant {variant.name!r} makes no footprints (no grid).")
        existing = self.find_footprints(variant)
        if existing is not None:
            return existing
        return self.particles(variant).footprints(
            variant.footprint, name=variant.name, geometry_hash=variant.geometry_hash
        )


F = TypeVar("F", bound="_Folder")

#: The ``date=YYYY-MM-DD`` folders, read as a ``date32`` column.
_DATE_PARTITIONING = pads.partitioning(
    pa.schema([("date", pa.date32())]), flavor="hive"
)


class _Folder:
    """
    What a particles folder and a footprint folder share.

    A folder is ``<output>/<tree>/settings=<key>/``: a ``_settings.yaml``
    and one Parquet file per receptor in ``date=YYYY-MM-DD`` folders.
    Get one from :class:`Output`.
    """

    #: The tree under the output directory: ``particles`` or ``footprints``.
    tree: ClassVar[str]
    #: What :meth:`table` returns when no file is read.
    _empty: ClassVar[pa.Table]
    #: Hash of the folder's settings, read through the current config classes;
    #: it is what finds the folder.
    hash: str

    def __init__(self, output: Output, key: str) -> None:
        self.output = output
        self.key = key
        #: The folder's ``_settings.yaml``, as written.
        self.record: dict[str, Any] = (
            yaml.safe_load((self.path / SETTINGS_FILE).read_text()) or {}
        )
        self.name: str = self.record["name"]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Folder)
            and type(other) is type(self)
            and other.output.path == self.output.path
            and other.key == self.key
        )

    def __hash__(self) -> int:
        return hash((self.tree, str(self.output.path), self.key))

    @property
    def path(self) -> Path:
        """The folder, ``<tree>/settings=<key>``."""
        return self.output.path / self.tree / f"settings={self.key}"

    def file(self, receptor_id: str) -> Path:
        """Return a receptor's file, whether or not it exists."""
        return self.path / _date_dir(receptor_id) / f"{receptor_id}.parquet"

    def has(self, receptor_id: str) -> bool:
        """Return whether a receptor's file exists."""
        return self.file(receptor_id).exists()

    def receptors(self, among: Iterable[str] | None = None) -> list[str]:
        """
        Return the ids of the receptors that have a file here, in date order.

        With *among*, only those receptors are checked, by listing their
        date folders alone.
        """
        return list(_list_receptor_files(self.path, ".parquet", among))

    def table(self, receptors: Iterable[str] | None = None) -> pa.Table:
        """
        Return the files of many receptors as one table.

        Columns are ``receptor``, the stored columns, and ``date`` (the
        receptor date, from the folder, as ``date32``). With *receptors*,
        only those files are read, and only their date folders are listed.
        """
        files = _list_receptor_files(self.path, ".parquet", receptors)
        return _read_files(self.path, files, self._empty)


class Particles(_Folder):
    """
    One particles folder: the particles of many receptors, one set of transport settings.

    Get one from :meth:`Output.particles`. ``key`` is the ``settings=`` value
    its folders share across the ``particles/``, ``logs/``, and ``scratch/``
    trees, so the logs of the HYSPLIT runs that made the particles are here
    too.
    """

    tree = "particles"
    _empty = pa.table({"receptor": pa.array([], pa.string())})

    def __init__(self, output: Output, key: str) -> None:
        super().__init__(output, key)
        #: The settings the particles were made with, read through the current config classes.
        self.settings: dict[str, Any] = read_run_settings(self.record["settings"])
        self.hash = settings_hash(self.settings)

    def write(
        self,
        receptor: Receptor,
        particles: pd.DataFrame,
        met_files: list[Path],
    ) -> Path:
        """
        Write a receptor's particles (:func:`stilt.write_particles`).

        The file records this folder's settings, so it reads alone, and the
        folder's settings hash and the PYSTILT version.
        """
        return write_particles(
            self.file(str(receptor.id)),
            particles,
            receptor,
            self.settings,
            met_files,
            metadata=_stamp(self.hash),
        )

    def read(self, receptor_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Read a receptor's particles (:func:`stilt.read_particles`)."""
        return read_particles(self.file(receptor_id), columns=columns)

    # -- logs and scratch --------------------------------------------------

    @property
    def logs_dir(self) -> Path:
        """The folder of this folder's HYSPLIT logs, ``logs/settings=<key>``."""
        return self.output.logs_dir / f"settings={self.key}"

    @property
    def scratch_dir(self) -> Path:
        """The folder kept HYSPLIT working directories go in, ``scratch/settings=<key>``."""
        return self.output.scratch_dir / f"settings={self.key}"

    def scratch_path(self, receptor_id: str) -> Path:
        """Return where a receptor's HYSPLIT working directory is kept when HYSPLIT fails."""
        return self.scratch_dir / _date_dir(receptor_id) / receptor_id

    def log_path(self, receptor_id: str) -> Path:
        """Return where a receptor's HYSPLIT log is kept."""
        return self.logs_dir / _date_dir(receptor_id) / f"{receptor_id}.log"

    def write_log(self, receptor_id: str, text: str) -> Path:
        """Write a receptor's HYSPLIT log."""
        path = self.log_path(receptor_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    # -- footprints --------------------------------------------------------

    def footprints(
        self,
        config: FootprintConfig,
        name: str | None = None,
        geometry_hash: str | None = None,
    ) -> Footprints:
        """
        Return the footprint folder for *config* on these particles, creating it on first use.

        The folder is ``footprints/settings=<name>-<hash>``, hashed over the
        particles and the footprint settings together, so two footprint
        configs on the same particles get different folders and the same
        config on different particles does too. *name* is the variant the
        footprints belong to; it defaults to the particles folder's name.
        *geometry_hash* is the hash of the geometry the grid was derived for.
        """
        if config.grid is None:
            raise ValueError("Footprint settings need a grid.")
        settings = footprint_settings(config, geometry_hash)
        digest = footprint_hash(self.hash, settings)
        existing = self.output._find(Footprints, self.output._footprints, digest)
        if existing is not None:
            return existing
        name = name or self.name
        return self.output._create(
            Footprints,
            self.output._footprints,
            f"{name}-{digest[:HASH_CHARS]}",
            {
                "name": name,
                "hash": digest,
                "particles": self.key,
                "particles_hash": self.hash,
                "pystilt": _pystilt_version(),
                "settings": settings,
            },
        )


class Footprints(_Folder):
    """
    One footprint folder: the footprints of many receptors, one set of footprint settings, one set of particles.

    Get one from :meth:`Output.footprints` or :meth:`Particles.footprints`.
    ``key`` is the ``settings=`` value of its folder under ``footprints/``.
    """

    tree = "footprints"
    _empty = FOOTPRINT_SCHEMA.append(pa.field("date", pa.date32())).empty_table()

    def __init__(self, output: Output, key: str) -> None:
        super().__init__(output, key)
        #: ``settings=`` value of the particles folder these were made from.
        self.particles_key: str = self.record["particles"]
        #: The footprint config the footprints were made with, and the hash of
        #: the geometry its grid was derived for, read through the current classes.
        self.config, self.geometry_hash = read_footprint_settings(
            self.record["settings"], str(self.path / SETTINGS_FILE)
        )
        if self.config.grid is None:
            raise ValueError(f"{self.path / SETTINGS_FILE} has no grid.")
        #: Grid the stored ``x`` and ``y`` index (``config.grid``).
        self.grid: Grid = self.config.grid
        # Over the particles' hash and the footprint settings together.
        self.hash = footprint_hash(
            self.particles.hash, footprint_settings(self.config, self.geometry_hash)
        )

    @property
    def particles(self) -> Particles:
        """The particles folder these footprints were made from."""
        return self.output._particles_folder(self.particles_key)

    def write(self, foot: xr.DataArray) -> Path:
        """
        Write one receptor's footprint (:func:`stilt.footprint.write_footprint`).

        The footprint must be on this folder's grid. The file records this
        folder's settings and hash, so it reads alone.
        """
        path = self.file(str(foot.stilt.receptor.id))
        return write_footprint(
            path, foot, self.config, _stamp(self.hash), geometry_hash=self.geometry_hash
        )

    def write_empty(self, receptor: Receptor, reason: str, name: str = "") -> Path:
        """Record that a receptor's footprint is empty (no particle over the grid), with the reason."""
        return write_empty_footprint(
            self.file(str(receptor.id)),
            receptor,
            reason,
            self.config,
            name,
            _stamp(self.hash),
            geometry_hash=self.geometry_hash,
        )

    def empty_reason(self, receptor_id: str) -> str | None:
        """Return why the receptor's footprint is empty, or ``None`` when it is not empty."""
        meta = pq.read_schema(self.file(receptor_id)).metadata or {}
        reason = meta.get(b"stilt:empty_reason", b"").decode()
        return reason or None

    def read(self, receptor_id: str) -> xr.DataArray | None:
        """
        Read one receptor's footprint as a dense array (:func:`stilt.read_footprint`).

        Returns ``None`` for an empty footprint (see :meth:`empty_reason`).
        """
        return read_footprint(self.file(receptor_id))

    def jacobian(
        self,
        target: Geometry,
        time_bins: pd.IntervalIndex,
        receptors: Iterable[str] | None = None,
    ) -> Jacobian:
        """
        Sum many footprints onto a target, per time bin, as one sparse matrix.

        The same operation as ``foot.stilt.aggregate``, for every
        receptor in the folder (or *receptors*) at once. Each footprint cell
        is split among the target cells it overlaps in proportion to area,
        time layers are summed within each of *time_bins*, and cells or
        layers outside the target or the bins are dropped.

        Parameters
        ----------
        target : Grid, Mesh, or Zones
            Cells to sum onto.
        time_bins : pandas.IntervalIndex
            Time intervals, such as a flux inventory's steps. They must be
            closed on the left (``closed="left"``): each bin holds the
            footprint hours that start in it.
        receptors : iterable of str, optional
            Receptor ids to include. All by default.

        Returns
        -------
        Jacobian
            Rows are receptors with a non-empty footprint; columns are
            ``(time bin, target cell)``.

        Raises
        ------
        ValueError
            If ``time_bins`` is not closed on the left.
        """
        if receptors is None:
            files = _list_receptor_files(self.path, ".parquet")
            requested = list(files)
        else:
            requested = list(dict.fromkeys(receptors))
            files = _list_receptor_files(self.path, ".parquet", requested)
        present = [r for r in requested if r in files]
        table = _read_files(self.path, {r: files[r] for r in present}, self._empty)
        return jacobian(
            table,
            self.config,
            target,
            time_bins,
            receptors=present,
            missing=[r for r in requested if r not in files],
            geometry_hash=self.geometry_hash,
        )


__all__ = [
    "Footprints",
    "Output",
    "Particles",
]
