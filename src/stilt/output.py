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

Start from :class:`Output`::

    out = Output("output")
    particles = out.particles("hrrr", settings)
    particles.write(receptor, frame, params, met_files)
    feet = particles.footprints(footprint_config)
    feet.write(footprint)
    H = feet.jacobian(target, time_bins)
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import xarray as xr
import yaml
from scipy import sparse

from stilt._atomic import atomic_path
from stilt.config import (
    FootprintConfig,
    Grid,
    TransportParams,
    TransportSettings,
    VariantConfig,
)
from stilt.config.transport import canonical, settings_hash
from stilt.footprint import (
    FOOTPRINT_SCHEMA,
    read_footprint,
    write_empty_footprint,
    write_footprint,
)
from stilt.particles import read_particles, write_particles
from stilt.receptors import Receptor, parse_receptor_id
from stilt.spatial import Geometry, check_resolution, overlap_weights

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
        # Folders found so far, by settings hash. A miss rescans the directory,
        # so a folder another worker created is picked up on the next lookup.
        self._particle_sets: dict[str, Particles] = {}
        self._footprints: dict[str, Footprints] = {}

    def __repr__(self) -> str:
        return f"Output({str(self.path)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and other.path == self.path

    def __hash__(self) -> int:
        return hash(str(self.path))

    @property
    def particles_dir(self) -> Path:
        return self.path / "particles"

    @property
    def footprints_dir(self) -> Path:
        return self.path / "footprints"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def scratch_dir(self) -> Path:
        return self.path / "scratch"

    def particle_sets(self) -> list[Particles]:
        """Return every particles folder in the directory, in folder-name order."""
        found = [Particles(self, key) for key in _settings_folders(self.particles_dir)]
        self._particle_sets = {p.hash: p for p in found}
        return found

    def footprint_sets(self) -> list[Footprints]:
        """Return every footprint folder in the directory, in folder-name order."""
        found = [
            Footprints(self, key) for key in _settings_folders(self.footprints_dir)
        ]
        self._footprints = {feet.hash: feet for feet in found}
        return found

    def find_footprints(self, variant: VariantConfig) -> Footprints | None:
        """
        Return the footprint folder of *variant*, or ``None``.

        ``None`` when the variant makes no footprints (it has no grid) or
        none have been written yet. The folder is found by the hash of the
        variant's transport and footprint settings, whatever name it carries.
        """
        if variant.footprint is None:
            return None
        return self._footprints_by_hash(
            Footprints.hash_for(variant.transport.hash, variant.footprint)
        )

    def _footprints_by_hash(self, digest: str) -> Footprints | None:
        """Return the footprint folder whose settings hash to *digest*, or ``None``."""
        if digest not in self._footprints:
            self.footprint_sets()
        return self._footprints.get(digest)

    def find_particles(self, settings: TransportSettings) -> Particles | None:
        """
        Return the particles folder for these settings, whatever name it carries, or ``None``.

        Each folder's stored settings are loaded back through
        :class:`~stilt.config.TransportSettings` and hashed again, so a
        field added since the folder was written, with a default, still
        matches.
        """
        digest = settings.hash
        if digest not in self._particle_sets:
            self.particle_sets()
        return self._particle_sets.get(digest)

    def particles(self, name: str, settings: TransportSettings) -> Particles:
        """
        Return the particles folder for *settings*, creating it on first use.

        The folder is ``particles/settings=<name>-<hash>``. An existing
        folder with the same settings is reused even if it was created under
        another name.
        """
        existing = self.find_particles(settings)
        if existing is not None:
            return existing
        digest = settings.hash
        key = f"{name}-{digest[:HASH_CHARS]}"
        _write_settings(
            self.particles_dir / f"settings={key}" / SETTINGS_FILE,
            {
                "name": name,
                "hash": digest,
                "pystilt": _pystilt_version(),
                "settings": settings.identity(),
            },
        )
        created = Particles(self, key)
        self._particle_sets[created.hash] = created
        return created


class Particles:
    """
    One particles folder: the particles of many receptors, one set of transport settings.

    Get one from :meth:`Output.particles`. ``key`` is the ``settings=`` value
    its folders share across the ``particles/``, ``logs/``, and ``scratch/``
    trees. The logs of the HYSPLIT runs that made the particles are here too.
    """

    def __init__(self, output: Output, key: str) -> None:
        self.output = output
        self.key = key
        record = yaml.safe_load((self.path / SETTINGS_FILE).read_text()) or {}
        self.name: str = record["name"]
        #: The settings the particles were made with, re-validated by the current model.
        self.settings = TransportSettings.from_stored(record["settings"])
        #: Hash of the re-validated settings (see :meth:`Output.find_particles`).
        self.hash: str = self.settings.hash

    def __repr__(self) -> str:
        return f"Particles({self.key!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Particles)
            and other.output.path == self.output.path
            and other.key == self.key
        )

    def __hash__(self) -> int:
        return hash((str(self.output.path), self.key))

    @property
    def path(self) -> Path:
        """The folder, ``particles/settings=<key>``."""
        return self.output.particles_dir / f"settings={self.key}"

    @property
    def logs_dir(self) -> Path:
        return self.output.logs_dir / f"settings={self.key}"

    @property
    def scratch_dir(self) -> Path:
        return self.output.scratch_dir / f"settings={self.key}"

    # -- particles ---------------------------------------------------------

    def file(self, receptor_id: str) -> Path:
        """Return the particle file for a receptor, whether or not it exists."""
        return self.path / _date_dir(receptor_id) / f"{receptor_id}.parquet"

    def has(self, receptor_id: str) -> bool:
        """Return whether the receptor's particle file exists."""
        return self.file(receptor_id).exists()

    def receptors(self, among: Iterable[str] | None = None) -> list[str]:
        """
        Return the ids of the receptors that have particles, in date order.

        With *among*, only those receptors are checked, by listing their
        date folders alone.
        """
        return list(_list_receptor_files(self.path, ".parquet", among))

    def write(
        self,
        receptor: Receptor,
        particles: pd.DataFrame,
        params: TransportParams,
        met_files: list[Path],
    ) -> Path:
        """
        Write a receptor's particles (:func:`stilt.write_particles`).

        The file also records this folder's settings hash and the PYSTILT
        version.
        """
        return write_particles(
            self.file(str(receptor.id)),
            particles,
            receptor,
            params,
            met_files,
            metadata={
                b"stilt:hash": self.hash.encode(),
                b"stilt:pystilt": _pystilt_version().encode(),
            },
        )

    def table(self, receptors: Iterable[str] | None = None) -> pa.Table:
        """
        Return the particles of many receptors as one table.

        Columns are ``receptor``, the particle columns as stored, and
        ``date`` (the receptor date, from the folder, as ``date32``). With
        *receptors*, only those files are read, and only their date folders
        are listed. Particle files written before the ``receptor`` column
        existed cannot be read this way.
        """
        files = _list_receptor_files(self.path, ".parquet", receptors)
        return _read_files(
            self.path, files, pa.table({"receptor": pa.array([], pa.string())})
        )

    def read(self, receptor_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Read a receptor's particles (:func:`stilt.read_particles`)."""
        return read_particles(self.file(receptor_id), columns=columns)

    # -- logs and scratch --------------------------------------------------

    def scratch_path(self, receptor_id: str) -> Path:
        """Return where a receptor's HYSPLIT working directory is kept when HYSPLIT fails."""
        return self.scratch_dir / _date_dir(receptor_id) / receptor_id

    def log_path(self, receptor_id: str) -> Path:
        return self.logs_dir / _date_dir(receptor_id) / f"{receptor_id}.log"

    def write_log(self, receptor_id: str, text: str) -> Path:
        path = self.log_path(receptor_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    # -- footprints --------------------------------------------------------

    def footprints(
        self, config: FootprintConfig, name: str | None = None
    ) -> Footprints:
        """
        Return the footprint folder for *config* on these particles, creating it on first use.

        The folder is ``footprints/settings=<name>-<hash>``, hashed over the
        transport settings and *config* together, so two footprint settings
        on the same particles get different folders and the same footprint
        settings on different particles do too. *name* is the variant the
        footprints belong to; it defaults to the particles folder's name.
        """
        if config.grid is None:
            raise ValueError("Footprint settings need a grid.")
        digest = Footprints.hash_for(self.hash, config)
        existing = self.output._footprints_by_hash(digest)
        if existing is not None:
            return existing
        name = name or self.name
        key = f"{name}-{digest[:HASH_CHARS]}"
        _write_settings(
            self.output.footprints_dir / f"settings={key}" / SETTINGS_FILE,
            {
                "name": name,
                "hash": digest,
                "particles": self.key,
                "particles_hash": self.hash,
                "pystilt": _pystilt_version(),
                "settings": canonical(config.model_dump(mode="json")),
            },
        )
        feet = Footprints(self.output, key)
        self.output._footprints[feet.hash] = feet
        return feet


#: The ``date=YYYY-MM-DD`` folders, read as a ``date32`` column.
_DATE_PARTITIONING = pads.partitioning(
    pa.schema([("date", pa.date32())]), flavor="hive"
)
#: What :meth:`Footprints.table` returns: the file columns plus ``date``.
_FOOTPRINT_TABLE_SCHEMA = FOOTPRINT_SCHEMA.append(pa.field("date", pa.date32()))


class Jacobian(NamedTuple):
    """
    Footprints of many receptors summed onto a target, as one sparse matrix.

    Attributes
    ----------
    data : scipy.sparse.csr_matrix
        Shape ``(n_receptors, n_bins * n_cells)``. Row ``i`` is
        ``receptors[i]``; the columns run through every target cell for the
        first time bin, then the second, in ``columns`` order.
    receptors : pandas.Index
        Receptor ids of the rows.
    columns : pandas.MultiIndex
        ``(time, cell)`` for each column: the left edge of the time bin and
        the target cell's label.
    empty : list of str
        Receptors whose footprint is empty. They have no row.
    missing : list of str
        Requested receptors that have no footprint file. They have no row.
    """

    data: sparse.csr_matrix
    receptors: pd.Index
    columns: pd.MultiIndex
    empty: list[str]
    missing: list[str]

    def to_frame(self) -> pd.DataFrame:
        """Return the matrix as a dense DataFrame (receptors × columns)."""
        return pd.DataFrame(
            self.data.toarray(), index=self.receptors, columns=self.columns
        )


class Footprints:
    """
    One footprint folder: the footprints of many receptors, one set of footprint settings, one set of particles.

    Get one from :meth:`Particles.footprints`. ``key`` is the ``settings=`` value
    of its folder under ``footprints/``.
    """

    def __init__(self, output: Output, key: str) -> None:
        self.output = output
        self.key = key
        self.path = output.footprints_dir / f"settings={key}"
        record = yaml.safe_load((self.path / SETTINGS_FILE).read_text()) or {}
        self.name: str = record["name"]
        #: ``settings=`` value of the particles folder these were made from.
        self.particles_key: str = record["particles"]
        #: The settings the footprints were made with, re-validated by the current model.
        self.config = FootprintConfig.model_validate(record["settings"])
        if self.config.grid is None:
            raise ValueError(f"{self.path / SETTINGS_FILE} has no grid.")
        #: Grid the stored ``x`` and ``y`` index (``config.grid``).
        self.grid: Grid = self.config.grid
        self._axes: tuple[np.ndarray, np.ndarray] | None = None

    def __repr__(self) -> str:
        return f"Footprints({self.key!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Footprints)
            and other.output.path == self.output.path
            and other.key == self.key
        )

    def __hash__(self) -> int:
        return hash((str(self.output.path), self.key))

    @functools.cached_property
    def particles(self) -> Particles:
        """The particles folder these footprints were made from."""
        return Particles(self.output, self.particles_key)

    @functools.cached_property
    def hash(self) -> str:
        """Hash of the transport settings and the footprint settings together, re-validated."""
        return self.hash_for(self.particles.hash, self.config)

    @staticmethod
    def hash_for(particles_hash: str, config: FootprintConfig) -> str:
        """Return the hash identifying footprints with *config* on the particles hashed *particles_hash*."""
        return settings_hash(
            {"particles": particles_hash, "footprint": config.model_dump(mode="json")}
        )

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-center coordinates ``(x, y)`` that the stored ``x`` and ``y`` index."""
        if self._axes is None:
            self._axes = self.grid.axes
        return self._axes

    def file(self, receptor_id: str) -> Path:
        """Return the footprint file for a receptor, whether or not it exists."""
        return self.path / _date_dir(receptor_id) / f"{receptor_id}.parquet"

    def has(self, receptor_id: str) -> bool:
        """Return whether the receptor has a footprint file, empty or not."""
        return self.file(receptor_id).exists()

    def receptors(self, among: Iterable[str] | None = None) -> list[str]:
        """
        Return the ids of the receptors that have a footprint file, in date order.

        With *among*, only those receptors are checked, by listing their
        date folders alone.
        """
        return list(_list_receptor_files(self.path, ".parquet", among))

    def write(self, foot: xr.DataArray) -> Path:
        """
        Write one receptor's footprint (:func:`stilt.footprint.write_footprint`).

        The footprint must be on this folder's grid. The file records this
        folder's settings and hash, so it reads alone.
        """
        path = self.file(str(foot.stilt.receptor.id))
        return write_footprint(path, foot, self.config, self._stamp())

    def write_empty(self, receptor: Receptor, reason: str, name: str = "") -> Path:
        """Record that a receptor's footprint is empty (no particle over the grid), with the reason."""
        path = self.file(str(receptor.id))
        return write_empty_footprint(
            path, receptor, reason, self.config, name, self._stamp()
        )

    def _stamp(self) -> dict[bytes, bytes]:
        """Return the metadata a file of this folder records besides its footprint."""
        return {
            b"stilt:hash": self.hash.encode(),
            b"stilt:pystilt": _pystilt_version().encode(),
        }

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

    # -- many receptors at once --------------------------------------------

    def table(self, receptors: Iterable[str] | None = None) -> pa.Table:
        """
        Return the non-zero cells of many footprints as one table.

        Columns are ``receptor``, ``hour``, ``y``, ``x``, ``foot``, and
        ``date`` (the receptor date, from the folder, as ``date32``). With
        *receptors*, only those files are read, and only their date folders
        are listed.
        """
        files = _list_receptor_files(self.path, ".parquet", receptors)
        return _read_files(self.path, files, _FOOTPRINT_TABLE_SCHEMA.empty_table())

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
        if time_bins.closed != "left":
            raise ValueError(
                f"time_bins must be closed on the left, not {time_bins.closed!r}. "
                "A footprint time is the start of its hour, so each bin takes "
                "the hours that start in it. Build the bins with "
                "closed='left', for example "
                "pd.interval_range(start, end, freq='1h', closed='left')."
            )
        if receptors is None:
            files = _list_receptor_files(self.path, ".parquet")
            requested = list(files)
        else:
            requested = list(dict.fromkeys(receptors))
            files = _list_receptor_files(self.path, ".parquet", requested)
        missing = [r for r in requested if r not in files]
        present = [r for r in requested if r in files]

        x_axis, y_axis = self.axes
        grid = self.grid
        check_resolution(target, grid.xres, grid.yres, grid.crs)
        weights = overlap_weights(
            target, x_axis, y_axis, grid.xres, grid.yres, grid.crs
        )  # (n_cells, ny * nx)
        n_cells = len(target.index)
        nx = len(x_axis)
        n_raster = nx * len(y_axis)

        bin_left = pd.DatetimeIndex(
            pd.to_datetime(time_bins.left, utc=True)
        ).tz_localize(None)
        bin_right = pd.DatetimeIndex(
            pd.to_datetime(time_bins.right, utc=True)
        ).tz_localize(None)
        n_bins = len(time_bins)

        table = _read_files(
            self.path,
            {r: files[r] for r in present},
            _FOOTPRINT_TABLE_SCHEMA.empty_table(),
        )
        if table.num_rows:
            # Work with the dictionary indices of the receptor column: one
            # small array of ids, and an int32 per row.
            table = table.unify_dictionaries().combine_chunks()
            receptor_col = table.column("receptor").combine_chunks()
            ids: list[str] = receptor_col.dictionary.to_pylist()  # type: ignore[attr-defined]
            dict_idx = receptor_col.indices.to_numpy()  # type: ignore[attr-defined]
        else:
            ids, dict_idx = [], np.zeros(0, dtype=np.int32)
        found = set(ids)
        rows = [r for r in present if r in found]
        empty = [r for r in present if r not in found]
        row_of = {r: i for i, r in enumerate(rows)}

        if table.num_rows:
            row_by_id = np.array([row_of[r] for r in ids], dtype=np.int64)
            row_idx = row_by_id[dict_idx]
            hour = table.column("hour").to_numpy().astype(np.int64)
            flat = (
                table.column("y").to_numpy().astype(np.int64) * nx
                + table.column("x").to_numpy()
            )
            foot = table.column("foot").to_numpy().astype(np.float64)

            ns_per_hour = 3_600_000_000_000
            release_ns = np.array(
                [np.datetime64(parse_receptor_id(r)[0], "ns") for r in ids]
            ).astype(np.int64)
            t_ns = release_ns[dict_idx] + hour * ns_per_hour
            # Explicit nanoseconds: pandas may hold these edges at another resolution.
            left_ns = np.asarray(bin_left, dtype="datetime64[ns]").astype(np.int64)
            right_ns = np.asarray(bin_right, dtype="datetime64[ns]").astype(np.int64)
            bin_idx = np.searchsorted(left_ns, t_ns, side="right") - 1
            inside = (bin_idx >= 0) & (t_ns < right_ns[np.clip(bin_idx, 0, n_bins - 1)])

            # F: (receptor) x (bin, raster cell); one product with the block
            # diagonal of W^T gives every bin at once.
            f_all = sparse.coo_matrix(
                (
                    foot[inside],
                    (row_idx[inside], bin_idx[inside] * n_raster + flat[inside]),
                ),
                shape=(len(rows), n_bins * n_raster),
            ).tocsr()
            blocks = sparse.kron(
                sparse.identity(n_bins, format="csr"), weights.T.tocsr()
            )
            data = sparse.csr_matrix(f_all @ blocks)
        else:
            data = sparse.csr_matrix((len(rows), n_bins * n_cells))

        # A grid target's cells are (x, y) tuples; keep them as one label each.
        cells = pd.Index(list(target.index), tupleize_cols=False)
        columns = pd.MultiIndex.from_product([bin_left, cells], names=["time", "cell"])
        return Jacobian(data, pd.Index(rows, name="receptor"), columns, empty, missing)


__all__ = [
    "Footprints",
    "Jacobian",
    "Output",
    "Particles",
]
