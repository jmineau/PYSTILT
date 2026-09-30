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
    run = out.run("hrrr", settings)
    run.write_particles(trajectories)
    feet = run.footprints(footprint_config)
    feet.write(footprint)
    H = feet.jacobian(target, time_bins)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from collections.abc import Iterable
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

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    TransportSettings,
    VariantConfig,
)
from stilt.config.transport import canonical, settings_hash
from stilt.footprint import Footprint
from stilt.geometry import Geometry, check_resolution, overlap_weights
from stilt.receptors import Receptor
from stilt.trajectory import Trajectories

logger = logging.getLogger(__name__)

#: Underscore-prefixed, so dataset readers skip it.
SETTINGS_FILE = "_settings.yaml"
HASH_CHARS = 6
_RECEPTOR_ID_RE = re.compile(r"^\d{12}_")

#: Particle columns stored as int32 rather than float64.
_INT_COLUMNS = ("time", "indx")


# -- identity -----------------------------------------------------------------


def _date_dir(receptor_id: str) -> str:
    """Return the ``date=YYYY-MM-DD`` folder of a receptor id, which starts with the receptor time."""
    if not _RECEPTOR_ID_RE.match(receptor_id):
        raise ValueError(f"Receptor id {receptor_id!r} does not start with a time.")
    return f"date={receptor_id[:4]}-{receptor_id[4:6]}-{receptor_id[6:8]}"


def _receptor_time(receptor_id: str) -> dt.datetime:
    """Return the receptor time encoded at the start of a receptor id."""
    return dt.datetime.strptime(receptor_id[:12], "%Y%m%d%H%M")


def _list_receptor_files(root: Path, suffix: str) -> dict[str, Path]:
    """Return ``{receptor_id: path}`` for every ``date=*/<id><suffix>`` under *root*, in date order."""
    if not root.exists():
        return {}
    found: dict[str, Path] = {}
    for day in sorted(os.scandir(root), key=lambda e: e.name):
        if not (day.is_dir() and day.name.startswith("date=")):
            continue
        for entry in sorted(os.scandir(day.path), key=lambda e: e.name):
            if entry.name.endswith(suffix):
                found[entry.name[: -len(suffix)]] = Path(entry.path)
    return found


def _write_atomic_table(table: pa.Table, path: Path) -> Path:
    """Write *table* as zstd Parquet through a temporary file, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(record, default_flow_style=False, sort_keys=False))
    os.replace(tmp, path)


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

    def __repr__(self) -> str:
        return f"Output({str(self.path)!r})"

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

    def runs(self) -> list[Run]:
        """Return every run (particles folder) in the directory, in folder-name order."""
        return [Run(self, key) for key in _settings_folders(self.particles_dir)]

    def footprint_sets(self) -> list[Footprints]:
        """Return every footprint folder in the directory, in folder-name order."""
        return [Footprints(self, key) for key in _settings_folders(self.footprints_dir)]

    def find_run(self, settings: TransportSettings) -> Run | None:
        """
        Return the run with these settings, whatever name its folder carries, or ``None``.

        Each folder's stored settings are loaded back through
        :class:`~stilt.config.TransportSettings` and hashed again, so a
        field added since the folder was written, with a default, still
        matches.
        """
        digest = settings.hash
        for run in self.runs():
            if run.hash == digest:
                return run
        return None

    def run(self, name: str, settings: TransportSettings) -> Run:
        """
        Return the run for *settings*, creating its folder on first use.

        The folder is ``particles/settings=<name>-<hash>``. An existing
        folder with the same settings is reused even if it was created under
        another name.
        """
        existing = self.find_run(settings)
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
        return Run(self, key)


class Run:
    """
    The particles run under one set of settings, and their logs.

    Get one from :meth:`Output.run`. ``key`` is the ``settings=`` value the
    run's folders share across the ``particles/`` and ``logs/`` trees.
    """

    def __init__(self, output: Output, key: str) -> None:
        self.output = output
        self.key = key
        record = yaml.safe_load((self.particles_dir / SETTINGS_FILE).read_text()) or {}
        self.name: str = record["name"]
        #: The settings the run was made with, re-validated by the current model.
        self.settings = TransportSettings.model_validate(record["settings"])
        #: Hash of the re-validated settings (see :meth:`Output.find_run`).
        self.hash: str = self.settings.hash

    def __repr__(self) -> str:
        return f"Run({self.key!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Run)
            and other.output.path == self.output.path
            and other.key == self.key
        )

    def __hash__(self) -> int:
        return hash((str(self.output.path), self.key))

    @property
    def path(self) -> Path:
        """The particles folder, ``particles/settings=<key>``."""
        return self.particles_dir

    @property
    def particles_dir(self) -> Path:
        return self.output.particles_dir / f"settings={self.key}"

    @property
    def logs_dir(self) -> Path:
        return self.output.logs_dir / f"settings={self.key}"

    @property
    def scratch_dir(self) -> Path:
        return self.output.scratch_dir / f"settings={self.key}"

    # -- particles ---------------------------------------------------------

    def particles_path(self, receptor_id: str) -> Path:
        """Return the particle file for a receptor, whether or not it exists."""
        return self.particles_dir / _date_dir(receptor_id) / f"{receptor_id}.parquet"

    def has_particles(self, receptor_id: str) -> bool:
        """Return whether the receptor's particle file exists."""
        return self.particles_path(receptor_id).exists()

    def receptors(self) -> list[str]:
        """Return the ids of the receptors that have particles, in date order."""
        return list(_list_receptor_files(self.particles_dir, ".parquet"))

    def write_particles(self, trajectories: Trajectories) -> Path:
        """
        Write a receptor's particles.

        ``time`` and ``indx`` are stored as int32 and ``datetime`` is left
        out, since it is the receptor time plus ``time``. Other columns keep
        their type. The receptor, transport parameters, and met files go in
        the file's metadata, as :meth:`stilt.Trajectories.to_parquet` does.
        """
        data = trajectories.data.drop(columns=["datetime"], errors="ignore")
        for name in _INT_COLUMNS:
            if name in data.columns:
                values = data[name].to_numpy()
                if not np.array_equal(values, np.round(values)):
                    raise ValueError(f"Particle column {name!r} is not whole numbers.")
                data = data.assign(**{name: values.astype(np.int32)})
        table = pa.Table.from_pandas(data, preserve_index=False)
        metadata = {
            b"stilt:receptor": json.dumps(trajectories.receptor.to_dict()).encode(),
            b"stilt:params": trajectories.params.model_dump_json().encode(),
            b"stilt:met_files": json.dumps(
                [str(p) for p in trajectories.met_files]
            ).encode(),
        }
        table = table.replace_schema_metadata(metadata)
        return _write_atomic_table(
            table, self.particles_path(str(trajectories.receptor.id))
        )

    def read_particles(
        self, receptor_id: str, columns: list[str] | None = None
    ) -> Trajectories:
        """
        Read a receptor's particles.

        ``datetime`` is rebuilt from the receptor time, and ``time`` and
        ``indx`` come back as float64, as HYSPLIT's output is read today.
        """
        path = self.particles_path(receptor_id)
        if columns is not None:
            columns = [c for c in columns if c != "datetime"]
        traj = Trajectories.from_parquet(path, columns=columns)
        data = traj.data
        for name in _INT_COLUMNS:
            if name in data.columns:
                data[name] = data[name].astype("float64")
        if "time" in data.columns and (columns is None or "datetime" in columns):
            data["datetime"] = pd.Timestamp(traj.receptor.time) + pd.to_timedelta(
                data["time"].to_numpy(), unit="min"
            )
        return traj

    # -- logs --------------------------------------------------------------

    def log_path(self, receptor_id: str) -> Path:
        return self.logs_dir / _date_dir(receptor_id) / f"{receptor_id}.log"

    def write_log(self, receptor_id: str, text: str) -> Path:
        path = self.log_path(receptor_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    # -- footprints --------------------------------------------------------

    def footprint_sets(self) -> list[Footprints]:
        """Return the footprint folders made from this run's particles."""
        return [f for f in self.output.footprint_sets() if f.run_key == self.key]

    def footprints(
        self, config: FootprintConfig, name: str | None = None
    ) -> Footprints:
        """
        Return the footprint folder for *config* on this run's particles, creating it on first use.

        The folder is ``footprints/settings=<name>-<hash>``, hashed over the
        run's settings and *config* together, so two footprint settings on
        the same particles get different folders and the same footprint
        settings on different particles do too. *name* is the variant the
        footprints belong to; it defaults to the run's name.
        """
        if config.grid is None:
            raise ValueError("Footprint settings need a grid.")
        digest = Footprints.hash_for(self, config)
        for existing in self.footprint_sets():
            if existing.hash == digest:
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
        return Footprints(self.output, key)


_FOOTPRINT_SCHEMA = pa.schema(
    [
        ("receptor", pa.dictionary(pa.int32(), pa.string())),
        ("hour", pa.int16()),
        ("y", pa.int16()),
        ("x", pa.int16()),
        ("foot", pa.float32()),
    ]
)
#: The ``date=YYYY-MM-DD`` folders, read as a ``date32`` column.
_DATE_PARTITIONING = pads.partitioning(
    pa.schema([("date", pa.date32())]), flavor="hive"
)
#: What :meth:`Footprints.table` returns: the file columns plus ``date``.
_FOOTPRINT_TABLE_SCHEMA = _FOOTPRINT_SCHEMA.append(pa.field("date", pa.date32()))


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
    One footprint folder: footprints of many receptors, one setting, one run.

    Get one from :meth:`Run.footprints`. ``key`` is the ``settings=`` value
    of its folder under ``footprints/``.
    """

    def __init__(self, output: Output, key: str) -> None:
        self.output = output
        self.key = key
        self.path = output.footprints_dir / f"settings={key}"
        record = yaml.safe_load((self.path / SETTINGS_FILE).read_text()) or {}
        self.name: str = record["name"]
        #: ``settings=`` value of the particles folder these were made from.
        self.run_key: str = record["particles"]
        #: The settings the footprints were made with, re-validated by the current model.
        self.config = FootprintConfig.model_validate(record["settings"])
        if self.config.grid is None:
            raise ValueError(f"{self.path / SETTINGS_FILE} has no grid.")
        #: Grid the stored ``x`` and ``y`` index (``config.grid``).
        self.grid: Grid = self.config.grid
        self._axes: tuple[np.ndarray, np.ndarray] | None = None
        self._run: Run | None = None

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

    @property
    def run(self) -> Run:
        """The run whose particles these footprints were made from."""
        if self._run is None:
            self._run = Run(self.output, self.run_key)
        return self._run

    @property
    def hash(self) -> str:
        """Hash of the run's settings and the footprint settings together, re-validated."""
        return self.hash_for(self.run, self.config)

    @staticmethod
    def hash_for(run: Run, config: FootprintConfig) -> str:
        """Return the hash that identifies footprints with *config* on *run*'s particles."""
        return settings_hash(
            {"particles": run.hash, "footprint": config.model_dump(mode="json")}
        )

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-center coordinates ``(x, y)`` that the stored ``x`` and ``y`` index."""
        if self._axes is None:
            self._axes = self.grid.axes
        return self._axes

    def footprint_path(self, receptor_id: str) -> Path:
        return self.path / _date_dir(receptor_id) / f"{receptor_id}.parquet"

    def has(self, receptor_id: str) -> bool:
        """Return whether the receptor has a footprint file, empty or not."""
        return self.footprint_path(receptor_id).exists()

    def receptors(self) -> list[str]:
        """Return the ids of the receptors that have a footprint file, in date order."""
        return list(_list_receptor_files(self.path, ".parquet"))

    def _indices(self, coords: np.ndarray, axis: np.ndarray, name: str) -> np.ndarray:
        """Return the index of each coordinate's cell in the regular grid axis, or raise."""
        step = axis[1] - axis[0] if len(axis) > 1 else 1.0
        idx = np.rint((coords - axis[0]) / step).astype(np.intp)
        if (
            idx.min() < 0
            or idx.max() >= len(axis)
            or not np.allclose(axis[idx], coords, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                f"Footprint {name} coordinates are not cells of the folder's grid."
            )
        return idx

    def write(self, footprint: Footprint) -> Path:
        """
        Write one receptor's footprint as its non-zero cells.

        The footprint must be on this folder's grid. ``hour`` is the offset
        of each time layer from the receptor time; every layer is recorded
        in the metadata, so the dense array reads back with the same shape.
        """
        data = footprint.data
        x_dim = "lon" if "lon" in data.dims else "x"
        y_dim = "lat" if "lat" in data.dims else "y"
        x_axis, y_axis = self.axes
        xi = self._indices(np.asarray(data[x_dim].values, dtype=float), x_axis, x_dim)
        yi = self._indices(np.asarray(data[y_dim].values, dtype=float), y_axis, y_dim)

        receptor_time = pd.Timestamp(footprint.receptor.time)
        times = pd.DatetimeIndex(pd.to_datetime(data["time"].values, utc=True))
        hours_f = (times.tz_convert(None) - receptor_time) / pd.Timedelta(hours=1)
        hours = np.asarray(hours_f, dtype=float)
        if not np.allclose(hours, np.round(hours)):
            raise ValueError(
                "Footprint time layers are not whole hours from the receptor time."
            )
        hours = np.round(hours).astype(np.int16)

        values = data.transpose("time", y_dim, x_dim).to_numpy()
        t, y, x = np.nonzero(np.nan_to_num(values, nan=0.0))
        table = pa.table(
            {
                "receptor": pa.array(
                    [str(footprint.receptor.id)] * len(t)
                ).dictionary_encode(),
                "hour": pa.array(hours[t]),
                "y": pa.array(yi[y].astype(np.int16)),
                "x": pa.array(xi[x].astype(np.int16)),
                "foot": pa.array(values[t, y, x].astype(np.float32)),
            },
            schema=_FOOTPRINT_SCHEMA,
        )
        table = table.replace_schema_metadata(
            self._metadata(footprint.receptor, footprint.name, hours.tolist(), "")
        )
        return _write_atomic_table(
            table, self.footprint_path(str(footprint.receptor.id))
        )

    def write_empty(self, receptor: Receptor, reason: str, name: str = "") -> Path:
        """Record that a receptor's footprint is empty (no particle over the grid), with the reason."""
        table = _FOOTPRINT_SCHEMA.empty_table().replace_schema_metadata(
            self._metadata(receptor, name, [], reason)
        )
        return _write_atomic_table(table, self.footprint_path(str(receptor.id)))

    @staticmethod
    def _metadata(
        receptor: Receptor, name: str, hours: list[int], empty_reason: str
    ) -> dict[bytes, bytes]:
        return {
            b"stilt:receptor": json.dumps(receptor.to_dict()).encode(),
            b"stilt:name": name.encode(),
            b"stilt:hours": json.dumps(hours).encode(),
            b"stilt:empty_reason": empty_reason.encode(),
        }

    def empty_reason(self, receptor_id: str) -> str | None:
        """Return why the receptor's footprint is empty, or ``None`` when it is not empty."""
        meta = pq.read_schema(self.footprint_path(receptor_id)).metadata or {}
        reason = meta.get(b"stilt:empty_reason", b"").decode()
        return reason or None

    def read(self, receptor_id: str) -> Footprint | None:
        """
        Read one receptor's footprint as a dense :class:`~stilt.Footprint`.

        Returns ``None`` for an empty footprint (see :meth:`empty_reason`).
        """
        table = pq.read_table(self.footprint_path(receptor_id))
        meta = table.schema.metadata or {}
        if meta.get(b"stilt:empty_reason", b""):
            return None
        receptor = Receptor.from_dict(json.loads(meta[b"stilt:receptor"]))
        hours = json.loads(meta[b"stilt:hours"])
        name = meta.get(b"stilt:name", b"").decode()

        x_axis, y_axis = self.axes
        values = np.zeros((len(hours), len(y_axis), len(x_axis)), dtype=np.float64)
        if table.num_rows:
            layer = {h: i for i, h in enumerate(hours)}
            t = np.fromiter(
                (layer[h] for h in table["hour"].to_numpy()),
                dtype=np.intp,
                count=table.num_rows,
            )
            values[t, table["y"].to_numpy(), table["x"].to_numpy()] = table[
                "foot"
            ].to_numpy()

        receptor_time = pd.Timestamp(receptor.time)
        times = pd.DatetimeIndex(
            [receptor_time + pd.Timedelta(hours=int(h)) for h in hours]
        )
        x_dim, y_dim = ("lon", "lat") if self.grid.is_longlat else ("x", "y")
        data = xr.DataArray(
            values,
            dims=["time", y_dim, x_dim],
            coords={"time": times, y_dim: y_axis, x_dim: x_axis},
            attrs={"units": "ppm m2 s umol-1"},
        )
        return Footprint(receptor=receptor, config=self.config, data=data, name=name)

    # -- many receptors at once --------------------------------------------

    def table(self, receptors: Iterable[str] | None = None) -> pa.Table:
        """
        Return the non-zero cells of many footprints as one table.

        Columns are ``receptor``, ``hour``, ``y``, ``x``, ``foot``, and
        ``date`` (the receptor date, from the folder, as ``date32``). With
        *receptors*, only those files are read.
        """
        files = _list_receptor_files(self.path, ".parquet")
        if receptors is not None:
            wanted = set(receptors)
            files = {rid: p for rid, p in files.items() if rid in wanted}
        if not files:
            return _FOOTPRINT_TABLE_SCHEMA.empty_table()
        dataset = pads.dataset(
            [str(p) for p in files.values()],
            format="parquet",
            partitioning=_DATE_PARTITIONING,
            partition_base_dir=str(self.path),
        )
        return dataset.to_table()

    def jacobian(
        self,
        target: Geometry,
        time_bins: pd.IntervalIndex,
        receptors: Iterable[str] | None = None,
    ) -> Jacobian:
        """
        Sum many footprints onto a target, per time bin, as one sparse matrix.

        The same operation as :meth:`stilt.Footprint.aggregate`, for every
        receptor in the folder (or *receptors*) at once. Each footprint cell
        is split among the target cells it overlaps in proportion to area,
        time layers are summed within each of *time_bins*, and cells or
        layers outside the target or the bins are dropped.

        Parameters
        ----------
        target : Grid, Mesh, or Zones
            Cells to sum onto.
        time_bins : pandas.IntervalIndex
            Left-closed time intervals, such as a flux inventory's steps.
        receptors : iterable of str, optional
            Receptor ids to include. All by default.

        Returns
        -------
        Jacobian
            Rows are receptors with a non-empty footprint; columns are
            ``(time bin, target cell)``.
        """
        files = _list_receptor_files(self.path, ".parquet")
        if receptors is None:
            requested = list(files)
            missing: list[str] = []
        else:
            requested = list(dict.fromkeys(receptors))
            missing = [r for r in requested if r not in files]
        present = [r for r in requested if r in files]

        x_axis, y_axis = self.axes
        grid = self.grid
        check_resolution(target, grid.xres, grid.yres, grid.projection)
        weights = overlap_weights(
            target, x_axis, y_axis, grid.xres, grid.yres, grid.projection
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

        table = (
            self.table(present) if present else _FOOTPRINT_TABLE_SCHEMA.empty_table()
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
                [np.datetime64(_receptor_time(r), "ns") for r in ids]
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


# -- converting an existing project ------------------------------------------


class ConvertReport(NamedTuple):
    """Counts from :func:`convert_project`."""

    particles: int
    particles_skipped: int
    footprints: int
    footprints_skipped: int
    logs: int
    incomplete: int


def convert_simulation(
    sim: Any, output: Output, met_config: MetConfig, *, verify: bool = True
) -> tuple[bool, bool, bool]:
    """
    Copy one simulation of the current layout into *output*.

    Returns ``(wrote_particles, wrote_footprint, wrote_log)``. Existing files
    are left alone, so the conversion can be resumed. With *verify*, each
    file written is read back and compared with its source.
    """
    variant: VariantConfig = sim.config
    run = output.run(variant.group, variant.transport_settings(met_config))
    rid = str(sim.receptor.id)

    wrote_particles = wrote_footprint = wrote_log = False
    traj_path = None if sim.is_derived else sim.resolve(sim.trajectories_path)
    if traj_path is not None and not run.has_particles(rid):
        traj = Trajectories.from_parquet(traj_path)
        run.write_particles(traj)
        if verify:
            back = run.read_particles(rid)
            if len(back.data) != len(traj.data) or not np.allclose(
                back.data["foot"].to_numpy(), traj.data["foot"].to_numpy()
            ):
                run.particles_path(rid).unlink()
                raise ValueError(f"{rid}: converted particles do not match the source.")
        wrote_particles = True

    if sim.makes_footprint:
        feet = run.footprints(sim.footprint_config, name=sim.variant)
        if not feet.has(rid):
            foot_path = sim.resolve(sim.footprint_path)
            if foot_path is not None:
                foot = Footprint.from_netcdf(foot_path)
                feet.write(foot)
                if verify:
                    back = feet.read(rid)
                    assert back is not None
                    if not np.allclose(
                        back.data.values,
                        np.nan_to_num(
                            foot.data.values.astype(np.float32).astype(np.float64)
                        ),
                    ):
                        feet.footprint_path(rid).unlink()
                        raise ValueError(
                            f"{rid}: converted footprint does not match the source."
                        )
                wrote_footprint = True
            elif sim.empty_reason is not None:
                feet.write_empty(sim.receptor, sim.empty_reason, name=sim.variant)
                wrote_footprint = True

    log_path = sim.resolve(sim.log_path)
    if log_path is not None and not run.log_path(rid).exists():
        run.write_log(rid, log_path.read_text())
        wrote_log = True
    return wrote_particles, wrote_footprint, wrote_log


def convert_project(
    model: Any,
    output: Output,
    *,
    receptors: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    verify: bool = True,
) -> ConvertReport:
    """
    Copy a project's ``simulations/by-id`` tree into an output directory.

    Every complete simulation of the model (or of the selected receptors and
    variants) is written to *output*: particles into the run for its
    transport settings, the footprint into that run's folder for its
    footprint settings, and the log beside them. Variants that only differ
    in footprint settings land in the same run. Files that already exist are
    skipped, so an interrupted conversion can be resumed. Incomplete
    simulations are counted and left out.

    Parameters
    ----------
    model : Model
        The project to convert.
    output : Output
        Where to write.
    receptors, variants : iterable of str, optional
        Receptor ids and variant names to convert. All by default.
    verify : bool, default True
        Read each written file back and compare it with its source.

    Returns
    -------
    ConvertReport
    """
    sims = model.simulations
    if receptors is not None or variants is not None:
        sims = sims.sel(receptor=receptors, variant=variants)
    counts = dict.fromkeys(ConvertReport._fields, 0)
    for sim in sims:
        if not sim.is_complete():
            counts["incomplete"] += 1
            continue
        wrote_particles, wrote_footprint, wrote_log = convert_simulation(
            sim, output, model.config.mets[sim.config.met], verify=verify
        )
        if not sim.is_derived:
            counts["particles" if wrote_particles else "particles_skipped"] += 1
        if sim.makes_footprint:
            counts["footprints" if wrote_footprint else "footprints_skipped"] += 1
        counts["logs"] += int(wrote_log)
    return ConvertReport(**counts)


__all__ = [
    "ConvertReport",
    "Footprints",
    "Jacobian",
    "Output",
    "Run",
    "convert_project",
    "convert_simulation",
]
