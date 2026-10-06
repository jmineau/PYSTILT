"""
A footprint as an array, and as a file.

A footprint is an :class:`xarray.DataArray` that carries its receptor and
settings in its attributes. This module builds that array, records the
settings as JSON, and reads and writes footprint files: the sparse Parquet
files of an output directory, and CF-1.8 NetCDF. :func:`open_footprints`
opens many stored footprints as one dataset.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from stilt._atomic import write_parquet
from stilt._paths import location, readable
from stilt.footprint.config import FootprintConfig
from stilt.identity import footprint_settings, read_footprint_settings
from stilt.receptors import Receptor
from stilt.spatial import Grid, _with_cf_grid, horizontal_dims

if TYPE_CHECKING:
    from upath import UPath

    from stilt._paths import Location


def _naive_utc(values: Any) -> pd.DatetimeIndex:
    """Return times as a naive DatetimeIndex in UTC; naive input is taken as UTC."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_localize(None)


def _naive_utc_ns(values: Any) -> np.ndarray:
    """Return times as nanoseconds since the epoch, in UTC; naive times are taken as UTC."""
    return np.asarray(_naive_utc(values), dtype="datetime64[ns]").astype(np.int64)


def _footprint_array(
    values: np.ndarray,
    hours: Any,
    receptor: Receptor,
    config: FootprintConfig,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    geometry_hash: str | None = None,
) -> xr.DataArray:
    """
    Return a footprint: *values* in ``(time, y, x)`` order, at cell centres *x* and *y*.

    Layer ``k`` is the hour from ``hours[k]`` hours after the receptor time,
    and is stamped there, as in STILT-R; a backward run's first hour is
    layer -1, and a time-integrated footprint is the one layer 0. This is
    the one place a footprint array is made, for :func:`calc_footprint` and for
    :func:`read_footprint` alike.
    """
    if config.grid is None:
        raise ValueError("Footprint settings need a grid.")
    times = _naive_utc([receptor.time + pd.Timedelta(hours=int(h)) for h in hours])
    y_dim, x_dim = config.grid.dims
    data = xr.DataArray(
        values,
        dims=["time", y_dim, x_dim],
        coords={"time": times, y_dim: y, x_dim: x},
    )
    return _describe(data, receptor, config, name, geometry_hash)


def _with_cf_metadata(ds: xr.Dataset, *, grid: Grid) -> xr.Dataset:
    """Add CF coordinate and CRS attributes to a footprint dataset."""
    ds = _with_cf_grid(ds, grid.crs)
    ds["foot"].attrs["grid_mapping"] = "crs"
    if "time" in ds.coords:
        ds["time"].attrs.update({"standard_name": "time", "axis": "T"})
    return ds


#: Units of a footprint: ppm per (µmol m⁻² s⁻¹).
UNITS = "ppm m2 s umol-1"


def _settings_json(config: FootprintConfig, geometry_hash: str | None) -> str:
    """Return a footprint's settings as JSON (:func:`stilt.identity.footprint_settings`)."""
    return json.dumps(footprint_settings(config, geometry_hash))


def _describe(
    data: xr.DataArray,
    receptor: Receptor,
    config: FootprintConfig,
    name: str,
    geometry_hash: str | None = None,
) -> xr.DataArray:
    """
    Name a footprint array and attach what it belongs to.

    The receptor id is a scalar coordinate, which xarray keeps through
    arithmetic in every version. The full receptor and the settings are
    attributes, which the ``.stilt`` accessor reads.
    """
    return (
        data.rename("foot")
        .assign_coords(receptor=str(receptor.id))
        .assign_attrs(
            {
                "units": UNITS,
                "long_name": "footprint",
                "stilt_name": name,
                "stilt_receptor": receptor.to_json(),
                "stilt_footprint": _settings_json(config, geometry_hash),
            }
        )
    )


def _from_sparse_table(
    table: Any, config: FootprintConfig, geometry_hash: str | None = None
) -> xr.DataArray | None:
    """Return the dense footprint of a stored sparse table, or ``None`` when it is empty."""
    meta = table.schema.metadata or {}
    if meta.get(b"stilt:empty_reason", b""):
        return None
    receptor = Receptor.from_json(meta[b"stilt:receptor"])
    hours = json.loads(meta[b"stilt:hours"])
    name = meta.get(b"stilt:name", b"").decode()
    grid = config.grid
    if grid is None:
        raise ValueError("A stored footprint needs settings with a grid.")

    x_axis, y_axis = grid.axes
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

    return _footprint_array(
        values, hours, receptor, config, name, x_axis, y_axis, geometry_hash
    )


def read_footprint(
    path: str | Path | UPath,
    *,
    chunks: Any | None = None,
) -> xr.DataArray | None:
    """
    Read a footprint file, NetCDF or the Parquet files PYSTILT stores.

    Each file holds its receptor and settings, so it needs nothing else.

    Parameters
    ----------
    path : str or Path
        A ``.nc`` file written by ``foot.stilt.to_netcdf``, or a footprint
        file from an output directory.
    chunks : dict, int or "auto", optional
        For a NetCDF file, passed to :func:`xarray.open_dataset` to load the
        data lazily with dask.

    Returns
    -------
    xarray.DataArray or None
        The footprint, or ``None`` for an empty one (no particle reached the
        grid).

    Examples
    --------
    >>> foot = stilt.read_footprint(sim.footprint_path)
    >>> foot.stilt.receptor
    """
    import pyarrow.parquet as pq

    path = location(path)
    if path.suffix == ".nc":
        if not isinstance(path, Path):
            raise ValueError(
                f"{path}: a NetCDF footprint is read from this filesystem. "
                "Copy it here first."
            )
        if chunks is not None:
            foot = xr.open_dataset(path, chunks=chunks)["foot"]
        else:
            with xr.open_dataset(path) as ds:
                foot = ds["foot"].load()
        foot.attrs.pop("grid_mapping", None)
        return foot
    # One file alone: pq.read_table would add the settings= and date=
    # folder names of an output directory as columns.
    with readable(path) as source:
        table = pq.ParquetFile(source).read()
    stored = (table.schema.metadata or {}).get(b"stilt:footprint")
    if stored is None:
        raise ValueError(f"{path} does not record its footprint settings.")
    config, geometry_hash = read_footprint_settings(json.loads(stored), path.name)
    return _from_sparse_table(table, config, geometry_hash)


def _empty_reason(path: str | Path | UPath) -> str | None:
    """Return why a stored footprint is empty, or ``None`` when it is not, from its metadata alone."""
    import pyarrow.parquet as pq

    with readable(path) as source:
        meta = pq.read_schema(source).metadata or {}
    return meta.get(b"stilt:empty_reason", b"").decode() or None


#: About how many bytes of footprint values one chunk of :func:`open_footprints` holds.
_CHUNK_BYTES = 128 * 2**20


def _footer(path: Location) -> dict[bytes, bytes]:
    """Return the metadata of a stored footprint, reading only its footer."""
    import pyarrow.parquet as pq

    with readable(path) as source:
        return dict(pq.read_schema(source).metadata or {})


def _dense_block(
    paths: list[Location], hours: np.ndarray, ny: int, nx: int
) -> np.ndarray:
    """Read stored footprints into one ``(receptor, hour, y, x)`` block of ``float32``."""
    import pyarrow.parquet as pq

    block = np.zeros((len(paths), len(hours), ny, nx), dtype=np.float32)
    for i, path in enumerate(paths):
        with readable(path) as source:
            table = pq.ParquetFile(source).read(columns=["hour", "y", "x", "foot"])
        layer = table["hour"].to_numpy().astype(np.intp) - int(hours[0])
        if layer.size and (layer.min() < 0 or layer.max() >= len(hours)):
            raise ValueError(f"{path} has cells outside the hours it records.")
        block[i, layer, table["y"].to_numpy(), table["x"].to_numpy()] = table[
            "foot"
        ].to_numpy()
    return block


def open_footprints(
    paths: Iterable[str | Path | UPath], *, workers: int | None = None
) -> xr.Dataset:
    """
    Open stored footprints as one dataset, stacked on the hour from each receptor's time.

    Footprints of many receptors are stacked by their hour offset rather
    than by absolute time, so receptors at different times share one
    ``hour`` axis. Only the files' metadata is read here. The values are
    read with dask when they are used, a block of receptors from one date
    folder at a time. :meth:`stilt.Project.footprints` opens a selection
    of a project's footprints this way.

    Parameters
    ----------
    paths : iterable of str or Path
        Footprint files from an output directory, all made with the same
        settings (one variant's). URLs read from an object store.
    workers : int, optional
        Threads that read the files' metadata. Defaults to the number of
        CPUs.

    Returns
    -------
    xarray.Dataset
        ``foot`` with dims ``(receptor, hour, lat, lon)`` (``y`` and ``x``
        on a projected grid), as ``float32`` like the stored values. ``hour``
        is the start of each layer, in hours after the receptor time, so a
        backward run's first hour is -1. The ``time`` coordinate, with dims
        ``(receptor, hour)``, is that start as a date and time. A receptor
        whose footprint is empty has no row; ``attrs["empty"]`` lists them.

    Raises
    ------
    ValueError
        If there are no files, or they were made with different settings.

    Examples
    --------
    >>> ds = open_footprints(sorted(folder.glob("date=2024-07-*/*.parquet")))
    >>> ds.foot.sum("hour").mean("receptor").plot()
    """
    import dask.array as da
    from dask.delayed import delayed

    paths = [location(p) for p in paths]
    if not paths:
        raise ValueError("No footprint files to open.")
    threads = workers if workers is not None else (os.cpu_count() or 1)
    if threads > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            footers = list(pool.map(_footer, paths))
    else:
        footers = [_footer(p) for p in paths]

    stored = {meta.get(b"stilt:footprint") for meta in footers}
    if None in stored:
        raise ValueError("A footprint file does not record its settings.")
    if len(stored) > 1:
        raise ValueError(
            "These footprints were made with different settings. Open one "
            "variant's footprints at a time."
        )
    config, geometry_hash = read_footprint_settings(
        json.loads(stored.pop() or b"{}"), paths[0].name
    )
    grid = config.grid
    if grid is None:
        raise ValueError("Stored footprints need settings with a grid.")

    receptors = [Receptor.from_json(meta[b"stilt:receptor"]) for meta in footers]
    is_empty = [bool(meta.get(b"stilt:empty_reason", b"")) for meta in footers]
    kept = [i for i, empty in enumerate(is_empty) if not empty]
    recorded = [h for i in kept for h in json.loads(footers[i][b"stilt:hours"])]
    hours = (
        np.arange(min(recorded), max(recorded) + 1)
        if recorded
        else np.zeros(0, dtype=int)
    )

    x_axis, y_axis = grid.axes
    y_dim, x_dim = grid.dims
    ny, nx = len(y_axis), len(x_axis)
    per_block = max(1, _CHUNK_BYTES // max(1, len(hours) * ny * nx * 4))
    blocks = []
    for _, run in itertools.groupby(kept, key=lambda i: paths[i].parent):
        ids = list(run)
        for start in range(0, len(ids), per_block):
            files = [paths[i] for i in ids[start : start + per_block]]
            read = delayed(_dense_block, pure=True)(files, hours, ny, nx)
            shape = (len(files), len(hours), ny, nx)
            blocks.append(da.from_delayed(read, shape=shape, dtype=np.float32))
    values = (
        da.concatenate(blocks, axis=0)
        if blocks
        else np.zeros((0, len(hours), ny, nx), dtype=np.float32)
    )

    start_times = _naive_utc([receptors[i].time for i in kept])
    times = start_times.to_numpy()[:, None] + hours.astype("timedelta64[h]")[None, :]
    foot = xr.DataArray(
        values,
        dims=("receptor", "hour", y_dim, x_dim),
        coords={
            "receptor": [str(receptors[i].id) for i in kept],
            "hour": hours,
            y_dim: y_axis,
            x_dim: x_axis,
            "time": (("receptor", "hour"), times),
        },
        name="foot",
        attrs={"units": UNITS, "long_name": "footprint"},
    )
    ds = foot.to_dataset()
    ds.attrs.update(
        {
            "stilt_name": footers[0].get(b"stilt:name", b"").decode(),
            "stilt_footprint": _settings_json(config, geometry_hash),
            "empty": [
                str(r.id) for r, empty in zip(receptors, is_empty, strict=True) if empty
            ],
        }
    )
    realization = footers[0].get(b"stilt:realization")
    if realization is not None:
        ds.attrs["realization"] = int(realization)
    return _with_cf_metadata(ds, grid=grid)


#: The columns of a stored footprint: its non-zero cells, indexed into its grid.
FOOTPRINT_SCHEMA = pa.schema(
    [
        ("receptor", pa.dictionary(pa.int32(), pa.string())),
        ("hour", pa.int16()),
        ("y", pa.int16()),
        ("x", pa.int16()),
        ("foot", pa.float32()),
    ]
)


def _cell_indices(coords: np.ndarray, axis: np.ndarray, name: str) -> np.ndarray:
    """Return the index of each coordinate's cell on a regular grid axis, or raise."""
    step = axis[1] - axis[0] if len(axis) > 1 else 1.0
    idx = np.rint((coords - axis[0]) / step).astype(np.intp)
    if (
        idx.min() < 0
        or idx.max() >= len(axis)
        or not np.allclose(axis[idx], coords, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"Footprint {name} coordinates are not cells of the grid.")
    return idx


def _file_metadata(
    receptor: Receptor,
    config: FootprintConfig,
    name: str,
    hours: list[int],
    empty_reason: str,
    metadata: dict[bytes, bytes] | None,
    geometry_hash: str | None = None,
) -> dict[bytes, bytes]:
    """Return what a footprint file records so that it reads alone."""
    return {
        b"stilt:receptor": receptor.to_json().encode(),
        b"stilt:name": name.encode(),
        b"stilt:hours": json.dumps(hours).encode(),
        b"stilt:empty_reason": empty_reason.encode(),
        b"stilt:footprint": _settings_json(config, geometry_hash).encode(),
        **(metadata or {}),
    }


def write_footprint(
    path: str | Path | UPath,
    foot: xr.DataArray,
    config: FootprintConfig | None = None,
    metadata: dict[bytes, bytes] | None = None,
    geometry_hash: str | None = None,
) -> Location:
    """
    Write a footprint to a Parquet file that :func:`read_footprint` reads alone.

    Only the non-zero cells are stored, as ``hour`` (the offset of the time
    layer from the receptor time), ``y`` and ``x`` (the cell's place on the
    grid), and ``foot`` in float32. Every layer is recorded in the metadata,
    so the dense array reads back with the same shape. The metadata also
    holds the receptor and the footprint settings.

    Parameters
    ----------
    path : str or Path
        File to write.
    foot : xarray.DataArray
        The footprint, as :func:`calc_footprint` returns it.
    config : FootprintConfig, optional
        Settings to record, whose grid the footprint must be on. Defaults to
        the footprint's own, ``foot.stilt.config``.
    metadata : dict, optional
        More file metadata, such as the settings hash.
    geometry_hash : str, optional
        Hash of the geometry the grid was derived for. Defaults to the
        footprint's own, ``foot.stilt.geometry_hash``.

    Returns
    -------
    Path
        The path written to.

    Raises
    ------
    ValueError
        If the footprint is not on the grid, or a time layer is not a whole
        number of hours from the receptor time.
    """
    settings = foot.stilt.config if config is None else config
    if settings.grid is None:
        raise ValueError("Footprint settings need a grid.")
    receptor = foot.stilt.receptor
    y_dim, x_dim = horizontal_dims(foot)
    x_axis, y_axis = settings.grid.axes
    xi = _cell_indices(np.asarray(foot[x_dim].values, dtype=float), x_axis, x_dim)
    yi = _cell_indices(np.asarray(foot[y_dim].values, dtype=float), y_axis, y_dim)

    times = _naive_utc(foot["time"].values)
    hours_f = (times - pd.Timestamp(receptor.time)) / pd.Timedelta(hours=1)
    hours = np.asarray(hours_f, dtype=float)
    if not np.allclose(hours, np.round(hours)):
        raise ValueError(
            "Footprint time layers are not whole hours from the receptor time."
        )
    hours = np.round(hours).astype(np.int16)

    values = foot.transpose("time", y_dim, x_dim).to_numpy()
    t, y, x = np.nonzero(np.nan_to_num(values, nan=0.0))
    table = pa.table(
        {
            "receptor": pa.array([str(receptor.id)] * len(t)).dictionary_encode(),
            "hour": pa.array(hours[t]),
            "y": pa.array(yi[y].astype(np.int16)),
            "x": pa.array(xi[x].astype(np.int16)),
            "foot": pa.array(values[t, y, x].astype(np.float32)),
        },
        schema=FOOTPRINT_SCHEMA,
    )
    meta = _file_metadata(
        receptor,
        settings,
        foot.stilt.name,
        hours.tolist(),
        "",
        metadata,
        foot.stilt.geometry_hash if geometry_hash is None else geometry_hash,
    )
    return write_parquet(table.replace_schema_metadata(meta), location(path))


def write_empty_footprint(
    path: str | Path | UPath,
    receptor: Receptor,
    reason: str,
    config: FootprintConfig,
    name: str = "",
    metadata: dict[bytes, bytes] | None = None,
    geometry_hash: str | None = None,
) -> Location:
    """
    Record in a footprint file that a receptor's footprint is empty, and why.

    The file has no rows; :func:`read_footprint` returns ``None`` for it.
    ``reason`` is, for example, ``"outside_domain"``.
    """
    meta = _file_metadata(receptor, config, name, [], reason, metadata, geometry_hash)
    table = FOOTPRINT_SCHEMA.empty_table().replace_schema_metadata(meta)
    return write_parquet(table, location(path))
