"""
A footprint as an array, and as a file.

A footprint is an :class:`xarray.DataArray` that carries its receptor and
settings in its attributes. This module builds that array, records the
settings as JSON, and reads and writes footprint files: the sparse Parquet
files of an output directory, and CF-1.8 NetCDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from stilt._atomic import write_parquet
from stilt.footprint.config import FootprintConfig
from stilt.identity import footprint_settings, read_footprint_settings
from stilt.receptors import Receptor
from stilt.spatial import Grid, _with_cf_grid, horizontal_dims


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
    the one place a footprint array is made, for :func:`calculate` and for
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
    path: str | Path,
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

    path = Path(path)
    if path.suffix == ".nc":
        if chunks is not None:
            foot = xr.open_dataset(path, chunks=chunks)["foot"]
        else:
            with xr.open_dataset(path) as ds:
                foot = ds["foot"].load()
        foot.attrs.pop("grid_mapping", None)
        return foot
    # One file alone: pq.read_table would add the settings= and date=
    # folder names of an output directory as columns.
    table = pq.ParquetFile(path).read()
    stored = (table.schema.metadata or {}).get(b"stilt:footprint")
    if stored is None:
        raise ValueError(f"{path} does not record its footprint settings.")
    config, geometry_hash = read_footprint_settings(json.loads(stored), path.name)
    return _from_sparse_table(table, config, geometry_hash)


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
    path: str | Path,
    foot: xr.DataArray,
    config: FootprintConfig | None = None,
    metadata: dict[bytes, bytes] | None = None,
    geometry_hash: str | None = None,
) -> Path:
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
        The footprint, as :func:`calculate` returns it.
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
    return write_parquet(table.replace_schema_metadata(meta), Path(path))


def write_empty_footprint(
    path: str | Path,
    receptor: Receptor,
    reason: str,
    config: FootprintConfig,
    name: str = "",
    metadata: dict[bytes, bytes] | None = None,
    geometry_hash: str | None = None,
) -> Path:
    """
    Record in a footprint file that a receptor's footprint is empty, and why.

    The file has no rows; :func:`read_footprint` returns ``None`` for it.
    ``reason`` is, for example, ``"outside_domain"``.
    """
    meta = _file_metadata(receptor, config, name, [], reason, metadata, geometry_hash)
    table = FOOTPRINT_SCHEMA.empty_table().replace_schema_metadata(meta)
    return write_parquet(table, Path(path))
