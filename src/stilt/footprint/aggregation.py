"""
Summing footprints onto the cells of another grid or set of polygons, per time bin.

:func:`aggregate` does it for one footprint (``foot.stilt.aggregate``) and
:func:`jacobian` for many at once. Both hand their footprint cells, each
with its time bin, to :func:`_sum_onto`, which applies the overlap
weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from scipy import sparse

from stilt.footprint.config import FootprintConfig
from stilt.receptors import parse_receptor_id
from stilt.spatial import Grid, horizontal_dims

from .io import UNITS, _naive_utc, _naive_utc_ns
from .targets import Geometry, Mesh, Zones, check_resolution, overlap_weights


def _bin_edges(time_bins: pd.IntervalIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the left and right edges of *time_bins*, as :func:`_naive_utc_ns` gives them.

    Raises
    ------
    ValueError
        If the bins are not closed on the left.
    """
    if time_bins.closed != "left":
        raise ValueError(
            f"time_bins must be closed on the left, not {time_bins.closed!r}. "
            "A footprint time is the start of its hour, so each bin takes "
            "the hours that start in it. Build the bins with "
            "closed='left', for example "
            "pd.interval_range(start, end, freq='1h', closed='left')."
        )
    return _naive_utc_ns(time_bins.left), _naive_utc_ns(time_bins.right)


def _time_bin(times_ns: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the bin each time falls in, or -1 for a time in no bin."""
    b = np.searchsorted(left, times_ns, side="right") - 1
    inside = (b >= 0) & (times_ns < right[np.clip(b, 0, len(left) - 1)])
    return np.where(inside, b, -1)


def _require_geometry(target: object) -> Geometry:
    """Return *target*, raising unless it is a geometry a footprint can be summed onto."""
    if not isinstance(target, (Grid, Mesh, Zones)):
        raise TypeError(
            "The target must be a stilt.Grid, stilt.Mesh, or stilt.Zones, "
            f"not {type(target).__name__}."
        )
    return target


def _target_weights(
    target: object,
    grid: Grid,
    geometry_hash: str | None,
    x: np.ndarray,
    y: np.ndarray,
    name: str = "",
) -> sparse.csr_matrix:
    """
    Return the overlap weights of footprint cells at *x*, *y* onto *target*, raster cells by target cells.

    Raises when the target is not a geometry, warns when it is not the mesh
    the footprint grid was chosen for, and warns when the grid is too coarse
    for the target's cells.
    """
    target = _require_geometry(target)
    expected = geometry_hash
    mesh = target.base if isinstance(target, Zones) else target
    if expected and isinstance(mesh, Mesh) and mesh.hash != expected:
        warnings.warn(
            f"Footprint {name!r} was derived for geometry {expected} but is "
            f"being aggregated onto geometry {mesh.hash}; the geometry source may "
            "have changed since the footprint was computed, so the raster "
            "resolution and extent may no longer suit it.",
            stacklevel=4,
        )
    check_resolution(target, grid.xres, grid.yres, grid.crs)
    return overlap_weights(target, x, y, grid.xres, grid.yres, grid.crs).T.tocsr()


def _sum_onto(
    weights: sparse.csr_matrix,
    n_rows: int,
    n_bins: int,
    row: np.ndarray,
    bin_idx: np.ndarray,
    cell: np.ndarray,
    foot: np.ndarray,
) -> sparse.csr_matrix:
    """
    Sum footprint cells onto a target, per time bin.

    Each footprint cell is given by its row (receptor), its time bin (-1
    for none), its flat raster index (``y * nx + x``), and its value.
    *weights* maps raster cells (rows) onto target cells (columns).

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_rows, n_bins * n_cells)``: every target cell for the
        first bin, then the second. Cells outside the bins are dropped.
    """
    keep = bin_idx >= 0
    n_raster, n_cells = weights.get_shape()
    # One row per (receptor, bin), summed onto the target in one product,
    # then laid out as (receptor) by (bin, target cell).
    per_bin = sparse.csr_matrix(
        (
            foot[keep],
            (row[keep].astype(np.int64) * n_bins + bin_idx[keep], cell[keep]),
        ),
        shape=(n_rows * n_bins, n_raster),
    )
    onto = per_bin @ weights
    return sparse.csr_matrix(onto.reshape(n_rows, n_bins * n_cells))


def _sum_table(
    table: pa.Table,
    receptors: list[str],
    weights: sparse.csr_matrix,
    nx: int,
    edges: tuple[np.ndarray, np.ndarray],
) -> tuple[sparse.csr_matrix, list[str]]:
    """
    Sum the footprint cells in *table* onto a target, a row per receptor.

    Returns the rows of the receptors of *receptors* that have cells in the
    table, in that order, and their ids. A receptor in *receptors* without
    cells is an empty footprint and gets no row; cells of receptors not in
    *receptors* are left out.
    """
    n_bins = len(edges[0])
    if not table.num_rows:
        return sparse.csr_matrix((0, n_bins * weights.get_shape()[1])), []
    # Work with the dictionary indices of the receptor column: one small
    # array of ids, and an int32 per cell.
    table = table.unify_dictionaries().combine_chunks()
    receptor_col = table.column("receptor").combine_chunks()
    ids: list[str] = receptor_col.dictionary.to_pylist()
    dict_idx = receptor_col.indices.to_numpy()
    found = set(ids)
    rows = [r for r in receptors if r in found]
    row_of = {r: i for i, r in enumerate(rows)}
    row_by_id = np.array([row_of.get(r, -1) for r in ids], dtype=np.int32)

    # A receptor has tens of hours and its cells hundreds of thousands, so
    # find the bin of each receptor-hour once and look it up per cell.
    hour = table.column("hour").to_numpy()
    first = int(hour.min())
    hours = np.arange(first, int(hour.max()) + 1, dtype=np.int64)
    release = _naive_utc_ns([parse_receptor_id(r)[0] for r in ids])
    starts = release[:, None] + hours[None, :] * 3_600_000_000_000
    bin_of = _time_bin(starts.ravel(), *edges).reshape(starts.shape).astype(np.int32)
    bin_of[row_by_id < 0] = -1  # receptors in the table that were not asked for
    bin_idx = bin_of[dict_idx, hour.astype(np.int32) - first]

    cell = table.column("y").to_numpy().astype(np.int32) * nx + table.column(
        "x"
    ).to_numpy().astype(np.int32)
    foot = table.column("foot").to_numpy().astype(np.float64)
    data = _sum_onto(
        weights, len(rows), n_bins, row_by_id[dict_idx], bin_idx, cell, foot
    )
    return data, rows


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
        Each column's time bin and target cell. The first level, ``time``,
        is the left edge of the time bin; the others are the target's cell
        index: ``lon`` and ``lat`` (``x`` and ``y`` when projected) for a
        grid, ``cell`` for a mesh or zones.
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

    def to_frame(self, sparse: bool = False) -> pd.DataFrame:
        """
        Return the matrix as a DataFrame, receptors by :attr:`columns`.

        Parameters
        ----------
        sparse : bool, default False
            Hold the values in pandas' sparse columns, built from the sparse
            matrix with no dense copy (``DataFrame.sparse``). By default
            they are a dense array.
        """
        if sparse:
            # pandas-stubs type the DataFrame.sparse accessor as `...`
            # pyrefly: ignore[missing-attribute]
            stored = pd.DataFrame.sparse.from_spmatrix(self.data)
            # A cell not stored is zero. pandas 3 fills from_spmatrix's
            # columns with NaN, so each is rebuilt with a fill of 0.
            columns = {
                j: pd.arrays.SparseArray(
                    stored[j].array.sp_values,
                    sparse_index=stored[j].array.sp_index,
                    fill_value=0.0,
                )
                for j in stored.columns
            }
            frame = pd.DataFrame(columns, index=self.receptors)
            frame.columns = self.columns
            return frame
        return pd.DataFrame(
            self.data.toarray(), index=self.receptors, columns=self.columns
        )

    def to_xarray(self, dense: bool = False) -> xr.DataArray:
        """
        Return the matrix as a DataArray with dims ``(receptor, time, cell)``.

        ``time`` is the left edge of each time bin and ``cell`` the target
        cell, as in ``columns``. For a grid, ``cell`` has the coordinates
        ``lon`` and ``lat`` (``x`` and ``y`` when projected), and
        ``.unstack("cell")`` makes them dimensions. The receptors with an
        empty footprint and those not run yet are the attributes ``empty``
        and ``missing``.

        Parameters
        ----------
        dense : bool, default False
            Hold the values in a NumPy array. By default they are a
            ``sparse.COO`` array from the optional ``sparse`` package
            (``pip install pystilt[sparse]``), whose entries not stored are
            zeros.

        Examples
        --------
        >>> H = project.jacobian(july, zones, bins).to_xarray()
        >>> (H * flux).sum(["time", "cell"])  # ppm at each receptor
        """
        times = pd.unique(self.columns.get_level_values("time"))
        n_cells = len(self.columns) // len(times) if len(times) else 0
        cells = self.columns.droplevel("time")[:n_cells]
        shape = (len(self.receptors), len(times), n_cells)
        if dense:
            values = self.data.toarray().reshape(shape)
        else:
            try:
                import sparse as pydata_sparse
            except ImportError as error:
                raise ImportError(
                    "Jacobian.to_xarray() needs the sparse package: pip install "
                    "pystilt[sparse]. dense=True returns a NumPy array instead."
                ) from error
            values = pydata_sparse.COO.from_scipy_sparse(self.data).reshape(shape)
        if isinstance(cells, pd.MultiIndex):
            cell_coords = xr.Coordinates.from_pandas_multiindex(cells, "cell")
        else:
            cell_coords = xr.Coordinates({"cell": np.asarray(cells)})
        coords = xr.Coordinates({"receptor": self.receptors, "time": times})
        return xr.DataArray(
            values,
            dims=("receptor", "time", "cell"),
            coords=coords.merge(cell_coords).coords,
            name="jacobian",
            attrs={
                "units": UNITS,
                "empty": list(self.empty),
                "missing": list(self.missing),
            },
        )


def jacobian(
    table: pa.Table,
    config: FootprintConfig,
    target: Geometry,
    time_bins: pd.IntervalIndex,
    receptors: list[str],
    missing: list[str] | None = None,
    geometry_hash: str | None = None,
) -> Jacobian:
    """
    Sum the footprints in *table* onto a target, per time bin, as one sparse matrix.

    The same operation as ``foot.stilt.aggregate``, for many footprints at
    once. *table* is what :meth:`stilt.output.Output.table` reads: the
    stored non-zero cells of each receptor, all on the grid of *config*.
    :meth:`stilt.Project.jacobian` reads and sums a project's
    footprints in batches, so a selection of any size fits in memory.

    Parameters
    ----------
    table : pyarrow.Table
        Footprint cells, with ``receptor``, ``hour``, ``y``, ``x``, ``foot``.
    config : FootprintConfig
        The footprints' settings, whose grid the cells index.
    target : Grid, Mesh, or Zones
        Cells to sum onto.
    time_bins : pandas.IntervalIndex
        Time intervals, closed on the left.
    receptors : list of str
        Receptors to give rows, in row order. Those with no cells in *table*
        are empty footprints and get none.
    missing : list of str, optional
        Receptors asked for that have no footprint file, to report.
    geometry_hash : str, optional
        Hash of the geometry the grid was derived for, to warn when *target*
        is another mesh.

    Returns
    -------
    Jacobian
    """
    return _jacobian(
        lambda batch: table,
        [receptors],
        config,
        target,
        time_bins,
        missing=missing,
        geometry_hash=geometry_hash,
    )


def _jacobian(
    read: Callable[[list[str]], pa.Table],
    batches: list[list[str]],
    config: FootprintConfig,
    target: Geometry,
    time_bins: pd.IntervalIndex,
    *,
    missing: list[str] | None = None,
    geometry_hash: str | None = None,
    workers: int = 1,
) -> Jacobian:
    """
    Sum footprints onto a target batch by batch, and stack the rows into one Jacobian.

    ``read(batch)`` returns the footprint cells of a batch of receptors.
    Batches are read and summed in *workers* threads, so only that many are
    in memory at once; the rows come back in batch order.
    """
    edges = _bin_edges(time_bins)  # raises for bins not closed on the left
    grid = config.grid
    if grid is None:
        raise ValueError("The footprint settings have no grid.")
    x_axis, y_axis = grid.axes
    weights = _target_weights(target, grid, geometry_hash, x_axis, y_axis)

    def one(batch: list[str]) -> tuple[sparse.csr_matrix, list[str]]:
        return _sum_table(read(batch), batch, weights, len(x_axis), edges)

    if workers <= 1 or len(batches) <= 1:
        parts = [one(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(one, batches))
    n_columns = len(time_bins) * len(target.index)
    blocks = [data for data, ids in parts if ids]
    data = (
        sparse.csr_matrix(sparse.vstack(blocks, format="csr"))
        if blocks
        else sparse.csr_matrix((0, n_columns))
    )
    rows = [r for _, ids in parts for r in ids]
    found = set(rows)
    empty = [r for batch in batches for r in batch if r not in found]

    columns = _columns(_naive_utc(time_bins.left), target.index)
    return Jacobian(
        data, pd.Index(rows, name="receptor"), columns, empty, list(missing or [])
    )


def _columns(times: pd.Index, cells: pd.Index) -> pd.MultiIndex:
    """
    Return the Jacobian's columns: each time bin, then each target cell within it.

    The levels are ``time`` and the target's own index levels (``lon`` and
    ``lat`` for a longitude/latitude grid, ``cell`` for a mesh or zones).
    Built from arrays, since a grid by its time bins can be millions of
    columns.
    """
    n_times, n_cells = len(times), len(cells)
    levels = (
        cells
        if isinstance(cells, pd.MultiIndex)
        else pd.MultiIndex.from_arrays([cells])
    )
    arrays = [np.repeat(np.asarray(times), n_cells)] + [
        np.tile(levels.get_level_values(i).to_numpy(), n_times)
        for i in range(levels.nlevels)
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["time", *levels.names])


def aggregate(
    foot: xr.DataArray, target: Geometry, time_bins: pd.IntervalIndex
) -> pd.DataFrame:
    """
    Sum a footprint onto the cells of a target, per time bin.

    The body of ``foot.stilt.aggregate``, which documents it. :func:`jacobian`
    does the same for many footprints at once.
    """
    edges = _bin_edges(time_bins)  # raises for bins not closed on the left
    y_dim, x_dim = horizontal_dims(foot)
    _require_geometry(target)
    if "time" not in foot.dims:
        raise ValueError(
            "The footprint has no time dimension, so its hours cannot be put "
            "in time bins. Aggregate the footprint before summing it over time."
        )
    columns = _naive_utc(time_bins.left)
    if foot.size == 0:
        return pd.DataFrame(0.0, index=target.index, columns=columns)
    x = np.asarray(foot[x_dim].values, dtype=float)
    y = np.asarray(foot[y_dim].values, dtype=float)
    weights = _target_weights(
        target, foot.stilt.grid, foot.stilt.geometry_hash, x, y, foot.stilt.name
    )

    # The footprint's non-zero cells, as one receptor's row of a Jacobian.
    data = foot.transpose("time", y_dim, x_dim).to_numpy()
    t, iy, ix = np.nonzero(data)
    layer_bin = _time_bin(_naive_utc_ns(foot["time"].values), *edges)
    summed = _sum_onto(
        weights,
        1,
        len(time_bins),
        np.zeros(len(t), dtype=np.int64),
        layer_bin[t],
        iy.astype(np.int64) * len(x) + ix,
        data[t, iy, ix].astype(np.float64),
    )
    n_cells = len(target.index)
    by_bin = summed.toarray().reshape(len(columns), n_cells).T
    return pd.DataFrame(by_bin, index=target.index, columns=columns)
