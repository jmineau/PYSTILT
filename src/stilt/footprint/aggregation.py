"""
Summing footprints onto the cells of another grid or set of polygons, per time bin.

:func:`aggregate` does it for one footprint (``foot.stilt.aggregate``) and
:func:`jacobian` for many at once. Both hand their footprint cells to
:func:`_sum_onto`, which bins the hours and applies the overlap weights.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from scipy import sparse

from stilt.footprint.config import FootprintConfig
from stilt.receptors import parse_receptor_id
from stilt.spatial import Grid, horizontal_dims

from .io import _naive_utc, _naive_utc_ns
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
    config: FootprintConfig,
    geometry_hash: str | None,
    x: np.ndarray,
    y: np.ndarray,
    name: str = "",
) -> sparse.csr_matrix:
    """
    Return the overlap weights of footprint cells at *x*, *y* onto *target*.

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
    grid = config.grid
    if grid is None:
        raise ValueError("The footprint settings have no grid.")
    check_resolution(target, grid.xres, grid.yres, grid.crs)
    return overlap_weights(target, x, y, grid.xres, grid.yres, grid.crs)


def _sum_onto(
    weights: sparse.csr_matrix,
    n_raster: int,
    time_bins: pd.IntervalIndex,
    n_rows: int,
    row: np.ndarray,
    time_ns: np.ndarray,
    cell: np.ndarray,
    foot: np.ndarray,
) -> sparse.csr_matrix:
    """
    Sum footprint cells onto a target, per time bin.

    Each footprint cell is given by its row (receptor), its hour's start
    time in naive UTC nanoseconds, its flat raster index (``y * nx + x``),
    and its value. *weights* maps the *n_raster* raster cells onto the
    target's cells.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_rows, n_bins * n_cells)``: every target cell for the
        first bin, then the second. Cells outside the bins are dropped.
    """
    left_ns, right_ns = _bin_edges(time_bins)
    n_bins = len(time_bins)
    bin_idx = _time_bin(time_ns, left_ns, right_ns)
    keep = bin_idx >= 0
    # F: (row) x (bin, raster cell); one product with the block diagonal of
    # W^T gives every bin at once.
    f_all = sparse.coo_matrix(
        (foot[keep], (row[keep], bin_idx[keep] * n_raster + cell[keep])),
        shape=(n_rows, n_bins * n_raster),
    ).tocsr()
    blocks = sparse.kron(sparse.identity(n_bins, format="csr"), weights.T.tocsr())
    return sparse.csr_matrix(f_all @ blocks)


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
    once. *table* is what :meth:`stilt.output.Footprints.table` reads: the
    stored non-zero cells of each receptor, all on the grid of *config*.

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
    _bin_edges(time_bins)  # raises for bins not closed on the left
    grid = config.grid
    if grid is None:
        raise ValueError("The footprint settings have no grid.")
    x_axis, y_axis = grid.axes
    weights = _target_weights(target, config, geometry_hash, x_axis, y_axis)
    nx = len(x_axis)

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
    rows = [r for r in receptors if r in found]
    empty = [r for r in receptors if r not in found]
    row_of = {r: i for i, r in enumerate(rows)}

    if table.num_rows:
        row_idx = np.array([row_of.get(r, -1) for r in ids], dtype=np.int64)[dict_idx]
        release_ns = _naive_utc_ns([parse_receptor_id(r)[0] for r in ids])
        hour = table.column("hour").to_numpy().astype(np.int64)
        flat = (
            table.column("y").to_numpy().astype(np.int64) * nx
            + table.column("x").to_numpy()
        )
        foot = table.column("foot").to_numpy().astype(np.float64)
        keep = row_idx >= 0  # receptors in the table that were not asked for
        data = _sum_onto(
            weights,
            nx * len(y_axis),
            time_bins,
            len(rows),
            row_idx[keep],
            (release_ns[dict_idx] + hour * 3_600_000_000_000)[keep],
            flat[keep],
            foot[keep],
        )
    else:
        data = sparse.csr_matrix((len(rows), len(time_bins) * len(target.index)))

    # A grid target's cells are (x, y) tuples; keep them as one label each.
    cells = pd.Index(list(target.index), tupleize_cols=False)
    bin_left = _naive_utc(time_bins.left)
    columns = pd.MultiIndex.from_product([bin_left, cells], names=["time", "cell"])
    return Jacobian(
        data, pd.Index(rows, name="receptor"), columns, empty, list(missing or [])
    )


def aggregate(
    foot: xr.DataArray, target: Geometry, time_bins: pd.IntervalIndex
) -> pd.DataFrame:
    """
    Sum a footprint onto the cells of a target, per time bin.

    The body of ``foot.stilt.aggregate``, which documents it. :func:`jacobian`
    does the same for many footprints at once.
    """
    _bin_edges(time_bins)  # raises for bins not closed on the left
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
        target, foot.stilt.config, foot.stilt.geometry_hash, x, y, foot.stilt.name
    )

    # The footprint's non-zero cells, as one receptor's row of a Jacobian.
    data = foot.transpose("time", y_dim, x_dim).to_numpy()
    t, iy, ix = np.nonzero(data)
    summed = _sum_onto(
        weights,
        len(x) * len(y),
        time_bins,
        1,
        np.zeros(len(t), dtype=np.int64),
        _naive_utc_ns(foot["time"].values)[t],
        iy.astype(np.int64) * len(x) + ix,
        data[t, iy, ix].astype(np.float64),
    )
    n_cells = len(target.index)
    by_bin = summed.toarray().reshape(len(columns), n_cells).T
    return pd.DataFrame(by_bin, index=target.index, columns=columns)
