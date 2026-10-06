"""
The averaging-kernel transform: weight each particle by a retrieval's kernel.

A kernel is given inline as ``levels`` and ``values``, or per receptor in a
table (:func:`averaging_kernel_table`, ``project.add_table``).
:func:`ak_weights` holds X-STILT's kernel weighting, public so your own
transforms can use it.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from stilt.receptors import Receptor

from stilt.transforms._common import release_coordinate


def ak_weights(
    particles: pd.DataFrame,
    levels: list[float],
    values: list[float],
    coordinate: str = "xhgt",
) -> np.ndarray:
    """
    Return the averaging kernel at each particle's release coordinate.

    The kernel is interpolated linearly to each particle's release value of
    ``coordinate`` and repeated on every row of that particle. Outside
    ``levels`` it is held at its end values.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``particle`` and ``coordinate``.
    levels : list of float
        Levels the kernel is given on, in the units of ``coordinate``.
    values : list of float
        Kernel value at each level.
    coordinate : str, default "xhgt"
        Particle column to interpolate on: ``xhgt`` (release height, m) or
        ``pres`` (hPa).

    Returns
    -------
    numpy.ndarray
        Kernel value for each row of ``particles``.
    """
    if coordinate not in particles.columns:
        raise ValueError(
            f"Particle DataFrame has no column {coordinate!r}. "
            "Assign release heights ('xhgt') before applying an averaging kernel, "
            "or pass coordinate='pres' for pressure-based interpolation."
        )
    lv = np.asarray(levels, dtype=float)
    va = np.asarray(values, dtype=float)
    order = np.argsort(lv)
    lv, va = lv[order], va[order]
    per_particle = release_coordinate(particles, coordinate)
    coords = per_particle.reindex(particles["particle"].to_numpy()).to_numpy(
        dtype=float
    )
    return np.interp(coords, lv, va, left=va[0], right=va[-1])


KERNEL_TABLE_COLUMNS = ("receptor", "level", "value")


def averaging_kernel_table(
    receptors: Iterable[Any],
    levels: Sequence[ArrayLike],
    values: Sequence[ArrayLike],
) -> pd.DataFrame:
    """
    Return a table of averaging kernels, one per receptor, for :class:`AveragingKernel`.

    The table is in long form, with ``receptor``, ``level``, and ``value``
    columns and one row per kernel level. Add it to the project with
    ``project.add_table("kernels", table)`` and name it as ``table:
    kernels`` in the ``averaging_kernel`` transform.

    Parameters
    ----------
    receptors : iterable
        :class:`~stilt.Receptor` objects or their ids.
    levels : sequence of array-like
        One array of levels per receptor, as for satellite soundings whose
        pressure grids differ, or a single array shared by all receptors.
    values : sequence of array-like
        One array of kernel values per receptor.

    Returns
    -------
    pandas.DataFrame
        The kernel table.
    """
    ids = [str(getattr(r, "id", r)) for r in receptors]
    value_arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    if len(value_arrays) != len(ids):
        raise ValueError(
            f"averaging_kernel_table: {len(ids)} receptors but "
            f"{len(value_arrays)} kernels."
        )
    try:
        shared = np.asarray(levels, dtype=float)
    except (TypeError, ValueError):  # ragged: one array per receptor
        shared = None
    if shared is not None and shared.ndim == 1:
        level_arrays = [shared] * len(ids)
    else:
        level_arrays = [np.asarray(lv, dtype=float).ravel() for lv in levels]
        if len(level_arrays) != len(ids):
            raise ValueError(
                f"averaging_kernel_table: {len(ids)} receptors but "
                f"{len(level_arrays)} level arrays."
            )
    frames = []
    for rid, lv, va in zip(ids, level_arrays, value_arrays, strict=True):
        if lv.size == 0 or lv.size != va.size:
            raise ValueError(
                f"averaging_kernel_table: receptor {rid} has {lv.size} levels "
                f"and {va.size} values."
            )
        frames.append(pd.DataFrame({"receptor": rid, "level": lv, "value": va}))
    return pd.concat(frames, ignore_index=True)


@functools.lru_cache(maxsize=8)
def _read_kernel_table(
    path: str, mtime: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Read a per-receptor averaging-kernel table from parquet or CSV.

    Cached by ``path`` and its modification time ``mtime``, so an edited
    table is read again.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
    elif suffix == ".csv":
        table = pd.read_csv(path)
    else:
        raise ValueError(
            f"averaging_kernel table {path!r} must be a .parquet or .csv file."
        )
    missing = [c for c in KERNEL_TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"averaging_kernel table {path!r} lacks columns {missing}; "
            f"expected {list(KERNEL_TABLE_COLUMNS)} (see averaging_kernel_table)."
        )
    kernels = {}
    for rid, group in table.groupby("receptor", sort=False):
        kernels[str(rid)] = (
            group["level"].to_numpy(dtype=float),
            group["value"].to_numpy(dtype=float),
        )
    return kernels


class AveragingKernel(BaseModel):
    """
    Transform that weights each particle's ``foot`` by an averaging kernel.

    Give the kernel as ``levels`` and ``values``, or name a ``table`` with
    one kernel per receptor (see :func:`averaging_kernel_table`): a project
    table added with ``project.add_table``, or a file, whose relative path
    is relative to the project root. A receptor missing from the table is
    an error.

    ``levels`` are release heights above ground in meters by default. Set
    ``coordinate: pres`` for a kernel on pressure levels in hPa. Fold any
    instrument-specific factor, such as TCCON's wet-air scaling, into the
    values. The kernel weight is added as an ``ak_weight`` column.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["averaging_kernel"] = "averaging_kernel"
    levels: list[float] | None = Field(
        default=None,
        description="Levels of the kernel, in meters above ground or hPa (see ``coordinate``).",
    )
    values: list[float] | None = Field(
        default=None, description="Normalized averaging-kernel value at each level."
    )
    table: str | None = Field(
        default=None,
        description=(
            "Table of kernels, one per receptor, with ``receptor``, ``level``, "
            "and ``value`` columns: the name of a project table "
            "(``project.add_table``), such as ``kernels``, or a .parquet or .csv "
            "file. A relative path is relative to the project root."
        ),
    )
    coordinate: str = Field(
        default="xhgt",
        description="Particle column the levels refer to: ``xhgt`` (release height) or ``pres``.",
    )

    @model_validator(mode="after")
    def _inline_or_table(self) -> Self:
        """Require either ``levels`` and ``values`` of equal length, or a ``table``."""
        inline = self.levels is not None or self.values is not None
        if self.table is not None:
            if inline:
                raise ValueError(
                    "averaging_kernel takes either levels/values or a table, not both."
                )
            if not self.table.strip():
                raise ValueError("averaging_kernel table must be a file name.")
            return self
        if not self.levels or not self.values:
            raise ValueError(
                "averaging_kernel requires non-empty levels and values, or a table."
            )
        if len(self.levels) != len(self.values):
            raise ValueError(
                f"averaging_kernel levels ({len(self.levels)}) and values "
                f"({len(self.values)}) must have the same length."
            )
        return self

    def kernel(
        self, receptor: Receptor | None = None, directory: str | Path | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Return ``(levels, values)`` of the kernel for *receptor*.

        A ``table`` is looked up by the receptor's id, and a relative
        ``table`` path starts from *directory*.
        """
        if self.table is None:
            assert self.levels is not None and self.values is not None
            return self.levels, self.values
        if receptor is None:
            raise ValueError(
                "averaging_kernel with a table needs the receptor to know which "
                "kernel to look up."
            )
        path = self.table
        if re.fullmatch(r"[A-Za-z0-9_-]+", path):  # a project table's name
            path = f"tables/{path}.parquet"
        if directory is not None and not Path(path).is_absolute():
            path = str(Path(directory) / path)
        kernels = _read_kernel_table(path, os.stat(path).st_mtime)
        rid = str(receptor.id)
        try:
            levels, values = kernels[rid]
        except KeyError:
            raise KeyError(
                f"averaging_kernel table {self.table!r} has no kernel for "
                f"receptor {rid!r}."
            ) from None
        return levels.tolist(), values.tolist()

    def apply(
        self,
        particles: pd.DataFrame,
        receptor: Receptor | None = None,
        directory: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` weighted by the averaging kernel."""
        levels, values = self.kernel(receptor, directory)
        weights = ak_weights(particles, levels, values, self.coordinate)
        out = particles.copy()
        out["ak_weight"] = weights
        out["foot"] = out["foot"] * weights
        return out
