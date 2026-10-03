"""The ``.stilt`` accessor on a footprint."""

from __future__ import annotations

import datetime as dt
import json
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from stilt._atomic import atomic_path
from stilt.config import FootprintConfig, Grid
from stilt.identity import read_footprint_settings
from stilt.receptors import Receptor

from .aggregation import aggregate
from .io import _utc_index, _with_cf_metadata
from .targets import Geometry

if TYPE_CHECKING:
    from stilt.visualization import FootprintPlotAccessor


@xr.register_dataarray_accessor("stilt")
class FootprintAccessor:
    """
    PYSTILT methods on a footprint, as ``foot.stilt``.

    A footprint is an :class:`xarray.DataArray`, so sums, selections, and
    plots are plain xarray (``foot.sum("time")``,
    ``foot.sel(time=slice(a, b))``). The accessor adds what needs the
    receptor or the footprint settings, which ride in the array's
    attributes. Recent xarray keeps attributes through arithmetic. Older
    versions drop them (``foot * 2``), so with those, call these methods
    on the footprint as read, or use ``xr.set_options(keep_attrs=True)``.

    Examples
    --------
    >>> foot = sim.footprint
    >>> foot.stilt.receptor
    >>> foot.stilt.enhancement(flux).sum()
    >>> foot.stilt.aggregate(counties, time_bins=bins)
    """

    def __init__(self, foot: xr.DataArray) -> None:
        self._foot = foot

    def _attr(self, key: str) -> str:
        """Return a PYSTILT attribute, or raise when arithmetic has dropped it."""
        value = self._foot.attrs.get(key)
        if value is None:
            raise ValueError(
                f"This array has no {key!r} attribute. Older xarray versions "
                "drop attributes in arithmetic. Call this on the footprint as "
                "read, or use xr.set_options(keep_attrs=True)."
            )
        return value

    # xarray keeps one accessor per array, so each attribute is parsed once.

    @cached_property
    def receptor(self) -> Receptor:
        """The receptor the footprint belongs to."""
        return Receptor.from_json(self._attr("stilt_receptor"))

    @cached_property
    def _settings(self) -> tuple[FootprintConfig, str | None]:
        return read_footprint_settings(
            json.loads(self._attr("stilt_footprint")), "footprint"
        )

    @property
    def config(self) -> FootprintConfig:
        """The footprint settings: grid, smoothing, and particle transforms."""
        return self._settings[0]

    @property
    def geometry_hash(self) -> str | None:
        """Hash of the geometry the grid was derived for, or ``None`` when the footprint has no geometry."""
        return self._settings[1]

    @property
    def grid(self) -> Grid:
        """The grid the footprint is on."""
        grid = self.config.grid
        if grid is None:  # calculate() never makes a footprint without one
            raise ValueError("The footprint settings have no grid.")
        return grid

    @property
    def name(self) -> str:
        """The footprint's name, usually the variant name."""
        return str(self._foot.attrs.get("stilt_name", ""))

    @property
    def plot(self) -> FootprintPlotAccessor:
        """Plotting methods, such as ``foot.stilt.plot.map()``."""
        from stilt.visualization import FootprintPlotAccessor

        return FootprintPlotAccessor(self._foot)

    def to_netcdf(self, path: str | Path) -> Path:
        """
        Write the footprint to a CF-1.8 NetCDF file.

        :func:`stilt.read_footprint` reads it back with its receptor and
        settings.

        Parameters
        ----------
        path : str or Path
            File to write.

        Returns
        -------
        Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset({"foot": self._foot})
        # NetCDF stores naive times; footprint times are UTC.
        ds = ds.assign_coords(time=_utc_index(ds["time"].values).tz_convert(None))
        ds = _with_cf_metadata(ds, grid=self.grid)
        ds.attrs["time_created"] = (
            dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()
        )
        with atomic_path(path) as tmp:
            ds.to_netcdf(tmp, encoding={"foot": {"zlib": True, "complevel": 4}})
        return path

    def enhancement(self, flux: xr.DataArray) -> xr.DataArray:
        """
        Return the modelled enhancement at the receptor, footprint times flux summed over the grid.

        The flux is taken at each footprint cell centre from the nearest
        flux cell (:func:`stilt.sampling.sample_field`). Regrid a flux with much
        smaller cells than the footprint's before calling this.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid (``y``/``x`` for a
            projected footprint), in µmol m⁻² s⁻¹ for an enhancement in
            ppm. A flux with a ``time`` dimension is taken at each footprint
            time step.

        Returns
        -------
        xarray.DataArray
            Enhancement for each footprint time step. ``.sum()`` gives the
            total.
        """
        from stilt.sampling import sample_field

        foot = self._foot
        y_dim, x_dim = foot.dims[-2], foot.dims[-1]
        yy, xx = np.meshgrid(
            foot[y_dim].to_numpy(), foot[x_dim].to_numpy(), indexing="ij"
        )
        shape = yy.shape
        if "time" in flux.dims:
            layers = [
                sample_field(
                    flux,
                    xx.ravel(),
                    yy.ravel(),
                    times=np.full(xx.size, t),
                    fill_value=0.0,
                ).reshape(shape)
                for t in foot["time"].to_numpy()
            ]
            sampled = np.stack(layers)
        else:
            sampled = sample_field(
                flux, xx.ravel(), yy.ravel(), fill_value=0.0
            ).reshape(shape)[None]
        values = (foot.to_numpy() * sampled).sum(axis=(1, 2))
        return xr.DataArray(
            values,
            dims=["time"],
            coords={"time": foot["time"]},
            name="enhancement",
        )

    def aggregate(
        self,
        target: Geometry,
        time_bins: pd.IntervalIndex,
    ) -> pd.DataFrame:
        """
        Sum the footprint onto the cells of another grid or set of polygons, per time bin.

        A footprint value belongs to its whole cell, so it is summed when
        cells are combined. Each footprint cell is split among the target
        cells it overlaps, in proportion to the overlapping area. For a
        target whose cells are whole blocks of footprint cells, this is a
        plain block sum. Footprint values outside the target are dropped.
        Time steps are summed within each of ``time_bins``.

        Parameters
        ----------
        target : Grid, Mesh, or Zones
            Cells to sum onto:

            - :class:`~stilt.config.Grid`, every cell of a regular grid, in
              ``Grid.index`` order.
            - :class:`~stilt.Mesh`, any polygons (a shapefile, H3 hexagons,
              nested grids). Rows are indexed by cell id.
            - :class:`~stilt.Zones`, groups of the cells of a grid or mesh.
              Use it to keep only some cells of a grid.

            A target in another coordinate system is projected onto the
            footprint grid.
        time_bins : pandas.IntervalIndex
            Time intervals to sum over, such as the time steps of a flux
            inventory. They must be closed on the left
            (``closed="left"``): each bin holds the footprint hours that
            start in it.

        Returns
        -------
        pandas.DataFrame
            One row per target cell and one column per time bin, labelled
            by the bin's left edge. Rows are indexed by ``(x, y)`` for grid
            targets and by cell id for meshes and zones. Cells and bins the
            footprint does not reach are 0.

        Raises
        ------
        ValueError
            If ``time_bins`` is not closed on the left.
        """
        return aggregate(self._foot, target, time_bins)
