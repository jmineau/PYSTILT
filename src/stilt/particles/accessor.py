"""
The ``.stilt`` accessor on a particle table: endpoints, grid exits, enhancement, export, plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from stilt.particles._cf import to_cf_trajectories
from stilt.sampling import sample_field
from stilt.spatial import _grid_cell_starts

if TYPE_CHECKING:
    import datetime as dt

    from stilt.spatial import Grid
    from stilt.visualization import ParticlesPlotAccessor

#: The edges of a grid, in the order of the columns of ``_edge_crossings``.
_EDGES = np.array(["n", "e", "s", "w"])

#: Rounding allowed when a point lies on a grid edge, in the grid's units.
_EDGE_TOLERANCE = 1e-9


def _grid_extent(grid: Grid) -> tuple[float, float, float, float]:
    """
    Return the outer edges of the grid's cells, ``(west, east, south, north)``, in the grid's units.

    Longitudes are not wrapped: a grid across the dateline has ``east`` above
    ``west`` (``170, 190`` for a grid from 170 to -170). The edges are those of
    the cells, which stop short of ``xmax`` and ``ymax`` when a whole number of
    cells does not fit.
    """
    if grid.is_longlat:
        span = grid.xmax - grid.xmin
        xmax = grid.xmin + (span if span > 0 else span + 360.0)
        x_starts = _grid_cell_starts(grid.xmin, xmax, grid.xres)
        y_starts = _grid_cell_starts(grid.ymin, grid.ymax, grid.yres)
        return (
            grid.xmin,
            float(x_starts[-1] + grid.xres),
            grid.ymin,
            float(y_starts[-1] + grid.yres),
        )
    x_centers, y_centers = grid.axes
    return (
        float(x_centers[0] - grid.xres / 2),
        float(x_centers[-1] + grid.xres / 2),
        float(y_centers[0] - grid.yres / 2),
        float(y_centers[-1] + grid.yres / 2),
    )


def _wrap_degrees(difference: np.ndarray) -> np.ndarray:
    """Return a difference of longitudes as the equivalent one in [-180, 180)."""
    return (difference + 180.0) % 360.0 - 180.0


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor:
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    Examples
    --------
    >>> particles = sim.particles  # doctest: +SKIP
    >>> particles.stilt.endpoints()  # doctest: +SKIP
    >>> particles.stilt.grid_exits(grid)  # doctest: +SKIP
    >>> particles.stilt.enhancement(flux).mean()  # doctest: +SKIP
    >>> particles.stilt.to_cf_trajectories()  # doctest: +SKIP
    >>> particles.stilt.plot.map()  # doctest: +SKIP
    """

    def __init__(self, particles: pd.DataFrame) -> None:
        self._particles = particles

    def endpoints(self) -> pd.DataFrame:
        """
        Return the last row of each particle, the one farthest in time from release.

        For a backward run this is where the air came from, which is where
        to sample a background concentration field. For a forward run it is
        where the air went. A particle that left the meteorology domain
        early ends where it left.

        Returns
        -------
        pandas.DataFrame
            One row per particle, with every column of the particle table.
            ``age`` is minutes since release and ``time`` the UTC time.
        """
        p = self._particles.reset_index(drop=True)
        if p.empty:
            return p
        reach = p["age"].abs()
        last = reach.groupby(p["particle"], sort=False).idxmax().to_numpy(dtype=int)
        return p.iloc[last]

    def grid_exits(self, grid: Grid) -> pd.DataFrame:
        """
        Return where each particle first left a footprint grid.

        For a backward run these are the points where the air entered the
        footprint's domain, which the export of a footprint to a regional
        model needs, and where to sample a background at the domain's edge.
        Unlike :meth:`endpoints`, which gives where a particle ended, this
        gives where it first crossed the edge of the grid, even if it came
        back later or stayed in the meteorology domain long after.

        A particle is outside when it is beyond the outer edge of the grid's
        cells, in the grid's own coordinates: a projected grid is a
        rectangle in its projection, not in longitude and latitude. A grid
        across the dateline or past 180 degrees is read modulo 360, and a
        grid that goes all the way around the earth has no west or east edge.
        Rows are read in order of ``abs(age)``, whatever their order in the
        table, so this works for a backward and a forward run.

        The point is on the edge: the exit is the straight line between the
        particle's last row inside the grid and its first row outside,
        and ``lon``, ``lat``, ``zagl``, ``age`` and ``time`` are interpolated
        along it to where it crosses. If the line crosses two edges at a
        corner, ``edge`` is the first it crosses. A particle whose first row is
        already outside has no row inside to interpolate from (the
        meteorology HYSPLIT builds bundled with PYSTILT writes no row at
        the release), so it is reported at that row, on the edge it is
        farthest beyond, in cells.

        Parameters
        ----------
        grid : stilt.Grid
            The footprint grid, such as ``project.config.footprint.grid``.

        Returns
        -------
        pandas.DataFrame
            One row per particle that left the grid, ordered by ``particle``;
            a particle that stayed inside is not in it. The columns are
            ``particle``, ``edge`` (``"n"``, ``"e"``, ``"s"`` or ``"w"``,
            the side of the grid it left by), ``lon`` and ``lat`` (degrees),
            ``zagl`` (meters), ``age`` (minutes since release) and, when the
            table has a ``time`` column, ``time`` (UTC). Longitudes are
            wrapped to -180 to 180 where the interpolation crosses 180.

        Examples
        --------
        One particle that leaves a grid by its east edge, between two rows:

        >>> import pandas as pd
        >>> import stilt
        >>> grid = stilt.Grid(
        ...     xmin=-112, xmax=-110, ymin=40, ymax=42, xres=0.5, yres=0.5
        ... )
        >>> particles = pd.DataFrame(
        ...     {
        ...         "particle": [1, 1, 1],
        ...         "age": [0.0, -60.0, -120.0],
        ...         "lon": [-111.0, -110.5, -109.5],
        ...         "lat": [41.0, 41.0, 41.0],
        ...         "zagl": [10.0, 20.0, 30.0],
        ...     }
        ... )
        >>> particles.stilt.grid_exits(grid)
           particle edge    lon   lat  zagl   age
        0         1    e -110.0  41.0  25.0 -90.0
        """
        p = self._particles
        p = p.iloc[np.lexsort((p["age"].abs().to_numpy(), p["particle"].to_numpy()))]
        p = p.reset_index(drop=True)
        particle = p["particle"].to_numpy()
        lon = p["lon"].to_numpy(dtype=float)
        lat = p["lat"].to_numpy(dtype=float)

        # Positions in the grid's units, measured from the grid's center.
        west, east, south, north = _grid_extent(grid)
        x, y = lon, lat
        if not grid.is_longlat:
            from pyproj import Transformer

            to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
            x, y = to_grid.transform(lon, lat)
        x = np.asarray(x, dtype=float) - (west + east) / 2
        y = np.asarray(y, dtype=float) - (south + north) / 2
        half_width, half_height = (east - west) / 2, (north - south) / 2
        if grid.is_longlat:
            x = _wrap_degrees(x)
            if east - west >= 360.0 - _EDGE_TOLERANCE:
                half_width = np.inf  # all the way around: no west or east edge

        # The first row of each particle that is beyond the edge of the grid.
        outside = (np.abs(x) > half_width + _EDGE_TOLERANCE) | (
            np.abs(y) > half_height + _EDGE_TOLERANCE
        )
        rows = np.flatnonzero(outside)
        rows = rows[np.unique(particle[rows], return_index=True)[1]]

        # The row before it, inside the grid; a particle that starts outside has none.
        before = np.maximum(rows - 1, 0)
        has_before = (rows > 0) & (particle[before] == particle[rows])
        before = np.where(has_before, before, rows)

        # Where the step from `before` to `rows` crosses each edge, as a fraction of it.
        step_x = x[rows] - x[before]
        if grid.is_longlat:
            step_x = _wrap_degrees(step_x)
        step_y = y[rows] - y[before]
        end_x, end_y = x[before] + step_x, y[before] + step_y
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = np.column_stack(
                [
                    np.where(
                        end_y > half_height, (half_height - y[before]) / step_y, np.inf
                    ),
                    np.where(
                        end_x > half_width, (half_width - x[before]) / step_x, np.inf
                    ),
                    np.where(
                        end_y < -half_height,
                        (-half_height - y[before]) / step_y,
                        np.inf,
                    ),
                    np.where(
                        end_x < -half_width, (-half_width - x[before]) / step_x, np.inf
                    ),
                ]
            )
            beyond = np.column_stack(  # how far past each edge, in cells
                [
                    (end_y - half_height) / grid.yres,
                    (end_x - half_width) / grid.xres,
                    (-half_height - end_y) / grid.yres,
                    (-half_width - end_x) / grid.xres,
                ]
            )
        first = np.argmin(crossings, axis=1)
        farthest = np.argmax(beyond, axis=1)
        edge = np.where(has_before, first, farthest)
        fraction = np.where(has_before, crossings[np.arange(rows.size), first], 1.0)
        fraction = np.clip(fraction, 0.0, 1.0)

        def at_edge(column: str) -> np.ndarray:
            values = p[column].to_numpy(dtype=float)
            return values[before] + fraction * (values[rows] - values[before])

        step_lon = _wrap_degrees(lon[rows] - lon[before])
        exit_lon = lon[before] + fraction * step_lon
        exit_lon = np.where(np.abs(exit_lon) > 180.0, _wrap_degrees(exit_lon), exit_lon)
        exits = pd.DataFrame(
            {
                "particle": particle[rows],
                "edge": _EDGES[edge],
                "lon": exit_lon,
                "lat": at_edge("lat"),
                "zagl": at_edge("zagl"),
                "age": at_edge("age"),
            }
        )
        if "time" in p.columns:
            time_before = p["time"].iloc[before].reset_index(drop=True)
            time_after = p["time"].iloc[rows].reset_index(drop=True)
            exits["time"] = time_before + (time_after - time_before) * fraction
        return exits

    def to_cf_trajectories(
        self, receptor_time: dt.datetime | None = None
    ) -> xr.Dataset:
        """
        Return the particles as a CF-1.10 trajectory dataset, one trajectory per particle.

        The dataset is what trajan, cf-xarray and other tools that read CF
        trajectories open. Write it to NetCDF with
        :meth:`xarray.Dataset.to_netcdf`. Each particle is one trajectory
        along the ``trajectory`` dimension, and its rows, from its release
        onward, run along ``obs``. A particle with fewer rows than the
        longest (one that left the meteorology domain early) is padded with
        missing values at the end. ``time`` is UTC, and ``age``, in minutes
        since release, is kept as a variable. ``obs`` counts output steps
        from the release, so in a backward run ``time`` decreases along
        ``obs``.

        This is CF's "incomplete multidimensional array" representation of
        trajectories (CF-1.10, appendix H.4.1).

        Parameters
        ----------
        receptor_time : datetime, optional
            Release time of the particles, in UTC. Needed only when the table
            has no ``time`` column. A table from ``sim.particles`` or
            :func:`~stilt.read_particles` has one; the table
            :func:`~stilt.run_trajectories` returns does not.

        Returns
        -------
        xarray.Dataset
            Dimensions ``(trajectory, obs)``. The coordinates are
            ``trajectory`` (the particle number), ``time``, ``lon``, ``lat``
            and ``zagl``. ``age`` and every other numeric column of the table
            are data variables, with units for the columns PYSTILT and
            HYSPLIT write.

        Raises
        ------
        ValueError
            If the table lacks one of ``particle``, ``age``, ``lon``, ``lat``
            or ``zagl``, if it has neither a ``time`` column nor
            *receptor_time*, if a particle has two rows at the same age, or if
            it holds more than one realization (select one:
            ``particles[particles.realization == k]``).

        Examples
        --------
        >>> import pandas as pd
        >>> particles = pd.DataFrame(
        ...     {
        ...         "particle": [1, 1, 1, 2, 2],
        ...         "age": [-1.0, -2.0, -3.0, -1.0, -2.0],
        ...         "lon": [-111.85, -111.86, -111.88, -111.84, -111.83],
        ...         "lat": [40.77, 40.78, 40.80, 40.76, 40.75],
        ...         "zagl": [5.0, 40.0, 120.0, 8.0, 60.0],
        ...     }
        ... )
        >>> ds = particles.stilt.to_cf_trajectories(
        ...     receptor_time=pd.Timestamp("2024-01-15 18:00")
        ... )
        >>> dict(ds.sizes)
        {'trajectory': 2, 'obs': 3}
        >>> ds.attrs["featureType"]
        'trajectory'
        >>> ds.age.sel(trajectory=2).values
        array([-1., -2., nan])
        >>> str(ds.time.sel(trajectory=1).values[-1])
        '2024-01-15T17:57:00.000000000'
        """
        return to_cf_trajectories(self._particles, receptor_time)

    def enhancement(self, flux: xr.DataArray) -> pd.Series:
        """
        Return each particle's enhancement, ``foot`` times flux summed along its trajectory.

        The mean over particles, after any weighting, is the modeled
        enhancement at the receptor. Unlike ``foot.stilt.enhancement``, the
        flux is taken at each particle position, with no gridding or
        smoothing.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid, in µmol m⁻² s⁻¹ for an
            enhancement in ppm, read the same way ``foot.stilt.enhancement``
            reads it: by its CF time bounds when it has them (a row before
            the first bound, after the last, or in a gap between bounds
            gets zero flux), or otherwise as steps that start at their
            labels (each particle row takes the last flux step that starts
            at or before the row's ``time``; a row before the first label
            gets zero flux, and a row after the last label takes the last
            step). A flux not labeled at the start of its steps and with no
            bounds must be relabeled first (see ``foot.stilt.enhancement``
            and :ref:`flux-step-labels`). Points outside the flux grid, and
            missing cells, count as zero flux.

        Returns
        -------
        pandas.Series
            Enhancement indexed by ``particle``. A particle that never crosses
            the flux field gets 0.

        Raises
        ------
        ValueError
            If the flux varies in time and the particles have no
            ``time`` column.
        """
        p = self._particles
        times = p["time"].to_numpy() if "time" in p.columns else None
        if "time" in flux.dims and times is None:
            raise ValueError(
                "flux varies in time but the particles have no 'time' column."
            )
        sampled = sample_field(
            flux,
            p["lon"].to_numpy(),
            p["lat"].to_numpy(),
            times=times,
            fill_value=0.0,
            step_starts=True,
        )
        contribution = p["foot"].to_numpy(dtype=float) * sampled
        ids = p["particle"].to_numpy()
        unique, inverse = np.unique(ids, return_inverse=True)
        sums = np.bincount(inverse, weights=contribution, minlength=unique.size)
        return pd.Series(
            sums, index=pd.Index(unique, name="particle"), name="enhancement"
        )

    @property
    def plot(self) -> ParticlesPlotAccessor:
        """Plotting methods, such as ``particles.stilt.plot.map()``."""
        from stilt.visualization import ParticlesPlotAccessor

        return ParticlesPlotAccessor(self._particles)
