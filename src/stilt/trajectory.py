"""Particle trajectories from a HYSPLIT run, and the near-field plume dilution correction."""

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stilt._atomic import atomic_path
from stilt.config import STILTParams
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor, Receptor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.config import FootprintConfig
    from stilt.footprint import Footprint
    from stilt.transforms import TransformContext
    from stilt.visualization import TrajectoriesPlotAccessor


# Below this horizontal spacing, release points cannot be told apart from a
# particle's first output row. In a test with HRRR at WBB, particles moved
# 200-600 m in the first minute, by an amount that varied with height. At
# 1000 m spacing the release height was recovered to about 15 m, and at 300 m
# it was off by about 190 m.
_MIN_RELIABLE_SPACING_M = 1000.0


def endpoint_rows(particles: pd.DataFrame) -> pd.DataFrame:
    """
    Return the last row of each particle's trajectory, the one with the largest ``|time|``.

    For a backward run this is where the air came from, and for a forward
    run where it went. A particle that left the meteorology domain early
    ends where it left.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``indx`` and ``time`` columns.

    Returns
    -------
    pandas.DataFrame
        One row per particle, with every column of *particles*.
    """
    p = particles.reset_index(drop=True)
    if p.empty:
        return p
    reach = p["time"].abs()
    last = reach.groupby(p["indx"], sort=False).idxmax().to_numpy(dtype=int)
    return p.iloc[last]


def _multipoint_release_heights(
    p: pd.DataFrame, receptor: MultiPointReceptor
) -> pd.Series:
    """
    Return each row's release altitude for a multipoint receptor.

    HYSPLIT does not record which starting location a particle came from.
    It is recovered from each particle's row nearest the release time:

    1. If the HYSPLIT build writes release-time (``t = 0``) rows, match the
       nearest release point horizontally. Nothing has moved yet, so this
       is exact.
    2. Otherwise the first row is one time step after release. Height
       drifts about 30 times less than horizontal position over that step,
       so when all release heights differ (as in a slanted column), match
       on height.
    3. Otherwise match on horizontal position, and warn when the release
       points are too close together for that to be reliable.
    """
    first = (
        p.assign(_age=p["time"].abs())
        .sort_values("_age", kind="stable")
        .drop_duplicates(subset="indx")
    )
    lons = np.asarray(receptor.longitudes, dtype=float)
    lats = np.asarray(receptor.latitudes, dtype=float)
    alts = np.asarray(receptor.altitudes, dtype=float)
    has_t0 = bool((first["_age"] == 0).all())

    # Height of each particle in the receptor's own vertical reference.
    height = None
    if "zagl" in first.columns:
        if receptor.altitude_ref == "agl":
            height = first["zagl"].to_numpy(dtype=float)
        elif "zsfc" in first.columns:
            height = (first["zagl"] + first["zsfc"]).to_numpy(dtype=float)

    if not has_t0 and height is not None and len(np.unique(alts)) == len(alts):
        nearest = np.argmin(np.abs(height[:, None] - alts[None, :]), axis=1)
    else:
        xy = first[["long", "lati"]].to_numpy(dtype=float)
        pts = np.column_stack((lons, lats))
        nearest = np.argmin(
            np.sum((xy[:, None, :] - pts[None, :, :]) ** 2, axis=2), axis=1
        )
        if not has_t0 and len(alts) > 1:
            x = lons * np.cos(np.radians(lats.mean())) * 111_320.0
            y = lats * 111_320.0
            gaps = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
            spacing = float(gaps[np.triu_indices(len(alts), k=1)].min())
            if spacing < _MIN_RELIABLE_SPACING_M:
                warnings.warn(
                    f"MultiPointReceptor release points are as close as "
                    f"{spacing:.0f} m and cannot be separated by altitude, and this "
                    "HYSPLIT build writes no t=0 row, so particles cannot be "
                    "reliably matched to their release points; 'xhgt' may be wrong. "
                    "Use a HYSPLIT build that writes release-time rows "
                    "(STILTParams.exe_dir) or space the points more than "
                    f"{_MIN_RELIABLE_SPACING_M:.0f} m apart.",
                    stacklevel=3,
                )

    mapping = dict(zip(first["indx"].to_numpy(), alts[nearest], strict=True))
    return cast(pd.Series, p["indx"]).map(mapping.get)


def _stored_params(stored: dict[str, Any], path: str | Path) -> STILTParams:
    """Return the params stored in a trajectory file, dropping settings this version does not have."""
    unknown = sorted(set(stored) - set(STILTParams.model_fields))
    if unknown:
        logger.debug(
            "%s: skipping stored params this version does not have: %s", path, unknown
        )
    return STILTParams.model_validate(
        {k: v for k, v in stored.items() if k not in unknown}
    )


class Trajectories:
    """
    Particle trajectories from one HYSPLIT run.

    ``data`` has one row per particle per output step. The columns are the
    variables in ``varsiwant`` (``indx``, ``time`` in minutes since release,
    ``long``, ``lati``, ``zagl``, ``foot``, ...), plus ``datetime`` (UTC),
    ``xhgt`` (release height, for column and multipoint receptors), and
    ``foot_no_hnf_dilution`` when ``hnf_plume`` is set.

    Trajectories normally come from a simulation (``sim.trajectories``) or
    a file (:meth:`from_parquet`).

    Parameters
    ----------
    receptor : Receptor
        Receptor the particles were released from.
    params : STILTParams
        Transport settings of the run.
    met_files : list of Path
        Meteorology files the run used.
    data : pandas.DataFrame
        Particle table.
    """

    def __init__(
        self,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
        data: pd.DataFrame,
    ):
        self.receptor = receptor
        self.params = params
        self.met_files = met_files
        self.data = data
        self._plot: TrajectoriesPlotAccessor | None = None

    def __repr__(self) -> str:
        return f"Trajectories(rows={len(self.data)!r}, receptor={self.receptor.id!r})"

    def endpoints(self) -> pd.DataFrame:
        """
        Return where each particle's trajectory ends.

        For a backward run this is where the air came from, which is where
        to sample a background concentration field. Each particle has one
        endpoint, including particles that left the meteorology domain
        early. Such a particle ends where it left the domain.

        Returns
        -------
        pandas.DataFrame
            One row per particle, with columns ``indx``, ``time`` (UTC time
            at the endpoint), ``lati``, ``long``, ``zagl``,
            ``endpoint_age_min`` (minutes since release, negative for a
            backward run), and ``run_time`` (receptor time).
        """
        cols = ["indx", "time", "lati", "long", "zagl", "endpoint_age_min", "run_time"]
        if self.data.empty:
            return pd.DataFrame(columns=pd.Index(cols))

        ep = endpoint_rows(self.data)

        if "datetime" in ep.columns:
            end_time = pd.to_datetime(ep["datetime"]).to_numpy()
        else:
            end_time = (
                pd.Timestamp(self.receptor.time)
                + pd.to_timedelta(ep["time"].to_numpy(dtype=float), unit="min")
            ).to_numpy()

        return pd.DataFrame(
            {
                "indx": ep["indx"].to_numpy(),
                "time": end_time,
                "lati": ep["lati"].to_numpy(),
                "long": ep["long"].to_numpy(),
                "zagl": ep["zagl"].to_numpy(),
                "endpoint_age_min": ep["time"].to_numpy(),
                "run_time": pd.Timestamp(self.receptor.time),
            }
        )

    @property
    def plot(self) -> "TrajectoriesPlotAccessor":
        """Plotting methods, such as ``traj.plot.map()``."""
        if self._plot is None:
            from stilt.visualization import TrajectoriesPlotAccessor

            self._plot = TrajectoriesPlotAccessor(self)
        return self._plot

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        columns: list[str] | None = None,
    ) -> Self:
        """
        Read trajectories from a Parquet file written by :meth:`to_parquet`.

        The receptor, params, and met files are read from the file's
        metadata. Stored settings that this version of PYSTILT does not
        have are ignored.

        Parameters
        ----------
        path : str or Path
            Trajectory file.
        columns : list of str, optional
            Columns to read. All columns by default.

        Returns
        -------
        Trajectories
        """
        # Get metadata
        pf = pq.ParquetFile(path)
        meta = pf.schema_arrow.metadata

        # Parse metadata
        receptor = Receptor.from_dict(json.loads(meta[b"stilt:receptor"]))
        params = _stored_params(json.loads(meta[b"stilt:params"]), path)
        met_files = [Path(p) for p in json.loads(meta[b"stilt:met_files"])]

        # Read data. `datetime` is written naive UTC by ``from_particles``; keep
        # it naive on read so the receptor/trajectory/footprint time axes align.
        data = pf.read(columns=columns).to_pandas()
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])

        return cls(
            receptor=receptor,
            params=params,
            met_files=met_files,
            data=data,
        )

    @classmethod
    def from_particles(
        cls,
        particles: pd.DataFrame,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
    ) -> "Trajectories":
        """
        Build trajectories from HYSPLIT's particle output.

        Adds the release height ``xhgt`` for column and multipoint receptors,
        applies the near-field plume dilution correction when
        ``params.hnf_plume`` is set (:func:`calc_plume_dilution`), and adds a
        ``datetime`` column from ``time``.

        Parameters
        ----------
        particles : pandas.DataFrame
            Particle table read from ``PARTICLE_STILT.DAT``.
        receptor : Receptor
            Receptor the particles were released from.
        params : STILTParams
            Transport settings of the run.
        met_files : list of Path
            Meteorology files the run used.

        Returns
        -------
        Trajectories
        """
        p = particles.copy()
        numpar = int(p["indx"].max())  # type: ignore[arg-type]

        if isinstance(receptor, ColumnReceptor):
            xhgt_step = (receptor.top - receptor.bottom) / numpar
            p["xhgt"] = (p["indx"] - 0.5) * xhgt_step + receptor.bottom
        elif isinstance(receptor, MultiPointReceptor):
            p["xhgt"] = _multipoint_release_heights(p, receptor)

        if params.hnf_plume:
            r_zagl = receptor.altitude if isinstance(receptor, PointReceptor) else None
            p = calc_plume_dilution(p, r_zagl, params.veght)

        p["datetime"] = receptor.time + pd.to_timedelta(
            p["time"].to_numpy(), unit="min"
        )

        return cls(
            receptor=receptor,
            data=p,
            met_files=met_files,
            params=params,
        )

    def footprint(
        self,
        config: "FootprintConfig",
        name: str = "",
        context: "TransformContext | None" = None,
    ) -> "Footprint":
        """
        Calculate a footprint from these particles.

        This is how every footprint is made. ``config.transforms`` are
        applied to the particles first, and the footprint records them. Use
        it for a footprint on another grid, with other smoothing, or with
        other transforms, instead of regridding a saved footprint. Same as
        :meth:`stilt.Footprint.calculate` with this run's receptor.

        Parameters
        ----------
        config : FootprintConfig
            Grid, smoothing, and particle transforms.
        name : str, optional
            Name of the footprint.
        context : TransformContext, optional
            Passed to every transform. :meth:`stilt.Simulation.transform_context`
            gives one with the project store, which a transform needs to find
            a file named relative to the project, such as an averaging-kernel
            table. Defaults to one with the receptor and ``name`` only.

        Returns
        -------
        Footprint

        Raises
        ------
        EmptyFootprint
            If no particle is over the grid.
        """
        from stilt.footprint import Footprint

        return Footprint.calculate(
            self.data, self.receptor, config, name=name, context=context
        )

    def to_parquet(self, path: str | Path) -> Path:
        """
        Write the trajectories to a Parquet file.

        The receptor, params, and met files are stored in the file's
        metadata, so :meth:`from_parquet` needs nothing else.

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

        table = pa.Table.from_pandas(self.data, preserve_index=False)
        meta = {
            b"stilt:receptor": json.dumps(self.receptor.to_dict()).encode(),
            b"stilt:params": self.params.model_dump_json().encode(),
            b"stilt:met_files": json.dumps([str(p) for p in self.met_files]).encode(),
        }
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, **meta})
        with atomic_path(path) as tmp:
            pq.write_table(table, tmp, compression="zstd")
        return path


def calc_plume_dilution(
    particles: pd.DataFrame, r_zagl: float | None, veght: float
) -> pd.DataFrame:
    """
    Correct ``foot`` for plume dilution in the hyper-near field.

    ``foot`` assumes surface fluxes are mixed through the lowest ``veght``
    fraction of the mixed layer. Close to the receptor, the plume from the
    release point is thinner than that. Following STILT-R, the plume depth
    grows from the release height with the turbulence each particle meets.
    While it is below ``veght`` times the mixed-layer height, ``foot`` is
    recalculated with the plume depth in its place.

    Needs the columns ``dens``, ``samt``, ``sigw``, ``tlgr``, ``foot``,
    and ``mlht``, so ``varsiwant`` must include them.

    Parameters
    ----------
    particles : pandas.DataFrame
        HYSPLIT particle table.
    r_zagl : float or None
        Release height above ground in metres. Used only when *particles*
        has no ``xhgt`` column.
    veght : float
        Fraction of the mixed-layer height that surface fluxes are mixed
        through (STILT's ``veght``).

    Returns
    -------
    pandas.DataFrame
        A copy of *particles* with ``foot`` corrected and the original
        values in ``foot_no_hnf_dilution``.

    Raises
    ------
    ValueError
        If a required column is missing, or neither *r_zagl* nor ``xhgt``
        gives the release height.
    """
    required = {"dens", "samt", "sigw", "tlgr", "foot", "mlht"}
    missing = required - set(particles.columns)
    if missing:
        raise ValueError(
            f"hnf_plume requires varsiwant to include: {', '.join(sorted(missing))}"
        )

    p = particles.copy()
    p["foot_no_hnf_dilution"] = p["foot"]

    abs_time_s = np.abs(p["time"] * 60)
    p["sigma"] = (
        p["samt"]
        * np.sqrt(2)
        * p["sigw"]
        * np.sqrt(
            p["tlgr"] * abs_time_s
            + p["tlgr"] ** 2 * np.exp(-abs_time_s / p["tlgr"])
            - 1
        )
    )
    p["pbl_mixing"] = veght * p["mlht"]

    start_h = p["xhgt"] if "xhgt" in p.columns else r_zagl
    if start_h is None:
        raise ValueError("r_zagl must be provided if 'xhgt' is not in particles.")
    # The plume grows outward from the release point, so the cumsum must walk
    # each particle track in order of elapsed time since release. That is
    # |time| ascending, which covers forward runs as well as backward ones.
    p["elapsed"] = abs_time_s
    p["plume"] = start_h + (
        p.sort_values("elapsed")
        .groupby("indx", sort=False)["sigma"]
        .cumsum()
        .reindex(p.index)
    )
    p["foot"] = np.where(
        p["plume"] < p["pbl_mixing"],
        0.02897 / (p["plume"] * p["dens"]) * p["samt"] * 60,
        p["foot"],
    )
    return p.drop(columns=["sigma", "pbl_mixing", "plume", "elapsed"])
