"""Trajectories data model and parquet serialization helpers for STILT."""

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self

from stilt.config import STILTParams
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor, Receptor

if TYPE_CHECKING:
    from stilt.config import FootprintConfig
    from stilt.footprint import Footprint
    from stilt.visualization import TrajectoriesPlotAccessor


def _write_parquet_table(
    table: pa.Table,
    path: Path,
    *,
    use_dictionary: list[str] | bool,
) -> None:
    """Write parquet with compact defaults and conservative codec fallback."""
    last_error: Exception | None = None
    for compression in ("zstd", "snappy", None):
        try:
            pq.write_table(
                table,
                path,
                compression=cast(Any, compression),
                use_dictionary=cast(Any, use_dictionary),
            )
            return
        except (pa.ArrowNotImplementedError, ValueError) as exc:
            message = str(exc).lower()
            if "codec" not in message and "compression" not in message:
                raise
            last_error = exc
    if last_error is not None:
        raise last_error


# Below this horizontal spacing, release points cannot be told apart from the
# first in-flight row. Measured with HRRR at WBB: bulk advection moves
# particles 200-600 m in the first minute, by an amount that varies with
# height; at 1000 m spacing the release height is still recovered to ~15 m,
# at 300 m it is off by ~190 m.
_MIN_RELIABLE_SPACING_M = 1000.0


def endpoint_rows(particles: pd.DataFrame) -> pd.DataFrame:
    """
    The row at the far end of each particle's trajectory: its largest ``|time|``.

    One row per ``indx`` with every column of *particles*. The far end is
    where the air came from for a backward run and where it went for a
    forward run; a particle that left the domain early ends where it left.
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
    Return each row's release altitude for a multipoint (or slant) receptor.

    HYSPLIT does not record which starting location a particle came from, so
    it is recovered from the row nearest the release time:

    1. If the HYSPLIT build writes release-time (``t=0``) rows, nothing has
       moved yet and the nearest release point horizontally is exact.
    2. Otherwise the first row is already a timestep of transport later.
       Height drifts ~30x less than horizontal position over that step, so
       when the release altitudes are all distinct (always true of a slant
       column) match on height instead.
    3. Otherwise fall back to horizontal position, and warn when the release
       points are too close together for that to be trusted.
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


class Trajectories:
    """STILT particle trajectory ensemble."""

    def __init__(
        self,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
        data: pd.DataFrame,
        is_error: bool = False,
    ):
        """
        Particle trajectory ensemble with associated metadata.

        Parameters
        ----------
        receptor : Receptor
            Receptor metadata associated with this trajectory ensemble.
        data : pd.DataFrame
            Particle trajectory table.
        met_files : list[Path]
            Meteorology files used for this run.
        params : STILTParams
            Transport/model parameters used for this run.
        is_error : bool, default=False
            Whether this is a wind-error-perturbed run.
        """
        self.receptor = receptor
        self.params = params
        self.met_files = met_files
        self.data = data
        self.is_error = is_error
        self._plot: TrajectoriesPlotAccessor | None = None

    def __repr__(self) -> str:
        """Compact developer-facing trajectory representation."""
        return (
            f"Trajectories(rows={len(self.data)!r}, "
            f"is_error={self.is_error!r}, receptor={self.receptor.id!r})"
        )

    def endpoints(self) -> pd.DataFrame:
        """
        Per-particle trajectory endpoints (the far end of each particle's path).

        For each particle the row at the largest ``|time|`` from release is kept:
        where the air **came from** for a backward run (``is_backward``, the usual
        receptor case) or **went** for a forward run. This is the point to sample a
        boundary/background field at (e.g. ``lair.noaa.CarbonTracker.background``)
        for a backward run. Direction is handled implicitly -- ``max |time|`` is the
        most-negative offset for a backward run and the most-positive for a forward
        run.

        Every particle contributes one endpoint, whether it ran the full duration or
        left the domain early. An early exit is a real endpoint: the point where that
        air entered (backward) or left (forward) the domain, which is exactly where
        the background should be sampled.

        Returns
        -------
        pandas.DataFrame
            One row per particle with columns ``indx``, ``time`` (endpoint absolute
            time, UTC, ready for field sampling), ``lati``, ``long``, ``zagl``,
            ``endpoint_age_min`` (signed minutes from release), and ``run_time``
            (receptor release time). Ready to pass to
            ``CarbonTracker.sample()/.background()``.
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
        """Plotting namespace (e.g. ``traj.plot.map()``)."""
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
        Load a Trajectories instance from a self-contained parquet file.

        Metadata (receptor, params, met_files, is_error) is read from
        Arrow schema metadata embedded by ``to_parquet``.

        Parameters
        ----------
        path : str or Path
            Parquet file path.

        Returns
        -------
        Trajectories
        """
        # Get metadata
        pf = pq.ParquetFile(path)
        meta = pf.schema_arrow.metadata

        # Parse metadata
        receptor = Receptor.from_dict(json.loads(meta[b"stilt:receptor"]))
        params = STILTParams.model_validate(json.loads(meta[b"stilt:params"]))
        met_files = [Path(p) for p in json.loads(meta[b"stilt:met_files"])]
        is_error = json.loads(meta[b"stilt:is_error"])

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
            is_error=is_error,
        )

    @classmethod
    def from_particles(
        cls,
        particles: pd.DataFrame,
        receptor: Receptor,
        params: STILTParams,
        met_files: list[Path],
        is_error: bool = False,
    ) -> "Trajectories":
        """
        Build a Trajectories instance from raw HYSPLIT particle output.

        Assigns ``xhgt`` for column/multipoint receptors, applies
        ``hnf_plume`` dilution correction if configured, and converts
        the ``time`` column (minutes) to absolute ``datetime``.

        Parameters
        ----------
        particles : pd.DataFrame
            Raw particle table from ``read_particle_dat``.
        receptor : Receptor
            Receptor used for the run.
        params : TransportParams
            Transport/model parameters used for the run.
        met_files : list[Path]
            Meteorology files used for the run.
        is_error : bool, default=False
            Whether these are wind-error-perturbed particles.
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
            is_error=is_error,
        )

    def footprint(self, config: "FootprintConfig", name: str = "") -> "Footprint":
        """
        Calculate a footprint from these trajectories on a new grid.

        This is the regeneration path for a target grid that differs from any
        stored footprint: rather than regridding a stored raster, rebuild the
        footprint from the particles with full kernel fidelity.  Equivalent to
        :meth:`stilt.Footprint.calculate` with this run's receptor.

        Parameters
        ----------
        config : FootprintConfig
            Grid and smoothing parameters for the new footprint.
        name : str, optional
            Name for the footprint.
        """
        from stilt.footprint import Footprint

        return Footprint.calculate(self.data, self.receptor, config, name=name)

    def to_parquet(self, path: str | Path) -> Path:
        """
        Persist trajectory data and metadata to a self-contained parquet file.

        Receptor, params, met_files, and is_error are stored in Arrow
        schema metadata so ``from_parquet`` needs no sibling files.

        Parameters
        ----------
        path : str or Path
            Destination file path.

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
            b"stilt:params": (
                self.params.model_dump_json()
                if hasattr(self.params, "model_dump_json")
                else json.dumps(self.params)
            ).encode(),
            b"stilt:met_files": json.dumps([str(p) for p in self.met_files]).encode(),
            b"stilt:is_error": json.dumps(self.is_error).encode(),
        }
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, **meta})
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            _write_parquet_table(
                table,
                tmp_path,
                use_dictionary=["indx"] if "indx" in self.data.columns else True,
            )
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path


def calc_plume_dilution(
    particles: pd.DataFrame, r_zagl: float | None, veght: float
) -> pd.DataFrame:
    """
    Rescale footprint for near-field plume dilution.

    Requires ``varsiwant`` to include: ``dens``, ``samt``, ``sigw``,
    ``tlgr``, ``foot``, ``mlht``.

    Parameters
    ----------
    particles : DataFrame
        HYSPLIT particle output with columns for each required variable.
    r_zagl : float or None
        Receptor height above ground level in metres. ``None`` disables the
        near-field correction.
    veght : float
        STILT ``veght`` parameter (vegetation height / mixing-layer threshold).

    Returns
    -------
    DataFrame
        Particles DataFrame with the ``foot`` column rescaled by the
        plume-dilution factor.
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
    # cumsum must accumulate within each particle track in descending time order
    p["plume"] = start_h + (
        p.sort_values("time", ascending=False)
        .groupby("indx", sort=False)["sigma"]
        .cumsum()
        .reindex(p.index)
    )
    p["foot"] = np.where(
        p["plume"] < p["pbl_mixing"],
        0.02897 / (p["plume"] * p["dens"]) * p["samt"] * 60,
        p["foot"],
    )
    return p.drop(columns=["sigma", "pbl_mixing", "plume"])
