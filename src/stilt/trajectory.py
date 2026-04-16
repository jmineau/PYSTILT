"""Trajectories data model and parquet serialization helpers for STILT."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self

from stilt.config import STILTParams
from stilt.receptor import Receptor

if TYPE_CHECKING:
    from stilt.visualization import TrajectoriesPlotAccessor


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

    @property
    def plot(self) -> "TrajectoriesPlotAccessor":
        """Plotting namespace (e.g. ``traj.plot.map()``)."""
        if self._plot is None:
            from stilt.visualization import TrajectoriesPlotAccessor

            self._plot = TrajectoriesPlotAccessor(self)
        return self._plot

    @classmethod
    def from_parquet(cls, path: str | Path) -> Self:
        """Load a Trajectories instance from a self-contained parquet file.

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
        params = json.loads(meta[b"stilt:params"])
        met_files = [Path(p) for p in json.loads(meta[b"stilt:met_files"])]
        is_error = json.loads(meta[b"stilt:is_error"])

        # Read data
        data = pd.read_parquet(path)
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
        """Build a Trajectories instance from raw HYSPLIT particle output.

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

        if receptor.kind == "column":
            xhgt_step = (receptor.top - receptor.bottom) / numpar
            p["xhgt"] = (p["indx"] - 0.5) * xhgt_step + receptor.bottom
        elif receptor.kind == "multipoint":
            release_rows = (
                p.loc[p["time"] == p["time"].max(), ["indx", "long", "lati"]]
                .drop_duplicates(subset=["indx"])
                .sort_values("indx")
            )
            release_points = np.column_stack(
                (
                    np.asarray(receptor.longitudes, dtype=float),
                    np.asarray(receptor.latitudes, dtype=float),
                )
            )
            if len(release_rows) == numpar:
                particle_points = release_rows[["long", "lati"]].to_numpy(dtype=float)
                distances = np.sum(
                    (particle_points[:, None, :] - release_points[None, :, :]) ** 2,
                    axis=2,
                )
                nearest_release = np.argmin(distances, axis=1)
                mapping = {
                    int(indx): float(receptor.altitudes[point_idx])
                    for indx, point_idx in zip(
                        release_rows["indx"], nearest_release, strict=False
                    )
                }
            else:
                hgts = receptor.altitudes
                n_locs = len(hgts)
                counts = [numpar // n_locs] * n_locs
                for i in range(numpar % n_locs):
                    counts[i] += 1
                mapping: dict[int, float] = {}
                idx = 1
                for height, count in zip(hgts, counts, strict=False):
                    for _ in range(count):
                        mapping[idx] = height
                        idx += 1
            p["xhgt"] = p["indx"].map(mapping.get)

        if params.hnf_plume:
            r_zagl = receptor.altitude if receptor.kind == "point" else None
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

    def to_parquet(self, path: str | Path) -> Path:
        """Persist trajectory data and metadata to a self-contained parquet file.

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
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path


def calc_plume_dilution(
    particles: pd.DataFrame, r_zagl: float | None, veght: float
) -> pd.DataFrame:
    """Rescale footprint for near-field plume dilution.

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
    p = p.sort_values("time", ascending=False)

    start_h = p["xhgt"] if "xhgt" in p.columns else r_zagl
    if start_h is None:
        raise ValueError("r_zagl must be provided if 'xhgt' is not in particles.")
    p["plume"] = start_h + p.groupby("indx")["sigma"].cumsum()
    p["foot"] = np.where(
        p["plume"] < p["pbl_mixing"],
        0.02897 / (p["plume"] * p["dens"]) * p["samt"] * 60,
        p["foot"],
    )
    return p.drop(columns=["sigma", "pbl_mixing", "plume"])
