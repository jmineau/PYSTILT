"""The particle table: reading and writing particle files, and the near-field plume dilution correction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stilt._atomic import write_parquet
from stilt.receptors import (
    Receptor,
    parse_receptor_id,
)
from stilt.sampling import sample_field
from stilt.transport import TransportConfig, get_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import xarray as xr

    from stilt.visualization import ParticlesPlotAccessor


# Below this horizontal spacing, release points cannot be told apart from a
# particle's first output row. In a test with HRRR at WBB, particles moved
# 200-600 m in the first minute, by an amount that varied with height. At
# 1000 m spacing the release height was recovered to about 15 m, and at 300 m
# it was off by about 190 m.


def _stored_params(stored: dict[str, Any], path: str | Path, model: str) -> Any:
    """Return the config stored in a particle file, read by *model*'s config class, dropping settings this version does not have."""
    config_class = get_model(model).config_class
    unknown = sorted(set(stored) - set(config_class.model_fields))
    if unknown:
        logger.debug(
            "%s: skipping stored params this version does not have: %s", path, unknown
        )
    return config_class.model_validate(
        {k: v for k, v in stored.items() if k not in unknown}
    )


#: Particle columns stored as int32 rather than float64.
_INT_COLUMNS = ("time", "indx")


class ParticleMetadata(NamedTuple):
    """What a particle file records about the HYSPLIT run that made it."""

    receptor: Receptor
    params: TransportConfig
    met_files: list[Path]


def prepare(raw: pd.DataFrame, receptor: Receptor) -> pd.DataFrame:
    """
    Return a transport model's particle output as the particle table PYSTILT keeps.

    Adds a ``datetime`` column from ``time``. The transport model has
    already done what its config asks, such as the near-field plume
    dilution correction (:func:`calc_plume_dilution`), and added each
    particle's release height ``xhgt`` for a column or multipoint receptor.

    Parameters
    ----------
    raw : pandas.DataFrame
        Particles as the transport model returns them.
    receptor : Receptor
        Receptor the particles were released from.

    Returns
    -------
    pandas.DataFrame
        One row per particle per output step.
    """
    p = raw.copy()
    p["datetime"] = receptor.time + pd.to_timedelta(p["time"].to_numpy(), unit="min")
    return p


def particles_metadata(path: str | Path) -> ParticleMetadata:
    """
    Return the receptor, transport settings, and met files a particle file records.

    Stored settings that this version of PYSTILT does not have are ignored.
    """
    meta = pq.read_schema(path).metadata or {}
    return ParticleMetadata(
        receptor=Receptor.from_dict(json.loads(meta[b"stilt:receptor"])),
        params=_stored_params(
            json.loads(meta[b"stilt:params"]),
            path,
            meta.get(b"stilt:model", b"hysplit").decode(),
        ),
        met_files=[Path(p) for p in json.loads(meta[b"stilt:met_files"])],
    )


def read_particles(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a particle file.

    It needs nothing but the file: the receptor time in its metadata gives
    the ``datetime`` column back. ``time`` and ``indx`` come back as
    float64, as HYSPLIT writes them. :func:`particles_metadata` reads the
    receptor and settings.

    Parameters
    ----------
    path : str or Path
        Particle file.
    columns : list of str, optional
        Columns to read. All columns by default.

    Returns
    -------
    pandas.DataFrame
        One row per particle per output step.

    Examples
    --------
    >>> particles = stilt.read_particles(sim.particles_path)
    >>> particles.stilt.endpoints()
    """
    pf = pq.ParquetFile(path)
    stored = pf.schema_arrow.names
    # The output directory adds a ``receptor`` column for scans of the whole
    # tree. One file is one receptor, so it is not read here.
    wanted = [c for c in (stored if columns is None else columns) if c != "receptor"]
    want_datetime = "datetime" in wanted or columns is None
    data = pf.read(columns=[c for c in wanted if c in stored]).to_pandas()
    for name in _INT_COLUMNS:
        if name in data.columns:
            data[name] = data[name].astype("float64")
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
    elif want_datetime and "time" in data.columns:
        receptor = particles_metadata(path).receptor
        data["datetime"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
            data["time"].to_numpy(), unit="min"
        )
    return data


def particles_from_table(table: pa.Table) -> pd.DataFrame:
    """
    Return a table of many receptors' particles as one DataFrame.

    The table is what :meth:`stilt.output.Particles.table` reads: a
    ``receptor`` column, the stored particle columns, and ``date``. The
    result has ``receptor`` first, ``time`` and ``indx`` as float64, and
    ``datetime`` rebuilt from each receptor's time; ``date`` is dropped.
    """
    data = table.unify_dictionaries().to_pandas()
    data = data.drop(columns=["date"], errors="ignore")
    if data.empty:
        return data
    data["receptor"] = data["receptor"].astype(str)
    for name in _INT_COLUMNS:
        if name in data.columns:
            data[name] = data[name].astype("float64")
    if "time" in data.columns:
        times = {r: parse_receptor_id(r)[0] for r in data["receptor"].unique()}
        receptor_time = pd.to_datetime(data["receptor"].map(times))
        data["datetime"] = receptor_time + pd.to_timedelta(
            data["time"].to_numpy(), unit="min"
        )
    return data


def write_particles(
    path: str | Path,
    particles: pd.DataFrame,
    receptor: Receptor,
    params: TransportConfig,
    met_files: list[Path],
    metadata: dict[bytes, bytes] | None = None,
) -> Path:
    """
    Write a particle table to a Parquet file that :func:`read_particles` reads alone.

    ``time`` and ``indx`` are stored as int32, and ``datetime`` is left out
    since it is the receptor time plus ``time``. A ``receptor`` column holds
    the receptor id, so a scan of many files can tell receptors apart. The
    receptor, transport settings, and met files go in the file's metadata.

    Parameters
    ----------
    path : str or Path
        File to write.
    particles : pandas.DataFrame
        The particle table.
    receptor : Receptor
        Receptor the particles were released from.
    params : TransportConfig
        The transport model's config for the run.
    met_files : list of Path
        Meteorology files the run used.
    metadata : dict, optional
        More file metadata, such as the settings hash.

    Returns
    -------
    Path
        The path written to.

    Raises
    ------
    ValueError
        If ``time`` or ``indx`` holds a value that is not a whole number.
    """
    data = particles.drop(columns=["datetime", "receptor"], errors="ignore")
    for name in _INT_COLUMNS:
        if name in data.columns:
            values = data[name].to_numpy()
            if not np.array_equal(values, np.round(values)):
                raise ValueError(f"Particle column {name!r} is not whole numbers.")
            data = data.assign(**{name: values.astype(np.int32)})
    table = pa.Table.from_pandas(data, preserve_index=False)
    table = table.add_column(
        0,
        pa.field("receptor", pa.dictionary(pa.int32(), pa.string())),
        pa.DictionaryArray.from_arrays(
            pa.array(np.zeros(table.num_rows, dtype=np.int32)),
            pa.array([str(receptor.id)]),
        ),
    )
    meta = {
        b"stilt:receptor": json.dumps(receptor.to_dict()).encode(),
        b"stilt:params": params.model_dump_json().encode(),
        b"stilt:met_files": json.dumps([str(p) for p in met_files]).encode(),
        **(metadata or {}),
    }
    table = table.replace_schema_metadata(meta)
    return write_parquet(table, Path(path))


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor:
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    Examples
    --------
    >>> particles = sim.particles
    >>> particles.stilt.endpoints()
    >>> particles.stilt.enhancement(flux).mean()
    >>> particles.stilt.plot.map()
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
            ``time`` is minutes since release and ``datetime`` the UTC time.
        """
        p = self._particles.reset_index(drop=True)
        if p.empty:
            return p
        reach = p["time"].abs()
        last = reach.groupby(p["indx"], sort=False).idxmax().to_numpy(dtype=int)
        return p.iloc[last]

    def enhancement(self, flux: xr.DataArray) -> pd.Series:
        """
        Return each particle's enhancement, ``foot`` times flux summed along its trajectory.

        The mean over particles, after any weighting, is the modelled
        enhancement at the receptor. Unlike ``foot.stilt.enhancement``, the
        flux is taken at each particle position, with no gridding or
        smoothing.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid, in µmol m⁻² s⁻¹ for an
            enhancement in ppm. A flux with a ``time`` dimension is taken at
            each particle's ``datetime``. Points outside it, and missing
            cells, count as zero flux.

        Returns
        -------
        pandas.Series
            Enhancement indexed by ``indx``. A particle that never crosses
            the flux field gets 0.

        Raises
        ------
        ValueError
            If the flux varies in time and the particles have no
            ``datetime`` column.
        """
        p = self._particles
        times = p["datetime"].to_numpy() if "datetime" in p.columns else None
        if "time" in flux.dims and times is None:
            raise ValueError(
                "flux varies in time but the particles have no 'datetime' column."
            )
        sampled = sample_field(
            flux,
            p["long"].to_numpy(),
            p["lati"].to_numpy(),
            times=times,
            fill_value=0.0,
        )
        contribution = p["foot"].to_numpy(dtype=float) * sampled
        indx = p["indx"].to_numpy()
        unique, inverse = np.unique(indx, return_inverse=True)
        sums = np.bincount(inverse, weights=contribution, minlength=unique.size)
        return pd.Series(sums, index=pd.Index(unique, name="indx"), name="enhancement")

    @property
    def plot(self) -> ParticlesPlotAccessor:
        """Plotting methods, such as ``particles.stilt.plot.map()``."""
        from stilt.visualization import ParticlesPlotAccessor

        return ParticlesPlotAccessor(self._particles)


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
