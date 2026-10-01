"""The particle table: reading and writing particle files, and the near-field plume dilution correction."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stilt._atomic import atomic_path
from stilt.config import STILTParams
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor, Receptor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stilt.visualization import ParticlesPlotAccessor


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
    """Return the params stored in a particle file, dropping settings this version does not have."""
    unknown = sorted(set(stored) - set(STILTParams.model_fields))
    if unknown:
        logger.debug(
            "%s: skipping stored params this version does not have: %s", path, unknown
        )
    return STILTParams.model_validate(
        {k: v for k, v in stored.items() if k not in unknown}
    )


#: Particle columns stored as int32 rather than float64.
_INT_COLUMNS = ("time", "indx")


class ParticleMetadata(NamedTuple):
    """What a particle file records about the HYSPLIT run that made it."""

    receptor: Receptor
    params: STILTParams
    met_files: list[Path]


def prepare(raw: pd.DataFrame, receptor: Receptor, params: STILTParams) -> pd.DataFrame:
    """
    Return HYSPLIT's particle output as the particle table PYSTILT keeps.

    Adds the release height ``xhgt`` for column and multipoint receptors,
    applies the near-field plume dilution correction when
    ``params.hnf_plume`` is set (:func:`calc_plume_dilution`), and adds a
    ``datetime`` column from ``time``.

    Parameters
    ----------
    raw : pandas.DataFrame
        Particle table read from ``PARTICLE_STILT.DAT``.
    receptor : Receptor
        Receptor the particles were released from.
    params : STILTParams
        Transport settings of the run.

    Returns
    -------
    pandas.DataFrame
        One row per particle per output step.
    """
    p = raw.copy()
    numpar = int(p["indx"].max())  # type: ignore[arg-type]

    if isinstance(receptor, ColumnReceptor):
        xhgt_step = (receptor.top - receptor.bottom) / numpar
        p["xhgt"] = (p["indx"] - 0.5) * xhgt_step + receptor.bottom
    elif isinstance(receptor, MultiPointReceptor):
        p["xhgt"] = _multipoint_release_heights(p, receptor)

    if params.hnf_plume:
        r_zagl = receptor.altitude if isinstance(receptor, PointReceptor) else None
        p = calc_plume_dilution(p, r_zagl, params.veght)

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
        params=_stored_params(json.loads(meta[b"stilt:params"]), path),
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
        stamps = data["receptor"].str.slice(0, 12)
        receptor_time = pd.to_datetime(stamps, format="%Y%m%d%H%M")
        data["datetime"] = receptor_time + pd.to_timedelta(
            data["time"].to_numpy(), unit="min"
        )
    return data


def write_particles(
    path: str | Path,
    particles: pd.DataFrame,
    receptor: Receptor,
    params: STILTParams,
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
    params : STILTParams
        Transport settings of the run.
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")
    return path


@pd.api.extensions.register_dataframe_accessor("stilt")
class ParticlesAccessor:
    """
    PYSTILT methods on a particle table, as ``particles.stilt``.

    Examples
    --------
    >>> particles = sim.particles
    >>> particles.stilt.endpoints()
    >>> particles.stilt.plot.map()
    """

    def __init__(self, particles: pd.DataFrame) -> None:
        self._particles = particles

    def endpoints(self) -> pd.DataFrame:
        """
        Return where each particle ends.

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
            backward run), and ``receptor_time``.

        Raises
        ------
        ValueError
            If the table has no ``datetime`` column.
        """
        cols = [
            "indx",
            "time",
            "lati",
            "long",
            "zagl",
            "endpoint_age_min",
            "receptor_time",
        ]
        p = self._particles
        if p.empty:
            return pd.DataFrame(columns=pd.Index(cols))
        if "datetime" not in p.columns:
            raise ValueError(
                "endpoints() needs the datetime column, which read_particles "
                "and sim.particles give."
            )
        ep = endpoint_rows(p)
        end_time = pd.to_datetime(ep["datetime"])
        age = ep["time"].to_numpy(dtype=float)
        receptor_time = end_time - pd.to_timedelta(age, unit="min")
        return pd.DataFrame(
            {
                "indx": ep["indx"].to_numpy(),
                "time": end_time.to_numpy(),
                "lati": ep["lati"].to_numpy(),
                "long": ep["long"].to_numpy(),
                "zagl": ep["zagl"].to_numpy(),
                "endpoint_age_min": age,
                "receptor_time": receptor_time.to_numpy(),
            }
        )

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
