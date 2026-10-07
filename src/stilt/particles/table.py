"""
The particle table: its columns, particle files, release heights, and the near-field plume correction.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stilt._paths import location, readable, write_parquet
from stilt.receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    parse_receptor_id,
)

if TYPE_CHECKING:
    from upath import UPath

    from stilt._paths import Location


#: The columns every particle table has, as a particle file stores them:
#: what a transport model's run must return. ``particle`` is the particle number
#: (1 to ``numpar``), ``time`` the minutes since release (negative for a
#: backward run), ``lon`` and ``lat`` the position in degrees, and ``zagl``
#: the height above ground in metres. The reference page on the particle
#: table says what the other columns are.
PARTICLE_SCHEMA = pa.schema(
    [
        ("particle", pa.int32()),
        ("time", pa.int32()),
        ("lon", pa.float64()),
        ("lat", pa.float64()),
        ("zagl", pa.float64()),
    ]
)

#: Particle columns stored as int32 rather than float64.
_INT_COLUMNS = ("time", "particle")

#: The column a footprint is made from: each particle's sensitivity to
#: surface fluxes over one output step.
FOOTPRINT_COLUMNS: tuple[str, ...] = ("foot",)

#: Particle columns the near-field plume correction reads (``varsiwant`` must include them).
HNF_PLUME_COLUMNS: tuple[str, ...] = ("dens", "samt", "sigw", "tlgr", "foot", "mlht")

#: Molar mass of dry air, kg/mol.
_MOLAR_MASS_DRY_AIR = 0.02897


class ParticleMetadata(NamedTuple):
    """
    What a particle file records about the run that made it.

    Attributes
    ----------
    receptor : Receptor
        Receptor the particles were released from.
    settings : dict
        The run's settings, as its folder's ``_settings.yaml`` records them:
        the transport model's settings, the met's, the model build, and
        whether it is an ensemble. :func:`stilt.identity.transport_from_settings`
        rebuilds the transport model's config from it.
    met_files : list of Path
        Meteorology files the run read.
    realization : int or None
        Which realization of an ensemble the particles are, which ran with
        ``seed + realization``; ``None`` for a single run.
    reach_minutes : float or None
        How far the particles got from the release, in minutes: the largest
        ``|time|``. A complete run reaches ``|n_hours| * 60``. ``None`` for
        a file written before particle files recorded it.
    """

    receptor: Receptor
    settings: dict[str, Any]
    met_files: list[Path]
    realization: int | None = None
    reach_minutes: float | None = None


def check_particles(particles: pd.DataFrame, need: Iterable[str] = ()) -> None:
    """
    Check that a particle table has the columns of :data:`PARTICLE_SCHEMA`, and *need*.

    Parameters
    ----------
    particles : pandas.DataFrame
        A particle table, such as a transport model returns.
    need : iterable of str, optional
        Other columns the next step reads, such as ``("foot",)`` for a
        footprint.

    Raises
    ------
    ValueError
        Naming the missing columns.
    """
    wanted = dict.fromkeys([*PARTICLE_SCHEMA.names, *need])
    missing = [name for name in wanted if name not in particles.columns]
    if missing:
        raise ValueError(
            f"The particle table has no {', '.join(repr(m) for m in missing)} "
            f"column{'s' if len(missing) > 1 else ''}. Every particle table has "
            f"{', '.join(PARTICLE_SCHEMA.names)} (stilt.particles.PARTICLE_SCHEMA)."
        )


def particles_metadata(path: str | Path | UPath) -> ParticleMetadata:
    """
    Return the receptor, run settings, and met files a particle file records.

    Raises
    ------
    ValueError
        If the file records no run settings, as files written before
        PYSTILT recorded them do.

    Examples
    --------
    >>> meta = stilt.particles.particles_metadata(sim.particles_path)
    >>> meta.settings["model"]
    {'name': 'hysplit', 'version': 'v5.1.0'}
    """
    with readable(path) as source:
        meta = pq.read_schema(source).metadata or {}
    if b"stilt:settings" not in meta:
        raise ValueError(
            f"{path} records no run settings. It was written before particle "
            "files recorded them; rewrite its output directory (see #132)."
        )
    realization = meta.get(b"stilt:realization")
    reach = meta.get(b"stilt:reach_minutes")
    return ParticleMetadata(
        receptor=Receptor.from_json(meta[b"stilt:receptor"]),
        settings=json.loads(meta[b"stilt:settings"]),
        met_files=[Path(p) for p in json.loads(meta[b"stilt:met_files"])],
        realization=None if realization is None else int(realization),
        reach_minutes=None if reach is None else float(reach),
    )


def read_particles(
    path: str | Path | UPath, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Read a particle file.

    It needs nothing but the file: the receptor time in its metadata gives
    the ``datetime`` column back. ``time`` and ``particle`` come back as
    float64, as HYSPLIT writes them. :func:`particles_metadata` reads the
    receptor and settings.

    Parameters
    ----------
    path : str or Path
        Particle file, or its URL on an object store.
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
    with readable(path) as source:
        pf = pq.ParquetFile(source)
        stored = pf.schema_arrow.names
        # The output directory adds a ``receptor`` column for scans of the
        # whole tree. One file is one receptor, so it is not read here.
        wanted = [
            c for c in (stored if columns is None else columns) if c != "receptor"
        ]
        want_datetime = "datetime" in wanted or columns is None
        data = pf.read(columns=[c for c in wanted if c in stored]).to_pandas()
    for name in _INT_COLUMNS:
        if name in data.columns:
            data[name] = data[name].astype("float64")
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
    elif want_datetime and "time" in data.columns:
        meta = pf.schema_arrow.metadata or {}
        receptor = Receptor.from_json(meta[b"stilt:receptor"])
        data["datetime"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
            data["time"].to_numpy(), unit="min"
        )
    return data


def particles_from_table(table: pa.Table) -> pd.DataFrame:
    """
    Return a table of many receptors' particles as one DataFrame.

    The table is what :meth:`stilt.output.Output.table` reads: a
    ``receptor`` column, the stored particle columns, and ``date``. The
    result has ``receptor`` first, ``time`` and ``particle`` as float64, and
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
    path: str | Path | UPath,
    particles: pd.DataFrame,
    receptor: Receptor,
    settings: Mapping[str, Any],
    met_files: list[Path],
    metadata: dict[bytes, bytes] | None = None,
) -> Location:
    """
    Write a particle table to a Parquet file that :func:`read_particles` reads alone.

    ``time`` and ``particle`` are stored as int32, and ``datetime`` is left out
    since it is the receptor time plus ``time``. A ``receptor`` column holds
    the receptor id, so a scan of many files can tell receptors apart. The
    receptor, the run's settings, the met files, and how far the particles
    got (``stilt:reach_minutes``) go in the file's metadata, so the file
    reads alone.

    Parameters
    ----------
    path : str or Path
        File to write.
    particles : pandas.DataFrame
        The particle table.
    receptor : Receptor
        Receptor the particles were released from.
    settings : mapping
        The run's settings, as its folder records them
        (:attr:`stilt.config.Variant.run_settings`).
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
        If ``time`` or ``particle`` holds a value that is not a whole number.
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
        b"stilt:receptor": receptor.to_json().encode(),
        b"stilt:settings": json.dumps(dict(settings)).encode(),
        b"stilt:met_files": json.dumps([str(p) for p in met_files]).encode(),
        b"stilt:reach_minutes": repr(
            float(np.abs(particles["time"].to_numpy(dtype=float)).max(initial=0.0))
        ).encode(),
        **(metadata or {}),
    }
    table = table.replace_schema_metadata(meta)
    return write_parquet(table, location(path))


# -- after a model run: release heights ------------------------------------

# Below this horizontal spacing, release points cannot be told apart from a
# particle's first output row. In a test with HRRR at WBB, particles moved
# 200-600 m in the first minute, by an amount that varied with height. At
# 1000 m spacing the release height was recovered to about 15 m, and at 300 m
# it was off by about 190 m.
_MIN_RELIABLE_SPACING_M = 1000.0


def add_release_heights(particles: pd.DataFrame, receptor: Receptor) -> pd.DataFrame:
    """
    Return *particles* with each particle's release height ``xhgt``, for a column or multipoint receptor.

    The release row (``time = 0``) says where each particle started. For a
    multipoint receptor that is one of its points, and ``xhgt`` is that
    point's altitude. For a column receptor it is inside one of the
    ``numpar`` slabs the column is split into, and ``xhgt`` is that slab's
    centre: the slab is what the particle stands for, not the random
    height inside it. A model that writes no release row, such as the
    bundled HYSPLIT, falls back to matching: a column's particles are
    released bottom to top in ``particle`` order, and a multipoint receptor's
    are matched to the nearest point from their first row.

    A point receptor's particles are returned as they are.
    """
    if isinstance(receptor, ColumnReceptor):
        return particles.assign(xhgt=_column_release_heights(particles, receptor))
    if isinstance(receptor, MultiPointReceptor):
        return particles.assign(xhgt=_multipoint_release_heights(particles, receptor))
    return particles


def _release_rows(p: pd.DataFrame) -> pd.DataFrame | None:
    """Return each particle's release row (``time = 0``), or ``None`` when the model wrote none."""
    released = p.loc[p["time"] == 0].drop_duplicates(subset="particle")
    if released.empty or len(released) != p["particle"].nunique():
        return None
    return released


def _height(rows: pd.DataFrame, receptor: Receptor) -> np.ndarray | None:
    """Return the rows' heights in the receptor's vertical reference, or ``None`` when they cannot be known."""
    if "zagl" not in rows.columns:
        return None
    if receptor.altitude_ref == "agl":
        return rows["zagl"].to_numpy(dtype=float)
    if "zsfc" in rows.columns:
        return (rows["zagl"] + rows["zsfc"]).to_numpy(dtype=float)
    return None


def _column_release_heights(p: pd.DataFrame, receptor: ColumnReceptor) -> pd.Series:
    """
    Return each row's release height for a column receptor: the centre of its particle's slab.

    The column is split into ``numpar`` slabs of equal depth. With release
    rows, a particle's slab is the one its release height falls in;
    without them, particle ``particle`` is in slab ``particle``, as the bundled
    HYSPLIT releases them.
    """
    numpar = int(p["particle"].max())  # type: ignore[arg-type]
    step = (receptor.top - receptor.bottom) / numpar
    released = _release_rows(p)
    height = None if released is None else _height(released, receptor)
    if released is None or height is None:
        return (p["particle"] - 0.5) * step + receptor.bottom
    slab = np.clip(np.floor((height - receptor.bottom) / step), 0, numpar - 1)
    centre = dict(
        zip(released["particle"], receptor.bottom + (slab + 0.5) * step, strict=True)
    )
    return pd.Series(p["particle"].to_numpy(), index=p.index).map(centre.get)


def _multipoint_release_heights(
    p: pd.DataFrame, receptor: MultiPointReceptor
) -> pd.Series:
    """
    Return each row's release altitude for a multipoint receptor.

    The particle table does not record which point a particle came from.
    It is recovered from each particle's row nearest the release time:

    1. If the model writes release (``t = 0``) rows, match the nearest
       release point horizontally. Nothing has moved yet, so this is exact.
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
        .drop_duplicates(subset="particle")
    )
    lons = np.asarray(receptor.longitudes, dtype=float)
    lats = np.asarray(receptor.latitudes, dtype=float)
    alts = np.asarray(receptor.altitudes, dtype=float)
    has_t0 = bool((first["_age"] == 0).all())

    # Height of each particle in the receptor's own vertical reference.
    height = _height(first, receptor)

    if not has_t0 and height is not None and len(np.unique(alts)) == len(alts):
        nearest = np.argmin(np.abs(height[:, None] - alts[None, :]), axis=1)
    else:
        xy = first[["lon", "lat"]].to_numpy(dtype=float)
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
                    "model writes no t=0 row, so particles cannot be "
                    "reliably matched to their release points; 'xhgt' may be wrong. "
                    "Use a HYSPLIT build that writes release-time rows "
                    "(HysplitConfig.exe_dir) or space the points more than "
                    f"{_MIN_RELIABLE_SPACING_M:.0f} m apart.",
                    stacklevel=3,
                )

    mapping = dict(zip(first["particle"].to_numpy(), alts[nearest], strict=True))
    return pd.Series(p["particle"].to_numpy(), index=p.index).map(mapping.get)


def correct_near_field(
    particles: pd.DataFrame, receptor: Receptor, veght: float
) -> pd.DataFrame:
    """
    Correct ``foot`` for plume dilution in the hyper-near field, as STILT-R's ``calc_plume_dilution`` does.

    ``foot`` assumes surface fluxes are mixed through the lowest ``veght``
    fraction of the mixed layer. Close to the receptor, the plume from the
    release point is thinner than that. Following STILT-R, the plume depth
    grows from the release height with the turbulence each particle meets.
    While it is below ``veght`` times the mixed-layer height, ``foot`` is
    recalculated with the plume depth in its place. The worker applies it
    to any model's particles when ``hnf_plume`` is set.

    Needs the columns :data:`HNF_PLUME_COLUMNS`, and the release height:
    ``xhgt`` (:func:`add_release_heights`) or the altitude of a point
    receptor.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table.
    receptor : Receptor
        Receptor the particles were released from.
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
        If a required column is missing, or the release height is unknown.
    """
    missing = set(HNF_PLUME_COLUMNS) - set(particles.columns)
    if missing:
        raise ValueError(
            "The near-field correction needs the particle columns "
            f"{', '.join(sorted(missing))}."
        )
    r_zagl = receptor.altitude if isinstance(receptor, PointReceptor) else None

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
        raise ValueError(
            "The near-field correction needs each particle's release height: "
            "add xhgt first (add_release_heights)."
        )
    # The plume grows outward from the release point, so the cumsum must walk
    # each particle track in order of elapsed time since release. That is
    # |time| ascending, which covers forward runs as well as backward ones.
    p["elapsed"] = abs_time_s
    p["plume"] = start_h + (
        p.sort_values("elapsed")
        .groupby("particle", sort=False)["sigma"]
        .cumsum()
        .reindex(p.index)
    )
    p["foot"] = np.where(
        p["plume"] < p["pbl_mixing"],
        _MOLAR_MASS_DRY_AIR / (p["plume"] * p["dens"]) * p["samt"] * 60,
        p["foot"],
    )
    return p.drop(columns=["sigma", "pbl_mixing", "plume", "elapsed"])
