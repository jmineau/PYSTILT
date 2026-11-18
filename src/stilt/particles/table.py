"""
The particle table: its columns, particle files, and release heights.

What STILT computes from a particle table before gridding it, ``foot`` for a
model that writes none and the near-field plume correction, is in
:mod:`stilt.footprint.foot`.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stilt._paths import location, readable, write_parquet
from stilt.releases import (
    ColumnReceptor,
    MultiPointReceptor,
    Receptor,
    parse_release_id,
)

if TYPE_CHECKING:
    from upath import UPath

    from stilt._paths import Location


#: The columns every particle table has, as a particle file stores them:
#: what a transport model's run must return. ``particle`` is the particle number
#: (1 to ``numpar``), ``age`` the minutes since release (negative for a
#: backward run), ``lon`` and ``lat`` the position in degrees, and ``zagl``
#: the height above ground in meters. The reference page on the particle
#: table says what the other columns are.
PARTICLE_SCHEMA = pa.schema(
    [
        ("particle", pa.int32()),
        ("age", pa.int32()),
        ("lon", pa.float64()),
        ("lat", pa.float64()),
        ("zagl", pa.float64()),
    ]
)

#: Particle columns stored as int32 rather than float64.
_INT_COLUMNS = ("age", "particle")

#: The column a footprint is made from. It is each particle's sensitivity to
#: surface fluxes over one output step.
FOOTPRINT_COLUMNS: tuple[str, ...] = ("foot",)

logger = logging.getLogger(__name__)


class ParticleMetadata(NamedTuple):
    """
    What a particle file records about the run that made it.

    Attributes
    ----------
    release : Receptor
        Release the particles were released from.
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
    """

    release: Receptor
    settings: dict[str, Any]
    met_files: list[Path]
    realization: int | None = None


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
    if missing == ["foot"]:
        _need_foot(particles)
    if missing:
        raise ValueError(
            f"The particle table has no {', '.join(repr(m) for m in missing)} "
            f"column{'s' if len(missing) > 1 else ''}. Every particle table has "
            f"{', '.join(PARTICLE_SCHEMA.names)} (stilt.particles.PARTICLE_SCHEMA)."
        )


def _need_foot(particles: pd.DataFrame) -> None:
    """
    Raise unless *particles* have a ``foot`` column, saying how to get one.

    A particle file holds what the transport model wrote, and a model that
    writes only positions writes no ``foot``: the footprint stage computes
    one when it needs it and stores it nowhere.

    Raises
    ------
    ValueError
        If there is no ``foot`` column.
    """
    if "foot" in particles.columns:
        return
    raise ValueError(
        "The particle table has no 'foot' column: the transport model wrote "
        "none, and the footprint stage stores the foot it computes nowhere. "
        "stilt.footprint.footprint_particles(particles, config, met_files) adds "
        "the one the footprint uses, from the met files the run read "
        "(stilt.particles.particles_metadata(path).met_files), with the "
        "variant's footprint settings (sim.variant.stages['footprint'])."
    )


def particles_metadata(path: str | Path | UPath) -> ParticleMetadata:
    """
    Return the release, run settings, and met files a particle file records.

    Raises
    ------
    ValueError
        If the file records no run settings, as files written before
        PYSTILT recorded them do.

    Examples
    --------
    >>> meta = stilt.particles.particles_metadata(sim.particles_path)  # doctest: +SKIP
    >>> meta.settings["model"]  # doctest: +SKIP
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
    return ParticleMetadata(
        release=Receptor.from_json(meta[b"stilt:release"]),
        settings=json.loads(meta[b"stilt:settings"]),
        met_files=[Path(p) for p in json.loads(meta[b"stilt:met_files"])],
        realization=None if realization is None else int(realization),
    )


def read_particles(
    path: str | Path | UPath, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Read a particle file.

    It needs nothing but the file: the release time in its metadata gives
    the ``time`` column (UTC) back, from ``age``. ``age`` and ``particle`` come back as
    float64, as HYSPLIT writes them. :func:`particles_metadata` reads the
    release and settings.

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
    >>> particles = stilt.read_particles(sim.particles_path)  # doctest: +SKIP
    >>> particles.stilt.endpoints()  # doctest: +SKIP
    """
    with readable(path) as source:
        pf = pq.ParquetFile(source)
        stored = pf.schema_arrow.names
        # The output directory adds a ``release`` column for scans of the
        # whole tree. One file is one release, so it is not read here.
        wanted = [c for c in (stored if columns is None else columns) if c != "release"]
        want_time = "time" in wanted or columns is None
        read = [c for c in wanted if c in stored]
        if want_time and "age" in stored and "age" not in read:
            read.append("age")  # time is built from it
        data = pf.read(columns=read).to_pandas()
    for name in _INT_COLUMNS:
        if name in data.columns:
            data[name] = data[name].astype("float64")
    if want_time and "age" in data.columns:
        meta = pf.schema_arrow.metadata or {}
        release = Receptor.from_json(meta[b"stilt:release"])
        data["time"] = pd.Timestamp(release.time) + pd.to_timedelta(
            data["age"].to_numpy(), unit="min"
        )
    return data


def particles_from_table(table: pa.Table, time: bool = True) -> pd.DataFrame:
    """
    Return a table of many releases' particles as one DataFrame.

    The table is what :meth:`stilt.output.Output.table` reads: a
    ``release`` column, the stored particle columns, and ``date``. The
    result has ``release`` first, ``age`` and ``particle`` as float64, and
    ``time`` (UTC) rebuilt from each release's time and ``age`` (unless
    *time* is false, or there is no ``age``); ``date`` is dropped.
    """
    data = table.unify_dictionaries().to_pandas()
    data = data.drop(columns=["date"], errors="ignore")
    if data.empty:
        return data
    # The release column is a dictionary of a few ids (a Categorical here), so
    # the strings and the times are made once per id and taken for every row:
    # converting each of millions of rows takes seconds.
    release = data["release"]
    if not isinstance(release.dtype, pd.CategoricalDtype):
        release = release.astype("category")
    ids = release.cat.categories.astype(str)
    codes = release.cat.codes.to_numpy()
    data["release"] = ids.take(codes)
    for name in _INT_COLUMNS:
        if name in data.columns:
            data[name] = data[name].astype("float64")
    if time and "age" in data.columns:
        release_time = pd.to_datetime([parse_release_id(r)[0] for r in ids])
        data["time"] = pd.Series(
            release_time.take(codes), index=data.index
        ) + pd.to_timedelta(data["age"].to_numpy(), unit="min")
    return data


def write_particles(
    path: str | Path | UPath,
    particles: pd.DataFrame,
    release: Receptor,
    settings: Mapping[str, Any],
    met_files: list[Path],
    metadata: dict[bytes, bytes] | None = None,
) -> Location:
    """
    Write a particle table to a Parquet file that :func:`read_particles` reads alone.

    ``age`` and ``particle`` are stored as int32, and ``time`` is left out
    since it is the release time plus ``age``. A ``release`` column holds
    the release id, so a scan of many files can tell releases apart. The
    release, the run's settings, and the met files go in the file's
    metadata, so the file reads alone.

    Parameters
    ----------
    path : str or Path
        File to write.
    particles : pandas.DataFrame
        The particle table.
    release : Receptor
        Release the particles were released from.
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
        If ``age`` or ``particle`` holds a value that is not a whole number.
    """
    data = particles.drop(columns=["time", "release"], errors="ignore")
    for name in _INT_COLUMNS:
        if name in data.columns:
            values = data[name].to_numpy()
            if not np.array_equal(values, np.round(values)):
                raise ValueError(f"Particle column {name!r} is not whole numbers.")
            data = data.assign(**{name: values.astype(np.int32)})
    table = pa.Table.from_pandas(data, preserve_index=False)
    table = table.add_column(
        0,
        pa.field("release", pa.dictionary(pa.int32(), pa.string())),
        pa.DictionaryArray.from_arrays(
            pa.array(np.zeros(table.num_rows, dtype=np.int32)),
            pa.array([str(release.id)]),
        ),
    )
    meta = {
        b"stilt:release": release.to_json().encode(),
        b"stilt:settings": json.dumps(dict(settings)).encode(),
        b"stilt:met_files": json.dumps([str(p) for p in met_files]).encode(),
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


def _closest_points_m(longitudes: np.ndarray, latitudes: np.ndarray) -> float:
    """Return the horizontal distance in meters between the two closest of at least two points."""
    x = longitudes * np.cos(np.radians(latitudes.mean())) * 111_320.0
    y = latitudes * 111_320.0
    gaps = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return float(gaps[np.triu_indices(len(x), k=1)].min())


def needs_height_to_match(
    longitudes: Iterable[float],
    latitudes: Iterable[float],
    altitudes: Iterable[float],
) -> bool:
    """
    Return whether a multipoint release's particles can be told from their points only by height.

    :func:`add_release_heights` matches them by height when every point has its
    own altitude and the model writes no release (``t = 0``) rows; for a
    release above sea level that height is ``zagl + zsfc``. Points that share
    an altitude are matched by position, and so are points more than
    ``1000 m`` apart, which position tells apart reliably. This is the one
    case left: every altitude differs and two points are closer than that.
    A model that writes release rows needs no height at all; this function
    does not know the model.
    """
    lons = np.asarray(list(longitudes), dtype=float)
    lats = np.asarray(list(latitudes), dtype=float)
    alts = np.asarray(list(altitudes), dtype=float)
    return (
        len(alts) > 1
        and len(np.unique(alts)) == len(alts)
        and _closest_points_m(lons, lats) < _MIN_RELIABLE_SPACING_M
    )


def add_release_heights(particles: pd.DataFrame, release: Receptor) -> pd.DataFrame:
    """
    Return *particles* with each particle's ``release_height``, for a column or multipoint release.

    The release row (``age = 0``) says where each particle started. For a
    multipoint release that is one of its points, and ``release_height`` is that
    point's altitude. For a column release it is inside one of the
    ``numpar`` slabs the column is split into, and ``release_height`` is that slab's
    center: the slab is what the particle stands for, not the random
    height inside it. A model that writes no release row, such as HYSPLIT
    5.1.0 (STILT-R's build), falls back to matching: a column's particles are
    released bottom to top in ``particle`` order, and a multipoint release's
    are matched to the nearest point from their first row.

    A point release's particles are returned as they are.
    """
    if isinstance(release, ColumnReceptor):
        return particles.assign(
            release_height=_column_release_heights(particles, release)
        )
    if isinstance(release, MultiPointReceptor):
        return particles.assign(
            release_height=_multipoint_release_heights(particles, release)
        )
    return particles


def _release_rows(p: pd.DataFrame) -> pd.DataFrame | None:
    """Return each particle's release row (``age = 0``), or ``None`` when the model wrote none."""
    released = p.loc[p["age"] == 0].drop_duplicates(subset="particle")
    if released.empty or len(released) != p["particle"].nunique():
        return None
    return released


def _height(rows: pd.DataFrame, release: Receptor) -> np.ndarray | None:
    """Return the rows' heights in the release's vertical reference, or ``None`` when they cannot be known."""
    if "zagl" not in rows.columns:
        return None
    if release.altitude_ref == "agl":
        return rows["zagl"].to_numpy(dtype=float)
    if "zsfc" in rows.columns:
        return (rows["zagl"] + rows["zsfc"]).to_numpy(dtype=float)
    return None


def _column_release_heights(p: pd.DataFrame, release: ColumnReceptor) -> pd.Series:
    """
    Return each row's release height for a column release: the center of its particle's slab.

    The column is split into ``numpar`` slabs of equal depth. With release
    rows, a particle's slab is the one its release height falls in;
    without them, particle ``particle`` is in slab ``particle``, as HYSPLIT
    releases them.
    """
    numpar = int(p["particle"].max())
    step = (release.top - release.bottom) / numpar
    released = _release_rows(p)
    height = None if released is None else _height(released, release)
    if released is None or height is None:
        return (p["particle"] - 0.5) * step + release.bottom
    slab = np.clip(np.floor((height - release.bottom) / step), 0, numpar - 1)
    center = dict(
        zip(released["particle"], release.bottom + (slab + 0.5) * step, strict=True)
    )
    return pd.Series(p["particle"].to_numpy(), index=p.index).map(center.get)


def _multipoint_release_heights(
    p: pd.DataFrame, release: MultiPointReceptor
) -> pd.Series:
    """
    Return each row's release altitude for a multipoint release.

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
        p.assign(_age=p["age"].abs())
        .sort_values("_age", kind="stable")
        .drop_duplicates(subset="particle")
    )
    lons = np.asarray(release.longitudes, dtype=float)
    lats = np.asarray(release.latitudes, dtype=float)
    alts = np.asarray(release.altitudes, dtype=float)
    has_t0 = bool((first["_age"] == 0).all())

    # Height of each particle in the release's own vertical reference.
    height = _height(first, release)

    if not has_t0 and height is not None and len(np.unique(alts)) == len(alts):
        nearest = np.argmin(np.abs(height[:, None] - alts[None, :]), axis=1)
    else:
        xy = first[["lon", "lat"]].to_numpy(dtype=float)
        pts = np.column_stack((lons, lats))
        nearest = np.argmin(
            np.sum((xy[:, None, :] - pts[None, :, :]) ** 2, axis=2), axis=1
        )
        if not has_t0 and len(alts) > 1:
            spacing = _closest_points_m(lons, lats)
            if spacing < _MIN_RELIABLE_SPACING_M:
                warnings.warn(
                    f"MultiPointReceptor release points are as close as "
                    f"{spacing:.0f} m and cannot be separated by altitude, and this "
                    "model writes no t=0 row, so particles cannot be "
                    "reliably matched to their release points; 'release_height' may be wrong. "
                    "Use a HYSPLIT build that writes release-time rows "
                    "(HysplitConfig.directory) or space the points more than "
                    f"{_MIN_RELIABLE_SPACING_M:.0f} m apart.",
                    stacklevel=3,
                )

    mapping = dict(zip(first["particle"].to_numpy(), alts[nearest], strict=True))
    return pd.Series(p["particle"].to_numpy(), index=p.index).map(mapping.get)
