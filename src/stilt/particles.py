"""
The particle table: reading and writing particle files, and what is computed from particles.

Besides the table itself (its schema, files, release heights, and the
near-field plume dilution correction), this module holds what a
simulation's particles give beyond the footprint: the background mole
fraction at the receptor (:func:`background`, ``sim.background``) and the
transport error of the modeled enhancement (:func:`transport_error`,
``sim.transport_error``).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from stilt._atomic import write_parquet
from stilt._paths import location, readable
from stilt.receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    parse_receptor_id,
)
from stilt.sampling import sample_field, vertical_dim
from stilt.transforms import apply_transforms, release_coordinate

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import xarray as xr
    from upath import UPath

    from stilt._paths import Location
    from stilt.visualization import ParticlesPlotAccessor


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
        the transport model's settings, the met's, the model build, and the
        realization number. :func:`stilt.identity.transport_from_settings`
        rebuilds the transport model's config from it.
    met_files : list of Path
        Meteorology files the run read.
    realization : int or None
        Which realization of an ensemble the particles are, which ran with
        ``seed + realization``; ``None`` for a single run.
    """

    receptor: Receptor
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
    >>> receptor, settings, met_files = stilt.particles.particles_metadata(
    ...     sim.particles_path
    ... )
    >>> settings["model"]
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
        receptor=Receptor.from_json(meta[b"stilt:receptor"]),
        settings=json.loads(meta[b"stilt:settings"]),
        met_files=[Path(p) for p in json.loads(meta[b"stilt:met_files"])],
        realization=None if realization is None else int(realization),
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
    receptor, the run's settings, and the met files go in the file's
    metadata, so the file reads alone.

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
        last = reach.groupby(p["particle"], sort=False).idxmax().to_numpy(dtype=int)
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
            Enhancement indexed by ``particle``. A particle that never crosses
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
            p["lon"].to_numpy(),
            p["lat"].to_numpy(),
            times=times,
            fill_value=0.0,
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


# ---------------------------------------------------------------------------
# Background
#
# Background mole fraction at a receptor, from a field sampled where the particles end.
#
# A back-trajectory ends where the receptor's air came from. Sampling a
# mole-fraction field, such as CarbonTracker or CAMS, at every particle's
# endpoint and averaging over the particles gives the background: what the
# receptor would see with no fluxes inside the domain. Adding the modeled
# enhancement gives the modeled mole fraction. X-STILT does the same in
# ``endpts.trajfoot``, and CT-STILT does it with CarbonTracker.
#
# The average is weighted the way the footprint is, so the particle
# transforms (averaging kernel, pressure weighting, lifetime decay) apply to
# the background too, and the background and enhancement add. The field is
# passed in as an :class:`xarray.DataArray`, or already sampled as one value
# per particle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Background:
    """
    Background at a receptor, returned by :func:`background`.

    Attributes
    ----------
    value : float
        Background at the receptor, weighted like the footprint:
        ``Σ weights × per_particle``. Particles with no value count as the
        weighted mean of the others.
    per_particle : pandas.Series
        Field value at each particle's endpoint, indexed by ``particle``. ``NaN``
        where the endpoint is outside the field.
    weights : pandas.Series
        Each particle's weight, indexed by ``particle``. Without transforms each
        is ``1 / N`` and they sum to one. With pressure weighting they sum to
        the fraction of the atmosphere's mass inside the column.
    """

    value: float
    per_particle: pd.Series
    weights: pd.Series


def _particle_background(particles: pd.DataFrame, field: xr.DataArray) -> pd.Series:
    """Return the field at each particle's endpoint, indexed by ``particle``."""
    ends = particles.stilt.endpoints()
    zdim = vertical_dim(field)
    z = None
    if zdim is not None:
        if zdim not in ends.columns:
            raise ValueError(
                f"The field's vertical dimension {zdim!r} is not a particle column. "
                "Name it after the column to match it against ('pres' or 'zagl'), "
                "or add that column to the particles."
            )
        z = ends[zdim].to_numpy(dtype=float)
    times = None
    if "time" in field.dims:
        if "datetime" not in ends.columns:
            raise ValueError(
                "field varies in time but the particles have no 'datetime' column."
            )
        times = ends["datetime"].to_numpy()
    values = sample_field(
        field, ends["lon"].to_numpy(), ends["lat"].to_numpy(), z=z, times=times
    )
    return pd.Series(
        values,
        index=pd.Index(ends["particle"].to_numpy(), name="particle"),
        name="background",
    )


def _endpoint_weights(
    particles: pd.DataFrame,
    transforms: Sequence[Any] = (),
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
) -> pd.Series:
    """
    Return each particle's transform weight at its endpoint, indexed by ``particle``.

    Transforms multiply ``foot``, so applying them to particles whose
    ``foot`` is 1 leaves each particle's weight: its averaging kernel and
    pressure weight, and the lifetime decay at its endpoint age. Without
    transforms every weight is 1.
    """
    transforms = list(transforms)
    if transforms:
        particles = apply_transforms(
            particles.assign(foot=1.0), transforms, receptor, directory
        )
        ends = particles.stilt.endpoints()
        weights = ends["foot"].to_numpy(dtype=float)
    else:
        ends = particles.stilt.endpoints()
        weights = np.ones(len(ends))
    return pd.Series(
        weights,
        index=pd.Index(ends["particle"].to_numpy(), name="particle"),
        name="weight",
    )


def _fill_missing(per_particle: pd.Series, weights: pd.Series) -> pd.Series:
    """
    Replace missing per-particle values with the weighted mean of the others.

    A particle whose endpoint is outside the field then does not change the
    background. The result is ``NaN`` everywhere when no particle has a
    value.
    """
    values = per_particle.reindex(weights.index).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    ok = np.isfinite(values)
    if ok.all():
        return pd.Series(values, index=weights.index, name=per_particle.name)
    total = w[ok].sum()
    mean = (w[ok] * values[ok]).sum() / total if total > 0 else np.nan
    return pd.Series(
        np.where(ok, values, mean), index=weights.index, name=per_particle.name
    )


def _background(
    particles: pd.DataFrame,
    field: xr.DataArray | pd.Series,
    *,
    transforms: Sequence[Any] = (),
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
) -> Background:
    """Return the background at a receptor (:func:`background`, which documents it)."""
    if isinstance(field, pd.Series):
        per_particle = field.rename("background")
    else:
        per_particle = _particle_background(particles, field)
    weights = _endpoint_weights(particles, transforms, receptor, directory)
    weights = weights / len(weights)
    filled = _fill_missing(per_particle, weights)
    value = float((weights * filled).sum()) if np.isfinite(filled).any() else np.nan
    return Background(
        value=value, per_particle=per_particle.reindex(weights.index), weights=weights
    )


def background(
    particles: pd.DataFrame,
    field: xr.DataArray | pd.Series,
    *,
    transforms: Sequence[Any] = (),
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
) -> Background:
    """
    Return the background mole fraction at a receptor.

    The background is the field at each particle's endpoint, averaged over
    the particles with the footprint's weights.

    Parameters
    ----------
    particles : pandas.DataFrame
        The simulation's particle table (``sim.particles``).
    field : xarray.DataArray or pandas.Series
        The background field, or one value per particle that you sampled
        yourself, as a Series indexed by ``particle``. For example, lair's
        ``CarbonTracker.sample`` on ``sim.particles.stilt.endpoints()``. A
        field is sampled at each particle's endpoint, the row farthest in
        time from release (``particles.stilt.endpoints()``). Its vertical
        dimension must be named after the particle column it is matched
        against: ``pres`` for pressure in hPa, ``zagl`` for height above
        ground in meters, or a column you add, such as height above sea
        level from ``zagl + zsfc``. Rename it with, for example,
        ``field.rename(level="pres")``. A field with a ``time`` dimension is
        sampled at the endpoint's ``datetime``.
    transforms : sequence, optional
        The footprint's particle transforms (``sim.variant.footprint.transforms``), so
        the background is weighted like the footprint and adds to its
        enhancement. A tower receptor has none.
    receptor : Receptor, optional
        The receptor (``sim.receptor``), for the transforms. An averaging
        kernel read from a table needs it, and pressure weighting reads its
        ``altitude_ref``.
    directory : str or Path, optional
        Where a kernel table's relative path starts (``project.directory``).

    Returns
    -------
    Background

    Notes
    -----
    Without transforms the value is the mean over particles. With pressure
    weighting the weights sum to the fraction of the atmosphere's mass inside
    the column, ``(p_sfc - p_top) / p_sfc``, the same fraction the
    enhancement covers. Add the part of the column above its top from the
    same field.
    """
    return _background(
        particles, field, transforms=transforms, receptor=receptor, directory=directory
    )


# ---------------------------------------------------------------------------
# Transport error
#
# Transport error of the modeled enhancement, from wind-perturbed trajectories.
#
# The method is that of Lin and Gerbig (2005, GRL, doi:10.1029/2004GL021127).
# A variant with wind-error settings (``siguverr``, ``tluverr``,
# ``zcoruverr``, ``horcoruverr``) runs the particles again with an extra
# random wind that has the statistics of the meteorology's errors. Each
# particle's enhancement is its ``foot × flux`` summed along its trajectory.
# The perturbed particles sample more of the flux field, so the enhancement
# varies more across them. The increase is the transport-error variance
# (their equation 4)::
#
#     var_transport = var(enhancement | perturbed) − var(enhancement | unperturbed)
#
# For a column receptor the difference is taken per release level, and the
# levels are combined with the column weighting and a vertical error
# correlation, as in X-STILT (Wu et al., 2018, GMD). The difference of two
# sample variances is noisy and can be negative. It is returned with its sign
# and with an estimate of its noise, so many receptors can be combined with a
# median and a single value can be compared with its noise.
# ---------------------------------------------------------------------------


#: X-STILT's empirical mean vertical correlation length of transport errors, m.
DEFAULT_LENGTH_SCALE = 356.0


@dataclass(frozen=True)
class TransportError:
    """
    Transport error of a modeled enhancement, returned by :func:`transport_error`.

    Values are in the flux's units times the footprint's, which is ppm for a
    flux in µmol m⁻² s⁻¹, and squared for variances. With a ``background``
    field, the per-particle values are modeled mole fractions (enhancement
    plus background at the endpoint), so ``enhancement`` and
    ``enhancement_perturbed`` are mole fractions and ``variance`` includes
    the background's response to the wind errors.

    Attributes
    ----------
    variance : float
        Transport-error variance, the extra spread the wind perturbation
        added. Kept with its sign, since a negative value is sampling noise.
    noise : float
        Standard deviation of ``variance`` expected with no perturbation,
        estimated from random halves of the unperturbed particles.
        ``variance`` is resolved only when it is several times ``noise``.
    enhancement : float
        Modeled enhancement from the unperturbed particles.
    enhancement_perturbed : float
        Modeled enhancement from the perturbed particles, averaged over the
        realizations.
    levels : pandas.DataFrame
        One row per release level, with columns ``height`` (mean release
        height, m), ``n`` (particles), ``weight`` (share of the particles),
        ``mean_orig``, ``var_orig``, ``mean_err``, and ``var_err`` (mean and
        variance of the per-particle enhancement without and with the
        perturbation), ``dvar`` (``var_err - var_orig``), and ``sd_trans``
        (signed square root of ``dvar``).
    length_scale : float or None
        Vertical correlation length used to combine the levels, in m.
    background : float
        Weighted background from the unperturbed particles, or 0 without a
        ``background`` field. ``enhancement - background`` is the
        enhancement alone.
    realizations : int
        Number of error realizations in the estimate.
    """

    variance: float
    noise: float
    enhancement: float
    enhancement_perturbed: float
    levels: pd.DataFrame
    length_scale: float | None
    background: float = 0.0
    realizations: int = 1

    @property
    def sd(self) -> float:
        """Transport-error standard deviation, or 0 when ``variance`` is negative."""
        return float(np.sqrt(max(self.variance, 0.0)))


def _level_edges(heights: pd.Series, levels: int | Sequence[float]) -> np.ndarray:
    """
    Return the release-height bin edges of the levels.

    Given edges are used as they are. With a number of levels, the edges
    split the height range evenly, or fall halfway between the distinct
    heights when there are no more of them than ``levels``. The outer edges
    are then open, so the perturbed particles fall in the same levels.
    """
    if not isinstance(levels, int):
        return np.asarray(levels, dtype=float)
    if levels < 1:
        raise ValueError("levels must be >= 1.")
    unique = np.unique(heights.to_numpy(dtype=float))
    if unique.size <= levels:
        edges = np.concatenate(([-np.inf], (unique[:-1] + unique[1:]) / 2.0, [np.inf]))
    else:
        edges = np.linspace(unique[0], unique[-1], levels + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _level_labels(heights: pd.Series, edges: np.ndarray) -> pd.Series:
    """Return each particle's level number, NaN outside the edges."""
    label = pd.cut(heights, edges, labels=False, include_lowest=True)
    return pd.Series(np.asarray(label, dtype=float), index=heights.index)


def _level_stats(x: pd.Series, labels: pd.Series, percentile: float) -> pd.DataFrame:
    """
    Return each level's mean and variance of ``x``, indexed by level.

    The mean is over all the level's finite values and the variance over
    those at or below ``percentile``. A level with fewer than two such
    values has NaN variance.
    """

    def var(values: pd.Series) -> float:
        v = _finite(values)
        if v.size and percentile < 1.0:
            v = v[v <= np.quantile(v, percentile)]
        return float(v.var(ddof=0)) if v.size >= 2 else np.nan

    groups = x.reindex(labels.index).groupby(labels)
    return pd.DataFrame({"mean": groups.apply(_mean), "var": groups.apply(var)})


def _finite(values: pd.Series) -> np.ndarray:
    """Return the finite values as an array."""
    v = values.to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _mean(values: pd.Series) -> float:
    """Return numpy's mean of the finite values, or NaN when there are none."""
    v = _finite(values)
    return float(v.mean()) if v.size else np.nan


def _signed_sqrt(values: np.ndarray) -> np.ndarray:
    """Return ``sign(v) * sqrt(|v|)``, with NaN as 0."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    return np.sign(v) * np.sqrt(np.abs(v))


def _combine(levels: pd.DataFrame, length_scale: float | None) -> float:
    """
    Return the column variance ``Σ_i w_i² v_i + Σ_{i≠j} w_i w_j s_i s_j corr_ij``.

    ``s`` is the per-level ``sd_trans``. On the diagonal, ``v`` is the
    signed variance difference, so a level whose spread fell counts
    negatively. The cross terms use the signed square roots, so correlated levels of the
    same sign add and levels of opposite sign cancel.
    """
    w = levels["weight"].to_numpy(dtype=float)
    s = levels["sd_trans"].to_numpy(dtype=float)
    h = levels["height"].to_numpy(dtype=float)
    diag = np.nan_to_num(levels["dvar"].to_numpy(dtype=float), nan=0.0)
    if length_scale is None:
        corr = np.eye(len(h))
    else:
        corr = np.exp(-np.abs(h[:, None] - h[None, :]) / length_scale)
    prod = w[:, None] * w[None, :] * s[:, None] * s[None, :] * corr
    np.fill_diagonal(prod, w**2 * diag)
    return float(prod.sum())


def _level_table(
    x_orig: pd.Series,
    x_errs: Sequence[pd.Series],
    label_orig: pd.Series,
    label_errs: Sequence[pd.Series],
    level_height: pd.Series,
    *,
    percentile: float,
) -> pd.DataFrame:
    """
    Return the table of means, variances, and weights per release level.

    The perturbed mean and variance of each level are averaged over the
    error realizations (one ``x_errs`` and ``label_errs`` pair each) before
    ``dvar`` is taken.
    """
    orig = _level_stats(x_orig, label_orig, percentile).reindex(level_height.index)
    errs = [
        _level_stats(x, label, percentile).reindex(level_height.index)
        for x, label in zip(x_errs, label_errs, strict=True)
    ]
    n = label_orig.value_counts().reindex(level_height.index, fill_value=0)
    table = pd.DataFrame(
        {
            "height": level_height.to_numpy(dtype=float),
            "n": n.to_numpy(dtype=int),
            "weight": n.to_numpy() / len(x_orig),
            "mean_orig": orig["mean"].to_numpy(),
            "mean_err": _nanmean([e["mean"].to_numpy(dtype=float) for e in errs]),
            "var_orig": orig["var"].to_numpy(),
            "var_err": _nanmean([e["var"].to_numpy(dtype=float) for e in errs]),
        }
    )
    table["dvar"] = table["var_err"] - table["var_orig"]
    table["sd_trans"] = _signed_sqrt(table["dvar"].to_numpy())
    return table


def _nanmean(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean across *columns* per row, ignoring NaN; NaN where all are."""
    arr = np.column_stack(columns)
    ok = np.isfinite(arr)
    count = ok.sum(axis=1)
    total = np.where(ok, arr, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(arr), np.nan), where=count > 0)


def _noise(
    x_orig: pd.Series,
    label_orig: pd.Series,
    level_height: pd.Series,
    *,
    splits: int,
    percentile: float,
    length_scale: float | None,
) -> float:
    """
    Return the standard deviation of ``variance`` with no perturbation.

    The unperturbed particles of each level are split into two random halves,
    treated as the unperturbed and perturbed tables. The spread of the
    estimate over ``splits`` random splits, divided by ``sqrt(2)`` to go from
    half to full ensembles, is the noise. Returns NaN for fewer than two
    splits.
    """
    if splits < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    indx = x_orig.index.to_numpy()
    lab = label_orig.reindex(indx).to_numpy()
    estimates = []
    for _ in range(splits):
        half = np.zeros(len(indx), dtype=bool)
        for lvl in np.unique(lab[np.isfinite(lab)]):
            members = np.flatnonzero(lab == lvl)
            chosen = rng.permutation(members)[: len(members) // 2]
            half[chosen] = True
        a, b = x_orig.iloc[half], x_orig.iloc[~half]
        table = _level_table(
            a,
            [b],
            label_orig.reindex(a.index),
            [label_orig.reindex(b.index)],
            level_height,
            percentile=percentile,
        )
        table["weight"] = table["n"] / table["n"].sum()
        estimates.append(_combine(table, length_scale))
    return float(np.std(estimates, ddof=1) / np.sqrt(2.0))


def transport_error(
    particles: pd.DataFrame,
    error_particles: pd.DataFrame | Sequence[pd.DataFrame],
    flux: xr.DataArray,
    *,
    transforms: Sequence[Any] = (),
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
    levels: int | Sequence[float] = 20,
    length_scale: float | None = DEFAULT_LENGTH_SCALE,
    percentile: float = 1.0,
    noise_splits: int = 16,
    background: xr.DataArray | None = None,
) -> TransportError:
    """
    Return the transport-error variance of a modeled enhancement (Lin and Gerbig, 2005).

    Parameters
    ----------
    particles : pandas.DataFrame
        Unperturbed particle table of a receptor (``sim.particles``).
    error_particles : pandas.DataFrame or sequence of pandas.DataFrame
        Particle table of a wind-error variant of the same receptor, or a
        list of them for a variant with ``realizations: N``, such as
        ``[rows for _, rows in project.particles(ensemble).groupby("variant")]``.
    flux : xarray.DataArray
        Surface flux field (see ``particles.stilt.enhancement``).
    transforms : sequence, optional
        The footprint's particle transforms (``sim.variant.footprint.transforms``),
        applied to both tables so the error is weighted like the footprint.
    receptor : Receptor, optional
        The receptor (``sim.receptor``), for the transforms.
    directory : str or Path, optional
        Where a kernel table's relative path starts (``project.directory``).
    levels : int or sequence of float, default 20
        Release-height levels to compute statistics on: a number of
        equal-width bins between the lowest and highest release height, or
        bin edges in meters. When the particles have no more distinct
        release heights than ``levels`` (a multipoint receptor), each height
        is a level. A point receptor is one level.
    length_scale : float or None, default 356.0
        Vertical e-folding length of the error correlation between levels,
        in meters (X-STILT's value). ``None`` treats the levels as
        uncorrelated. It has no effect for a point receptor.
    percentile : float, default 1.0
        In each level, drop particles above this quantile of the enhancement
        before taking the variance. 1.0 keeps every particle, as Lin and
        Gerbig do. X-STILT uses 0.99 to limit the few particles that cross a
        point source. Means always use every particle.
    noise_splits : int, default 16
        Number of random half-splits of the unperturbed particles used to
        estimate ``noise``. Fewer than 2 skips it and gives NaN.
    background : xarray.DataArray, optional
        Background field sampled at each particle's endpoint (see
        :func:`background`). Wind errors move the
        endpoints as well as the surface contact, so with a field the
        statistics are of the modeled mole fraction per particle, as X-STILT
        computes them.

    Returns
    -------
    TransportError

    Notes
    -----
    Per level ``l`` with ``n_l`` of ``N`` particles, ``dvar_l`` is the change
    in the variance of the per-particle enhancement under the perturbation,
    ``s_l = sign(dvar_l) sqrt(|dvar_l|)``, and the column variance is
    ``Σ_ij w_i w_j s_i s_j exp(-|h_i - h_j| / L)`` with ``w_l = n_l / N`` and
    ``h`` the level heights. For one level it is ``dvar`` itself. The
    transforms are applied to the particles first, so the level statistics
    are already column-weighted and the level weights are particle counts.

    With several error realizations, each level's ``mean_err`` and
    ``var_err`` are averaged over them before ``dvar`` is formed, so the
    perturbed side's sampling noise falls as ``1/sqrt(N)``. The unperturbed
    side is the same particles in every realization, so its noise does not
    fall: with ``N`` realizations the null spread of ``variance`` is
    ``sqrt((1 + 1/N) / 2)`` times the single-realization ``noise``, which
    tends to ``1/sqrt(2)``. ``noise`` includes that factor. More
    realizations therefore give at most a ``sqrt(2)`` tighter estimate and
    cannot resolve a case that one realization leaves unresolved.

    The signal is the extra spread the perturbation adds. It is small when
    the wind error decorrelates quickly (HYSPLIT decorrelates it with the
    distance a particle travels as well as with time), and when turbulence
    already spreads the particles widely, as on a convective afternoon.
    Compare a single value with ``noise``. Over many receptors, combine the
    signed ``variance`` values with a median instead of clipping each at
    zero.
    """
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1].")
    if length_scale is not None and length_scale <= 0:
        raise ValueError("length_scale must be > 0 or None.")
    if noise_splits < 0:
        raise ValueError("noise_splits must be >= 0.")
    transforms = list(transforms)
    error_tables = (
        [error_particles]
        if isinstance(error_particles, pd.DataFrame)
        else list(error_particles)
    )
    if not error_tables:
        raise ValueError("error_particles must hold at least one realization.")

    def _prepare(
        table: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, float]:
        """Return the modeled value and release height per particle, and the background."""
        weighted = (
            apply_transforms(table, transforms, receptor, directory)
            if transforms
            else table
        )
        x = weighted.stilt.enhancement(flux)
        if background is None:
            return x, _release_heights(weighted), 0.0
        bg = _background(
            table,
            background,
            transforms=transforms,
            receptor=receptor,
            directory=directory,
        )
        # Each particle's background, weighted like its enhancement. A
        # particle's enhancement is N times its share of the footprint's.
        filled = _fill_missing(bg.per_particle, bg.weights)
        x = x + (len(bg.weights) * bg.weights * filled).reindex(x.index)
        return x, _release_heights(weighted), bg.value

    x_orig, h_orig, background_value = _prepare(particles)

    edges = _level_edges(h_orig, levels)
    label_orig = _level_labels(h_orig, edges)
    level_height = h_orig.groupby(label_orig).apply(_mean)

    x_errs, label_errs = [], []
    for err in error_tables:
        x_err, h_err, _ = _prepare(err)
        x_errs.append(x_err)
        label_errs.append(_level_labels(h_err, edges))

    table = _level_table(
        x_orig,
        x_errs,
        label_orig,
        label_errs,
        level_height,
        percentile=percentile,
    )
    n_real = len(error_tables)
    # The main particles are shared by every realization, so only the
    # perturbed side's noise averages down; see the Notes.
    noise_factor = float(np.sqrt((1.0 + 1.0 / n_real) / 2.0))
    w = table["weight"].to_numpy()
    return TransportError(
        variance=_combine(table, length_scale),
        noise=noise_factor
        * _noise(
            x_orig,
            label_orig,
            level_height,
            splits=noise_splits,
            percentile=percentile,
            length_scale=length_scale,
        ),
        enhancement=float(np.nansum(w * table["mean_orig"].to_numpy())),
        enhancement_perturbed=float(np.nansum(w * table["mean_err"].to_numpy())),
        levels=table,
        length_scale=length_scale,
        background=background_value,
        realizations=n_real,
    )


def _release_heights(particles: pd.DataFrame) -> pd.Series:
    """Return each particle's release height (``xhgt``), or zeros without ``xhgt``."""
    if "xhgt" in particles.columns:
        return release_coordinate(particles, "xhgt")
    indx = np.unique(particles["particle"].to_numpy())
    return pd.Series(0.0, index=indx)
