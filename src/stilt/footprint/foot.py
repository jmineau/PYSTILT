"""
What STILT computes from a particle table before gridding it: ``foot`` and the near-field correction.

:func:`add_foot` gives each particle its sensitivity to surface fluxes
(``foot``) for a transport model that writes only positions, or with
another mixed-layer height (``pbl_height``). :func:`correct_near_field`
corrects ``foot`` for the thin plume close to the release, as STILT-R's
``calc_plume_dilution`` does. The footprint stage applies both
(:func:`stilt.footprint.footprint_particles`); a particle file holds what
the model wrote.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from runtraj.releases import PointRelease, Release

from stilt.sampling import calc_pbl_height, sample_met

logger = logging.getLogger(__name__)

#: Particle columns the near-field plume correction reads (``varsiwant`` must include them).
HNF_PLUME_COLUMNS: tuple[str, ...] = ("dens", "samt", "sigw", "tlgr", "foot", "mlht")

#: Molar mass of dry air, kg/mol.
_MOLAR_MASS_DRY_AIR = 0.02897

#: Standard gravity, m/s².
_GRAVITY = 9.80665


def _above_ground(p: pd.DataFrame, start: pd.Series) -> pd.Series:
    """
    Return release heights above sea level (*start*, one per row) as heights above ground.

    The terrain under a release height is the median ``zsfc`` of the first
    rows of the particles released there: a step after release they have
    moved a little, so the median stands for the terrain at the release
    point. A height below that terrain is the ground.
    """
    first = (
        p.assign(_age=p["age"].abs(), _start=start)
        .sort_values("_age", kind="stable")
        .drop_duplicates(subset="particle")
    )
    terrain = first.groupby("_start")["zsfc"].median()
    return (start - start.map(terrain)).clip(lower=0.0)


def _mixing_depth(mlht: pd.Series, footprint_depth: float) -> pd.Series | float:
    """Return the depth surface fluxes are mixed through: *footprint_depth* of *mlht*, or *footprint_depth* meters above 1."""
    return footprint_depth * mlht if footprint_depth <= 1 else footprint_depth


def add_foot(
    particles: pd.DataFrame,
    met_files: Iterable[str | os.PathLike[str]],
    footprint_depth: float,
    pbl_height: str | float | None = None,
    pbl_height_min: float = 0.0,
    z_top: float | None = None,
) -> pd.DataFrame:
    """
    Add ``foot``, each particle's sensitivity to surface fluxes, for a transport model that writes only positions.

    A particle below the depth surface fluxes are mixed through,
    *footprint_depth* times ``mlht`` (or *footprint_depth* meters when it is
    above 1), takes up the
    flux for the time it spends there. Its ``foot`` for one output step is
    that time divided by the air in the layer, STILT's footprint (Lin et
    al., 2003)::

        foot = M · t / (ρ̄ · depth)

    in ppm per (µmol m⁻² s⁻¹), with M = 0.02897 kg mol⁻¹, the molar mass of
    dry air, t the time in seconds, and ρ̄ the mean air density below the
    depth. ρ̄ · depth, the mass of air in the layer, is the drop in pressure
    from the ground to the depth over g, from the met files under the
    particle. The time is ``samt`` when the table has it. Otherwise the
    particle counts for the whole step when it is below the depth at the
    end of the step, and not at all when it is above. A step is the time
    since the particle's previous row, or since its release for its first
    row, so a release row at ``age = 0`` gets zero.

    The mixed-layer height comes from *pbl_height*. Unset, a ``mlht``
    column is used as it is, with ``samt``; without one, the met's
    boundary-layer height is sampled. Set, ``mlht`` is replaced with the
    height it names, and ``samt``, which counts the minutes below the
    model's own layer, is not used. Every model moved its particles with
    its own mixed layer, so a height found here can differ from the one the
    particles were moved with.

    HYSPLIT computes ``foot`` itself in STILT mode, the same way in its
    turbulence steps. :doc:`/development/validation` says how far a
    footprint made from this ``foot`` is from one made from HYSPLIT's. The
    footprint stage calls it when it needs a ``foot`` the particles do not
    have (:func:`stilt.footprint.footprint_particles`), and stores it
    nowhere.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``time``, the release time plus ``age``
        minutes. A ``mlht`` column (m above ground) is used as it is
        unless *pbl_height* is set.
    met_files : iterable of path-like
        ARL met files that cover the particles' times, such as the files
        the run read. They are read on their own levels, which start at the
        ground.
    footprint_depth : float
        Height that surface fluxes are mixed through (``VEGHT`` in HYSPLIT
        and STILT-R): a fraction of the mixed-layer height when 1 or less,
        else meters above ground.
    pbl_height : {"met", "richardson"}, float, or None, default None
        The mixed-layer height: ``"met"`` samples the met's ``PBLH``
        (:func:`stilt.sampling.sample_met`), ``"richardson"`` finds it from
        the met's profile as HYSPLIT's ``kmixd: 3`` does
        (:func:`stilt.sampling.calc_pbl_height`), and a number is a fixed
        height in meters. ``None`` uses the particles' ``mlht``, or the
        met's ``PBLH`` when they have none.
    pbl_height_min : float, default 0
        Lowest mixed-layer height in meters, for a height found here rather
        than the particles' own, as ``kmix0`` is for HYSPLIT.
    z_top : float, optional
        The top of a HYSPLIT run's domain in meters, which sets the internal
        levels ``"richardson"`` works on, as in that run. ``None`` uses
        HYSPLIT's default, 25000 m. Only ``"richardson"`` reads it.

    Returns
    -------
    pandas.DataFrame
        A copy of *particles* with ``foot``, and ``mlht`` when it was
        found here. A row off the met's grid gets NaN, which no footprint
        counts.

    Raises
    ------
    ValueError
        If *particles* has a ``foot`` column already, has no ``time``
        column, or has a row outside the files' times.

    Examples
    --------
    >>> particles = sim.particles.drop(columns="foot")  # doctest: +SKIP
    >>> meta = runtraj.particles.particles_metadata(
    ...     sim.particles_path
    ... )  # doctest: +SKIP
    >>> stilt.footprint.add_foot(
    ...     particles, meta.met_files, footprint_depth=0.5
    ... )  # doctest: +SKIP
    """
    if "foot" in particles.columns:
        raise ValueError(
            "The particles have a foot column already. Drop it first to compute it."
        )
    files = list(met_files)
    own_mlht = pbl_height is None and "mlht" in particles.columns
    out = particles.copy()
    if not own_mlht:
        found = _pbl_height(
            out.drop(columns="mlht", errors="ignore"), files, pbl_height, z_top
        )
        out["mlht"] = np.maximum(found, pbl_height_min)
    depth = np.broadcast_to(
        np.asarray(_mixing_depth(out["mlht"], footprint_depth), dtype=float), len(out)
    )
    # The pressure at the ground and at the depth, in one call.
    where = out[["lon", "lat", "time"]]
    ends = pd.concat(
        [where.assign(zagl=0.0), where.assign(zagl=depth)], ignore_index=True
    )
    pres = sample_met(ends, files, ["pres"], z_top=None)["pres"].to_numpy()
    air = 100.0 * (pres[: len(out)] - pres[len(out) :]) / _GRAVITY  # kg m⁻²
    if own_mlht and "samt" in out.columns:
        minutes = out["samt"].to_numpy(dtype=float)
    else:
        below = out["zagl"].to_numpy(dtype=float) < depth
        minutes = np.where(below, _step_minutes(out), 0.0)
    out["foot"] = _MOLAR_MASS_DRY_AIR * minutes * 60 / air
    return out


def _pbl_height(
    particles: pd.DataFrame,
    files: list[Any],
    pbl_height: str | float | None,
    z_top: float | None = None,
) -> np.ndarray:
    """Return the mixed-layer height at each row that *pbl_height* names, the met's ``PBLH`` for ``None``."""
    if pbl_height is None or pbl_height == "met":
        return sample_met(particles, files, ["mlht"], z_top=None)["mlht"].to_numpy()
    if pbl_height == "richardson":
        if z_top is None:
            return calc_pbl_height(particles, files)
        return calc_pbl_height(particles, files, z_top=z_top)
    if isinstance(pbl_height, str):
        raise ValueError(
            "pbl_height must be met, richardson, or a height in meters; "
            f"got {pbl_height!r}."
        )
    return np.full(len(particles), float(pbl_height))


def _step_minutes(particles: pd.DataFrame) -> np.ndarray:
    """Return each row's minutes since the particle's previous row, or since its release for its first."""
    elapsed = np.abs(particles["age"].to_numpy(dtype=float))
    number = particles["particle"].to_numpy()
    order = np.lexsort((elapsed, number))
    walked = elapsed[order]
    step = np.diff(walked, prepend=0.0)
    first = np.r_[True, number[order][1:] != number[order][:-1]]
    step[first] = walked[first]
    minutes = np.empty_like(step)
    minutes[order] = step
    return minutes


def correct_near_field(
    particles: pd.DataFrame, release: Release, footprint_depth: float
) -> pd.DataFrame:
    """
    Correct ``foot`` for plume dilution in the hyper-near field, as STILT-R's ``calc_plume_dilution`` does.

    ``foot`` assumes surface fluxes are mixed through the air below
    *footprint_depth*: a fraction of the mixed-layer height when it is 1 or
    less, and a height in meters above ground when it is larger, as HYSPLIT
    reads ``VEGHT``. Close to the release, the plume from the release point is
    thinner than that. Following STILT-R, the plume depth grows from the
    release height with the turbulence each particle meets. While it is
    below that height, ``foot`` is recalculated with the plume
    depth in its place. :func:`stilt.calc_footprint` applies it to any
    model's particles when ``hnf_plume`` is set, so a stored particle file
    holds the uncorrected ``foot``; call it yourself before gridding the
    particles another way, or before ``particles.stilt.enhancement``.

    Needs the columns :data:`HNF_PLUME_COLUMNS`, and the release height:
    ``release_height`` (:func:`runtraj.particles.add_release_heights`) or the altitude of a point
    release. The plume grows from the ground, so for a release whose
    heights are above sea level (``altitude_ref="msl"``) it starts at the
    release height minus the terrain under it, the median ``zsfc`` of the
    first rows of the particles released there, which must be a column too.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table.
    release : Release
        Release the particles were released from.
    footprint_depth : float
        Height that surface fluxes are mixed through (``VEGHT`` in HYSPLIT
        and STILT-R): a fraction of the mixed-layer height when 1 or less,
        else meters above ground.

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
    if release.altitude_ref == "msl" and "zsfc" not in particles.columns:
        missing.add("zsfc")
    if missing:
        raise ValueError(
            "The near-field correction needs the particle columns "
            f"{', '.join(sorted(missing))}."
        )
    r_zagl = release.altitude if isinstance(release, PointRelease) else None

    p = particles.copy()
    p["foot_no_hnf_dilution"] = p["foot"]

    abs_time_s = np.abs(p["age"] * 60)
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
    p["pbl_mixing"] = _mixing_depth(p["mlht"], footprint_depth)

    start_h = p["release_height"] if "release_height" in p.columns else r_zagl
    if start_h is None:
        raise ValueError(
            "The near-field correction needs each particle's release height: "
            "add release_height first (add_release_heights)."
        )
    if release.altitude_ref == "msl":
        # mlht and footprint_depth are heights above ground, so the plume is too.
        start_h = _above_ground(p, pd.Series(start_h, index=p.index, dtype=float))
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


def _near_field(
    particles: pd.DataFrame, release: Release, footprint_depth: float
) -> pd.DataFrame:
    """
    Return *particles* with the near-field correction, or as they are when they lack its columns.

    A transport model that writes no ``dens``, ``samt``, ``sigw``,
    ``tlgr``, or ``mlht`` (or no ``zsfc``, for a release above sea level)
    gets no correction, and a warning says so.
    """
    needed = set(HNF_PLUME_COLUMNS) | (
        {"zsfc"} if release.altitude_ref == "msl" else set()
    )
    missing = sorted(needed - set(particles.columns))
    if missing:
        logger.warning(
            "%s: near-field correction skipped: no %s columns",
            release.id,
            ", ".join(missing),
        )
        return particles
    return correct_near_field(particles, release, footprint_depth)


__all__ = ["HNF_PLUME_COLUMNS", "add_foot", "correct_near_field"]
