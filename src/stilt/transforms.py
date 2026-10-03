"""
Particle transforms that reweight ``foot`` before the footprint is computed.

A transform is any object with ``apply(particles, context) -> DataFrame``.
The built-in ones are pydantic models whose fields are their ``config.yaml``
keys::

    transforms:
      - kind: averaging_kernel
        levels: [0, 500, 1000]
        values: [1.0, 0.9, 0.7]
      - kind: pressure_weighting

An averaging kernel that differs per receptor (every satellite sounding has
its own) comes from a table in the project instead::

      - kind: averaging_kernel
        table: kernels.parquet

A ``kind`` containing a dot is the import path of your own transform class
(see the *Custom transforms* guide). Transforms run in list order, starting
from the unweighted particle table. Each returns a new table and leaves its
input unchanged.

:func:`particle_pwf` and :func:`ak_weights` hold the column weighting ported
from X-STILT. They are public so your own transforms can use them.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from stilt.receptors import Receptor


# -- interface ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransformContext:
    """
    Information about the simulation a transform is applied for.

    Attributes
    ----------
    receptor : Receptor
        Receptor the particles were released from. Its ``id`` selects
        per-receptor inputs such as a row of an averaging-kernel table.
    variant : str
        Variant the footprint is computed for.
    directory : Path or None
        Directory that file names in transform settings are relative to,
        normally the project directory.
    """

    receptor: Receptor
    variant: str = ""
    directory: Path | None = None


@runtime_checkable
class ParticleTransform(Protocol):
    """
    Interface for a transform: any object with ``apply(particles, context) -> DataFrame``.

    ``context`` is ``None`` when the caller has none to give, as when a
    background or transport error is computed from particles alone. A
    transform that needs it, such as an averaging kernel read from a table,
    raises then.
    """

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None
    ) -> pd.DataFrame:
        """Return a new, reweighted particle table, leaving ``particles`` unchanged."""
        ...


# -- science ----------------------------------------------------------------------


def release_coordinate(particles: pd.DataFrame, coordinate: str) -> pd.Series:
    """
    Return each particle's value of ``coordinate`` at release, indexed by ``indx``.

    Some columns, such as ``xhgt``, are constant along a trajectory. Others,
    such as ``pres``, change at every time step. The release value is taken
    from the row nearest the receptor time (smallest ``|time|``), or from the
    first row when there is no numeric ``time`` column.
    """
    p = particles
    if "time" in p.columns and pd.api.types.is_numeric_dtype(p["time"]):
        ordered = p.assign(_age=p["time"].abs()).sort_values("_age", kind="stable")
    else:
        ordered = p
    first = ordered.drop_duplicates(subset="indx")
    return pd.Series(
        first[coordinate].to_numpy(dtype=float), index=first["indx"].to_numpy()
    )


def particle_pwf(
    particles: pd.DataFrame,
    surface_pressure: float | None = None,
    altitude_ref: Literal["agl", "msl"] = "agl",
) -> tuple[pd.Series, pd.Series]:
    """
    Return each particle's release pressure and pressure weight.

    As in X-STILT's ``get.wgt.*.func``, a hypsometric curve
    ``ln p = b + a·z`` is fit to the particles' first-step heights and
    pressures. The fit smooths out the one time step of turbulence between
    release and the first output, and gives a surface pressure when none is
    supplied. It is evaluated at each release height, and the spacing
    between neighboring release pressures gives the air mass each release
    height represents.

    HYSPLIT spreads column particles evenly over height, each placed at
    random within its own ``1/numpar`` slab. Each release height therefore
    stands for the slab centered on it. Slab edges sit midway between
    neighboring release pressures, the ground closes the bottom slab, and the
    top slab extends above its release height as far as it does below.
    X-STILT instead gives each particle the layer below it, which shifts
    every weight down by half a slab and leaves the lowest particle with
    almost none.

    A multipoint receptor releases several particles from each point. Those
    particles share one release height, so the point's slab is split evenly
    among them.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``indx``, ``pres`` (hPa), and ``zagl`` (m), and
        optionally ``xhgt``, the release height (m). Without ``xhgt`` the
        first-step height is used. An MSL receptor also needs ``zsfc``, the
        terrain height (m above sea level).
    surface_pressure : float, optional
        Surface pressure in hPa. Defaults to the fitted curve at the ground.
    altitude_ref : {"agl", "msl"}, default "agl"
        Vertical reference of the release heights, ``receptor.altitude_ref``.
        With ``"msl"`` the fit is made against ``zagl + zsfc`` so that it can
        be evaluated at the release heights, and the ground closing the
        bottom slab is the terrain under the lowest release height.

    Returns
    -------
    xpres : pandas.Series
        Release pressure of each particle in hPa, indexed by ``indx``.
    pwf : pandas.Series
        Pressure weight of each particle, indexed by ``indx``. The weights
        sum to the fraction of the atmosphere's mass inside the column,
        ``(p_sfc - p_top) / p_sfc``. The rest lies above the column top,
        where surface fluxes do not reach the receptor.
    """
    for col in ("pres", "zagl"):
        if col not in particles.columns:
            raise ValueError(
                f"Pressure weighting requires the {col!r} particle variable; "
                "include it in varsiwant."
            )
    pres = release_coordinate(particles, "pres")
    zagl = release_coordinate(particles, "zagl")

    # Particle heights in the receptor's own vertical reference, so the fit
    # can be evaluated at the release heights.
    if altitude_ref == "msl":
        if "zsfc" not in particles.columns:
            raise ValueError(
                "Pressure weighting for a receptor with altitude_ref='msl' "
                "requires the 'zsfc' particle variable; include it in "
                "varsiwant."
            )
        zsfc = release_coordinate(particles, "zsfc")
        z_fit = zagl + zsfc
    else:
        zsfc = None
        z_fit = zagl
    z_release = (
        release_coordinate(particles, "xhgt") if "xhgt" in particles.columns else z_fit
    )

    heights = np.unique(z_release.to_numpy())  # distinct release heights, ascending
    if len(heights) < 2 or z_fit.nunique() < 2:
        raise ValueError(
            "Pressure weighting needs particles released over a range of heights "
            "(a ColumnReceptor); all particles share one release height."
        )
    a, b = np.polyfit(z_fit.to_numpy(), np.log(pres.to_numpy()), 1)
    if a >= 0:
        raise ValueError(
            "Could not fit a pressure profile to the particles (pressure does "
            "not decrease with height)."
        )

    # The ground closes the bottom slab: z = 0 above ground, or the terrain
    # under the lowest release height above sea level.
    z_ground = 0.0
    if zsfc is not None:
        at_bottom = z_release.to_numpy() == heights[0]
        z_ground = float(np.median(zsfc.to_numpy()[at_bottom]))
    p_sfc = (
        float(surface_pressure)
        if surface_pressure is not None
        else float(np.exp(b + a * z_ground))
    )

    xpres = pd.Series(
        p_sfc * np.exp(a * (z_release.to_numpy() - z_ground)), index=z_release.index
    )

    # One slab per distinct release height, surface upward, shared evenly by
    # the particles released at that height.
    levels = p_sfc * np.exp(a * (heights - z_ground))
    mids = (levels[:-1] + levels[1:]) / 2.0
    lower_edges = np.concatenate(([p_sfc], mids))
    upper_edges = np.concatenate((mids, [2 * levels[-1] - mids[-1]]))
    slab = (lower_edges - upper_edges) / p_sfc
    level = np.searchsorted(heights, z_release.to_numpy())
    count = np.bincount(level, minlength=len(heights))
    pwf = pd.Series(slab[level] / count[level], index=z_release.index)
    return xpres, pwf


def ak_weights(
    particles: pd.DataFrame,
    levels: list[float],
    values: list[float],
    coordinate: str = "xhgt",
) -> np.ndarray:
    """
    Return the averaging kernel at each particle's release coordinate.

    The kernel is interpolated linearly to each particle's release value of
    ``coordinate`` and repeated on every row of that particle. Outside
    ``levels`` it is held at its end values.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``indx`` and ``coordinate``.
    levels : list of float
        Levels the kernel is given on, in the units of ``coordinate``.
    values : list of float
        Kernel value at each level.
    coordinate : str, default "xhgt"
        Particle column to interpolate on: ``xhgt`` (release height, m) or
        ``pres`` (hPa).

    Returns
    -------
    numpy.ndarray
        Kernel value for each row of ``particles``.
    """
    if coordinate not in particles.columns:
        raise ValueError(
            f"Particle DataFrame has no column {coordinate!r}. "
            "Assign release heights ('xhgt') before applying an averaging kernel, "
            "or pass coordinate='pres' for pressure-based interpolation."
        )
    lv = np.asarray(levels, dtype=float)
    va = np.asarray(values, dtype=float)
    order = np.argsort(lv)
    lv, va = lv[order], va[order]
    per_particle = release_coordinate(particles, coordinate)
    coords = per_particle.reindex(particles["indx"].to_numpy()).to_numpy(dtype=float)
    return np.interp(coords, lv, va, left=va[0], right=va[-1])


# -- built-in transforms ----------------------------------------------------------


KERNEL_TABLE_COLUMNS = ("receptor", "level", "value")


def averaging_kernel_table(
    receptors: Iterable[Any],
    levels: Sequence[ArrayLike],
    values: Sequence[ArrayLike],
) -> pd.DataFrame:
    """
    Return a table of averaging kernels, one per receptor, for :class:`AveragingKernel`.

    The table is in long form, with ``receptor``, ``level``, and ``value``
    columns and one row per kernel level. Save it in the project with
    ``.to_parquet()`` or ``.to_csv(index=False)`` and give the file name as
    ``table`` in the ``averaging_kernel`` transform.

    Parameters
    ----------
    receptors : iterable
        :class:`~stilt.Receptor` objects or their ids.
    levels : sequence of array-like
        One array of levels per receptor, as for satellite soundings whose
        pressure grids differ, or a single array shared by all receptors.
    values : sequence of array-like
        One array of kernel values per receptor.

    Returns
    -------
    pandas.DataFrame
        The kernel table.
    """
    ids = [str(getattr(r, "id", r)) for r in receptors]
    value_arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    if len(value_arrays) != len(ids):
        raise ValueError(
            f"averaging_kernel_table: {len(ids)} receptors but "
            f"{len(value_arrays)} kernels."
        )
    try:
        shared = np.asarray(levels, dtype=float)
    except (TypeError, ValueError):  # ragged: one array per receptor
        shared = None
    if shared is not None and shared.ndim == 1:
        level_arrays = [shared] * len(ids)
    else:
        level_arrays = [np.asarray(lv, dtype=float).ravel() for lv in levels]
        if len(level_arrays) != len(ids):
            raise ValueError(
                f"averaging_kernel_table: {len(ids)} receptors but "
                f"{len(level_arrays)} level arrays."
            )
    frames = []
    for rid, lv, va in zip(ids, level_arrays, value_arrays, strict=True):
        if lv.size == 0 or lv.size != va.size:
            raise ValueError(
                f"averaging_kernel_table: receptor {rid} has {lv.size} levels "
                f"and {va.size} values."
            )
        frames.append(pd.DataFrame({"receptor": rid, "level": lv, "value": va}))
    return pd.concat(frames, ignore_index=True)


@functools.lru_cache(maxsize=8)
def _read_kernel_table(
    path: str, mtime: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Read a per-receptor averaging-kernel table from parquet or CSV.

    Cached by ``path`` and its modification time ``mtime``, so an edited
    table is read again.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
    elif suffix == ".csv":
        table = pd.read_csv(path)
    else:
        raise ValueError(
            f"averaging_kernel table {path!r} must be a .parquet or .csv file."
        )
    missing = [c for c in KERNEL_TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"averaging_kernel table {path!r} lacks columns {missing}; "
            f"expected {list(KERNEL_TABLE_COLUMNS)} (see averaging_kernel_table)."
        )
    kernels = {}
    for rid, group in table.groupby("receptor", sort=False):
        kernels[str(rid)] = (
            group["level"].to_numpy(dtype=float),
            group["value"].to_numpy(dtype=float),
        )
    return kernels


class AveragingKernel(BaseModel):
    """
    Transform that weights each particle's ``foot`` by an averaging kernel.

    Give the kernel as ``levels`` and ``values``, or name a ``table`` with
    one kernel per receptor (see :func:`averaging_kernel_table`). A relative
    ``table`` path is relative to the project root. A receptor missing from
    the table is an error.

    ``levels`` are release heights above ground in meters by default. Set
    ``coordinate: pres`` for a kernel on pressure levels in hPa. Fold any
    instrument-specific factor, such as TCCON's wet-air scaling, into the
    values. The kernel weight is added as an ``ak_weight`` column.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["averaging_kernel"] = "averaging_kernel"
    levels: list[float] | None = Field(
        default=None,
        description="Levels of the kernel, in meters above ground or hPa (see ``coordinate``).",
    )
    values: list[float] | None = Field(
        default=None, description="Normalized averaging-kernel value at each level."
    )
    table: str | None = Field(
        default=None,
        description=(
            "Table of kernels, one per receptor: a .parquet or .csv file with "
            "``receptor``, ``level``, and ``value`` columns. A relative path is "
            "relative to the project root."
        ),
    )
    coordinate: str = Field(
        default="xhgt",
        description="Particle column the levels refer to: ``xhgt`` (release height) or ``pres``.",
    )

    @model_validator(mode="after")
    def _inline_or_table(self) -> Self:
        """Require either ``levels`` and ``values`` of equal length, or a ``table``."""
        inline = self.levels is not None or self.values is not None
        if self.table is not None:
            if inline:
                raise ValueError(
                    "averaging_kernel takes either levels/values or a table, not both."
                )
            if not self.table.strip():
                raise ValueError("averaging_kernel table must be a file name.")
            return self
        if not self.levels or not self.values:
            raise ValueError(
                "averaging_kernel requires non-empty levels and values, or a table."
            )
        if len(self.levels) != len(self.values):
            raise ValueError(
                f"averaging_kernel levels ({len(self.levels)}) and values "
                f"({len(self.values)}) must have the same length."
            )
        return self

    def kernel(
        self, context: TransformContext | None = None
    ) -> tuple[list[float], list[float]]:
        """Return ``(levels, values)`` of the kernel for the receptor in ``context``."""
        if self.table is None:
            assert self.levels is not None and self.values is not None
            return self.levels, self.values
        if context is None:
            raise ValueError(
                "averaging_kernel with a table needs a TransformContext to know "
                "which receptor to look up."
            )
        path = self.table
        if context.directory is not None and not Path(path).is_absolute():
            path = str(Path(context.directory) / path)
        kernels = _read_kernel_table(path, os.stat(path).st_mtime)
        rid = str(context.receptor.id)
        try:
            levels, values = kernels[rid]
        except KeyError:
            raise KeyError(
                f"averaging_kernel table {self.table!r} has no kernel for "
                f"receptor {rid!r}."
            ) from None
        return levels.tolist(), values.tolist()

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` weighted by the averaging kernel."""
        levels, values = self.kernel(context)
        weights = ak_weights(particles, levels, values, self.coordinate)
        out = particles.copy()
        out["ak_weight"] = weights
        out["foot"] = out["foot"] * weights
        return out


class PressureWeighting(BaseModel):
    """
    Transform that weights each particle by its share of the column's air mass.

    The weights come from the particles' own first-step heights and
    pressures (see :func:`particle_pwf`). :func:`stilt.footprint.calculate`
    divides by the particle count, so the weights are multiplied by the
    number of particles and the weighted footprint does not change with
    ``numpar``. The particles of a multipoint receptor share their point's
    weight evenly. Adds ``xpres`` (release pressure, hPa) and ``pwf``
    columns. Requires ``pres`` and ``zagl`` in ``varsiwant``, which are both
    in the default, and ``zsfc`` for a receptor with ``altitude_ref="msl"``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pressure_weighting"] = "pressure_weighting"
    surface_pressure: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Surface pressure at the bottom of the column, in hPa. Unset "
            "estimates it from the particles."
        ),
    )

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        """
        Return the particles with ``foot`` weighted by pressure.

        The receptor's ``altitude_ref`` is read from ``context``. Without a
        context the release heights are taken to be above ground.
        """
        altitude_ref = context.receptor.altitude_ref if context is not None else "agl"
        xpres, pwf = particle_pwf(particles, self.surface_pressure, altitude_ref)
        out = particles.copy()
        indx = out["indx"].to_numpy()
        out["xpres"] = xpres.reindex(indx).to_numpy()
        out["pwf"] = pwf.reindex(indx).to_numpy()
        out["foot"] = out["foot"] * out["pwf"] * len(pwf)
        return out


class FirstOrderLifetime(BaseModel):
    """
    Transform that decays each particle's ``foot`` by ``exp(-age / lifetime)``.

    ``age`` is the particle's travel time since release, from the ``time``
    column (minutes), and the lifetime is the species' e-folding lifetime.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["first_order_lifetime"] = "first_order_lifetime"
    lifetime_hours: float = Field(gt=0, description="E-folding lifetime, in hours.")

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` decayed by age."""
        if "time" not in particles.columns:
            raise ValueError(
                "Particle DataFrame has no 'time' column, required for "
                "first_order_lifetime."
            )
        age_hours = np.abs(particles["time"].to_numpy(dtype=float)) / 60.0
        out = particles.copy()
        out["foot"] = out["foot"] * np.exp(-age_hours / self.lifetime_hours)
        return out


BuiltinTransform = Annotated[
    AveragingKernel | PressureWeighting | FirstOrderLifetime,
    Field(discriminator="kind"),
]
_BUILTIN_ADAPTER: TypeAdapter[Any] = TypeAdapter(BuiltinTransform)


# -- loading and dumping ------------------------------------------------------------


def transform_kind(transform: Any) -> str:
    """Return the ``kind`` that declares a transform in a config."""
    kind = getattr(transform, "kind", None)
    if isinstance(kind, str):
        return kind
    cls = type(transform)
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_kind(kind: str) -> type:
    """Import and return the class named by a ``module.Class`` string."""
    module_name, _, attr = kind.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def load_transform(spec: Any) -> Any:
    """
    Return the transform for one config entry.

    ``spec`` is either a transform already (any object with ``apply``) or a
    mapping with a ``kind``. The ``kind`` is a built-in name or the import
    path of a pydantic model class, which is validated from the remaining
    keys. Only a pydantic model can be written back to ``config.yaml``.

    Raises
    ------
    ImportError
        If ``kind`` names a class that cannot be imported on this machine.
    """
    if hasattr(spec, "apply"):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"Transform entries must be mappings with a 'kind' or objects with "
            f"apply(); got {type(spec).__name__}."
        )
    kind = spec.get("kind")
    if not isinstance(kind, str):
        raise ValueError("Transform entries require a string 'kind'.")
    if "." not in kind:
        return _BUILTIN_ADAPTER.validate_python(spec)

    fields = {k: v for k, v in spec.items() if k != "kind"}
    try:
        cls = _import_kind(kind)
    except ImportError as exc:
        raise ImportError(
            f"Transform {kind!r} could not be imported ({exc}). Install the "
            "package that defines it on this machine."
        ) from exc
    if not callable(getattr(cls, "apply", None)):
        raise TypeError(f"{kind} does not define an apply() method.")
    if not hasattr(cls, "model_validate"):
        raise TypeError(
            f"{kind} is not a pydantic model. A transform named in a config "
            "must be one so its settings can be written back."
        )
    return cls.model_validate(fields)


def dump_transform(transform: Any) -> dict[str, Any]:
    """
    Return the config mapping for one transform, the inverse of :func:`load_transform`.

    A mapping is returned as it is. A footprint read from a file keeps a
    transform it cannot import that way.
    """
    if isinstance(transform, dict):
        return dict(transform)
    if hasattr(transform, "model_dump"):
        data = dict(transform.model_dump(mode="json", exclude_none=True))
    else:
        raise TypeError(
            f"{type(transform).__qualname__} cannot be written to config: transforms "
            "declared in config.yaml must be pydantic models."
        )
    data.pop("kind", None)
    return {"kind": transform_kind(transform), **data}


def apply_transforms(
    particles: pd.DataFrame,
    transforms: list[Any],
    context: TransformContext | None = None,
) -> pd.DataFrame:
    """Apply ``transforms`` in order. Returns ``particles`` itself when there are none."""
    for transform in transforms:
        particles = transform.apply(particles, context)
    return particles


__all__ = [
    "KERNEL_TABLE_COLUMNS",
    "AveragingKernel",
    "BuiltinTransform",
    "FirstOrderLifetime",
    "ParticleTransform",
    "PressureWeighting",
    "TransformContext",
    "ak_weights",
    "apply_transforms",
    "averaging_kernel_table",
    "dump_transform",
    "load_transform",
    "particle_pwf",
    "release_coordinate",
    "transform_kind",
]
