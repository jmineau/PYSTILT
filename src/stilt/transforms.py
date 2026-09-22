"""
Particle transforms: re-weight each particle's ``foot`` before rasterization.

A transform is any object with ``apply(particles, context) -> DataFrame``.
The built-in ones are pydantic models whose fields are their ``config.yaml``
keys, so the same object describes the transform and performs it::

    footprints:
      column:
        grid: slv
        transforms:
          - kind: averaging_kernel
            levels: [0, 500, 1000]
            values: [1.0, 0.9, 0.7]
          - kind: pressure_weighting

An averaging kernel that differs per receptor (every satellite sounding has
its own) comes from a table in the project instead::

          - kind: averaging_kernel
            table: kernels.parquet

A ``kind`` containing a dot is an import path to a user-defined transform
class (see the *Custom transforms* guide). Transforms run once, in list order,
on the unweighted particle table, and return a new table — they never mutate
their input.

The science functions (:func:`particle_pwf`, :func:`ak_weights`) are the
X-STILT column weighting port and are public so user transforms can build on
them.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from stilt.receptors import Receptor
    from stilt.store import Store


# -- interface ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransformContext:
    """
    What a transform may know about the footprint it is applied for.

    ``receptor`` is the receptor the particles were released from (its ``id``
    keys per-receptor inputs such as an averaging-kernel table),
    ``footprint_name`` the footprint being generated, ``is_error`` whether
    these are the error trajectories, and ``store`` the project store, so
    files named relative to the project root can be found wherever the
    footprint is generated.
    """

    receptor: Receptor
    footprint_name: str = ""
    is_error: bool = False
    store: Store | None = None


@runtime_checkable
class ParticleTransform(Protocol):
    """Anything with ``apply(particles, context) -> DataFrame``."""

    def apply(self, particles: pd.DataFrame, context: TransformContext) -> pd.DataFrame:
        """Return a new particle table; never mutate *particles*."""
        ...


# -- science ----------------------------------------------------------------------

HOURS_PER: dict[str, float] = {
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "m": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "h": 1.0,
    "hr": 1.0,
    "hours": 1.0,
    "d": 24.0,
    "days": 24.0,
}


def release_coordinate(particles: pd.DataFrame, coordinate: str) -> pd.Series:
    """
    Return each particle's *release* value of ``coordinate``, indexed by ``indx``.

    Release coordinates such as ``xhgt`` are constant along a trajectory, but
    HYSPLIT diagnostics such as ``pres`` are written at every time step. The
    row nearest the receptor time (smallest ``|time|`` when ``time`` is in
    minutes; otherwise the first row) defines the release value.
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
    particles: pd.DataFrame, surface_pressure: float | None = None
) -> tuple[pd.Series, pd.Series]:
    """
    Derive each particle's release pressure and pressure weight.

    Follows X-STILT's ``get.wgt.*.func``: fit a hypsometric curve
    ``ln p = b + a·z`` to the particles' first-step heights and pressures
    (this smooths the one-time-step offset from the true release state and
    yields a surface pressure when none is supplied), evaluate it at each
    particle's release height, and turn the spacing between neighbouring
    release pressures into the air mass each particle represents.

    HYSPLIT spreads column particles evenly over height, each one randomized
    within its own ``1/numpar`` slab, so a particle stands for the slab
    centred on it: the cell edges sit midway between adjacent release
    pressures, the surface closes the bottom, and the topmost cell mirrors its
    lower half-width. (X-STILT instead gives each particle the layer *below*
    it, which shifts every weight down by half a cell and leaves the lowest
    particle with almost none.)

    Returns ``(xpres, pwf)`` indexed by ``indx``. ``pwf`` sums to the fraction
    of the atmosphere's mass the column covers, ``(p_sfc - p_top) / p_sfc``;
    the rest lies above the column top, where surface fluxes cannot reach the
    receptor within the back-trajectory.
    """
    for col in ("pres", "zagl"):
        if col not in particles.columns:
            raise ValueError(
                f"Pressure weighting requires the {col!r} particle variable; "
                "include it in STILTParams.varsiwant."
            )
    pres = release_coordinate(particles, "pres")
    zagl = release_coordinate(particles, "zagl")
    z_release = (
        release_coordinate(particles, "xhgt") if "xhgt" in particles.columns else zagl
    )

    if zagl.nunique() < 2:
        raise ValueError(
            "Pressure weighting needs particles released over a range of heights "
            "(a ColumnReceptor); all particles share one release height."
        )
    a, b = np.polyfit(zagl.to_numpy(), np.log(pres.to_numpy()), 1)
    if a >= 0:
        raise ValueError(
            "Could not fit a pressure profile to the particles (pressure does "
            "not decrease with height)."
        )
    p_sfc = (
        float(surface_pressure) if surface_pressure is not None else float(np.exp(b))
    )

    xpres = pd.Series(p_sfc * np.exp(a * z_release.to_numpy()), index=z_release.index)
    ordered = xpres.sort_values(ascending=False)  # surface upward
    levels = ordered.to_numpy()

    mids = (levels[:-1] + levels[1:]) / 2.0
    lower_edges = np.concatenate(([p_sfc], mids))
    upper_edges = np.concatenate((mids, [2 * levels[-1] - mids[-1]]))
    pwf = pd.Series((lower_edges - upper_edges) / p_sfc, index=ordered.index)
    return xpres, pwf


def ak_weights(
    particles: pd.DataFrame,
    levels: list[float],
    values: list[float],
    coordinate: str = "xhgt",
) -> np.ndarray:
    """
    Interpolate an averaging kernel to each row's particle release coordinate.

    One value per particle, taken at its release row and broadcast along its
    whole trajectory. Outside ``levels`` the kernel is held at its end values.
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
    Build the per-receptor averaging-kernel table for :class:`AveragingKernel`.

    One kernel per receptor, in long form: a ``receptor`` column holding the
    receptor id, and one ``level`` / ``value`` row per kernel point. Write it
    into the project with ``.to_parquet()`` or ``.to_csv(index=False)`` and
    name the file as ``table:`` in the footprint's ``averaging_kernel``
    transform.

    ``receptors`` are :class:`~stilt.Receptor` objects or their ids.
    ``values`` holds one array per receptor. ``levels`` is either one array
    per receptor (satellite retrievals, whose pressure grids differ per
    sounding) or a single array shared by all of them (a fixed altitude grid).
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


def _read_kernel_table(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


@functools.lru_cache(maxsize=8)
def _cached_kernel_table(
    path: str, _mtime: float | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _read_kernel_table(path)


def _load_kernel_table(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a kernel table once per file version (local files are keyed by mtime)."""
    try:
        mtime: float | None = os.stat(path).st_mtime
    except OSError:
        mtime = None
    return _cached_kernel_table(path, mtime)


class AveragingKernel(BaseModel):
    """
    Weight each particle's ``foot`` by an averaging kernel at its release coordinate.

    Give the kernel inline as ``levels`` and ``values``, or name a ``table``
    holding one kernel per receptor (see :func:`averaging_kernel_table`); the
    receptor's row is picked by ``context.receptor.id`` when the footprint is
    generated. A relative ``table`` path is resolved against the project
    root, so it works inside ``stilt run`` and on Slurm and Kubernetes
    workers. A receptor missing from the table is an error.

    ``levels`` are release heights AGL in metres by default; set
    ``coordinate: pres`` for a kernel on pressure levels (hPa). Fold any
    instrument-specific factor (for example TCCON's wet-air scaling) into the
    values. Adds an ``ak_weight`` column.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["averaging_kernel"] = "averaging_kernel"
    levels: list[float] | None = Field(
        default=None, description="Vertical coordinates of the averaging-kernel values."
    )
    values: list[float] | None = Field(
        default=None, description="Normalized averaging-kernel values at ``levels``."
    )
    table: str | None = Field(
        default=None,
        description=(
            "Per-receptor kernel table (.parquet or .csv with receptor, level, "
            "value columns), relative to the project root."
        ),
    )
    coordinate: str = Field(
        default="xhgt",
        description="Particle column the kernel is defined on ('xhgt' or 'pres').",
    )

    @model_validator(mode="after")
    def _inline_or_table(self) -> Self:
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
        """Return ``(levels, values)`` for the receptor in *context*."""
        if self.table is None:
            assert self.levels is not None and self.values is not None
            return self.levels, self.values
        if context is None:
            raise ValueError(
                "averaging_kernel with a table needs a TransformContext to know "
                "which receptor to look up."
            )
        path = self.table
        if context.store is not None and not Path(path).is_absolute():
            path = str(context.store.local_path(path))
        kernels = _load_kernel_table(path)
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
        levels, values = self.kernel(context)
        weights = ak_weights(particles, levels, values, self.coordinate)
        out = particles.copy()
        out["ak_weight"] = weights
        out["foot"] = out["foot"] * weights
        return out


class PressureWeighting(BaseModel):
    """
    Weight each particle by the fraction of the column's air mass it represents.

    The pressure weighting function is derived from the particles' own
    first-step heights and pressures (see :func:`particle_pwf`); nothing needs
    to be supplied. Because :meth:`stilt.Footprint.calculate` divides by the
    particle count, the weights are multiplied by ``n_particles`` so the
    weighted footprint does not scale with ``numpar``. Adds ``xpres`` (release
    pressure, hPa) and ``pwf`` columns. Requires ``pres`` and ``zagl`` in
    ``varsiwant`` (both are defaults).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pressure_weighting"] = "pressure_weighting"
    surface_pressure: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Surface pressure (hPa) closing the bottom of the column. "
            "Estimated from the particles when omitted."
        ),
    )

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        xpres, pwf = particle_pwf(particles, self.surface_pressure)
        out = particles.copy()
        indx = out["indx"].to_numpy()
        out["xpres"] = xpres.reindex(indx).to_numpy()
        out["pwf"] = pwf.reindex(indx).to_numpy()
        out["foot"] = out["foot"] * out["pwf"] * len(pwf)
        return out


class FirstOrderLifetime(BaseModel):
    """
    Decay each particle's ``foot`` by ``exp(-age / lifetime)``.

    ``age`` is the particle's transport time from ``time_column`` (minutes by
    default) and ``lifetime_hours`` the species e-folding lifetime.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["first_order_lifetime"] = "first_order_lifetime"
    lifetime_hours: float = Field(gt=0, description="E-folding lifetime in hours.")
    time_column: str = Field(
        default="time", description="Particle column holding the transport age."
    )
    time_unit: Literal[
        "s", "sec", "seconds", "m", "min", "minutes", "h", "hr", "hours", "d", "days"
    ] = Field(default="min", description="Unit of ``time_column``.")

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        if self.time_column not in particles.columns:
            raise ValueError(
                f"Particle DataFrame has no column {self.time_column!r} required "
                "for first_order_lifetime."
            )
        ages = np.abs(particles[self.time_column].to_numpy(dtype=float))
        tau = self.lifetime_hours / HOURS_PER[self.time_unit]
        out = particles.copy()
        out["foot"] = out["foot"] * np.exp(-ages / tau)
        return out


BuiltinTransform = Annotated[
    AveragingKernel | PressureWeighting | FirstOrderLifetime,
    Field(discriminator="kind"),
]
_BUILTIN_ADAPTER: TypeAdapter[Any] = TypeAdapter(BuiltinTransform)


class UnresolvedTransform(BaseModel):
    """
    A transform whose class could not be imported.

    Produced when a stored config names a ``kind`` such as
    ``mypkg.transforms.MyKernel`` and that module is not importable here, so
    the config (and any footprint carrying it) can still be read. Applying it
    raises with the original import error.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    kind: str
    reason: str = Field(default="", exclude=True)

    def apply(
        self, particles: pd.DataFrame, context: TransformContext | None = None
    ) -> pd.DataFrame:
        raise ImportError(
            f"Transform {self.kind!r} could not be imported: {self.reason}. "
            "Install the package that defines it on this machine."
        )


# -- loading and dumping ------------------------------------------------------------


def transform_kind(transform: Any) -> str:
    """Return the ``kind`` a transform is declared by in config."""
    kind = getattr(transform, "kind", None)
    if isinstance(kind, str):
        return kind
    cls = type(transform)
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_kind(kind: str) -> type:
    module_name, _, attr = kind.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def load_transform(spec: Any) -> Any:
    """
    Return a transform for one config entry.

    *spec* may already be a transform (anything with ``apply``), or a mapping
    with a ``kind``: a built-in name, or a dotted import path to a user class
    that is constructed from the remaining keys (``cls.model_validate`` for a
    pydantic model, ``cls(**keys)`` otherwise). An unimportable path yields an
    :class:`UnresolvedTransform` rather than failing the load.
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
        return UnresolvedTransform(kind=kind, reason=str(exc), **fields)
    if not callable(getattr(cls, "apply", None)):
        raise TypeError(f"{kind} does not define an apply() method.")
    if hasattr(cls, "model_validate"):
        return cls.model_validate(fields)
    return cls(**fields)


def dump_transform(transform: Any) -> dict[str, Any]:
    """Return the config mapping for one transform (inverse of :func:`load_transform`)."""
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
    context: TransformContext,
) -> pd.DataFrame:
    """Apply *transforms* in order; returns *particles* itself when there are none."""
    for transform in transforms:
        particles = transform.apply(particles, context)
    return particles


__all__ = [
    "HOURS_PER",
    "KERNEL_TABLE_COLUMNS",
    "AveragingKernel",
    "BuiltinTransform",
    "FirstOrderLifetime",
    "ParticleTransform",
    "PressureWeighting",
    "TransformContext",
    "UnresolvedTransform",
    "ak_weights",
    "apply_transforms",
    "averaging_kernel_table",
    "dump_transform",
    "load_transform",
    "particle_pwf",
    "release_coordinate",
    "transform_kind",
]
