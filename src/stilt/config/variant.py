"""
Variants as declared: each merged with the defaults and checked.

A project runs every receptor under every variant. ``config.yaml`` declares
variants as overrides of its defaults. :func:`expand_variants` merges each
with the defaults, expands ``realizations``, and splits the result into a
transport config and a footprint config: one :class:`VariantConfig` per
simulation name. It reads no file, so a config loads on any machine. The
grid of a footprint given by a geometry, and the transport model build, are
found by :func:`stilt.variants.resolve`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import TransportParams

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class VariantConfig(BaseModel):
    """
    One variant as declared, merged with the defaults.

    ``name`` is the name its simulations run under and ``group`` the name
    declared in ``config.yaml``. They differ only for realizations:
    ``hrrr-err`` with ``realizations: 3`` gives ``hrrr-err-0`` to
    ``hrrr-err-2``. A footprint given by a geometry has no grid yet;
    :func:`stilt.variants.resolve` derives it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Variant name its simulations run under.")
    group: str = Field(description="Variant name as declared in ``config.yaml``.")
    met: str = Field(description="Name of the met this variant runs with.")
    transport: TransportParams = Field(description="Transport config.")
    footprint: FootprintConfig | None = Field(
        None,
        description="Footprint config, or ``None`` for particles only (``grid: null``).",
    )
    realization: int | None = Field(
        None,
        description="Realization number within an ensemble, or ``None`` for a single run.",
    )


_TRANSPORT_FIELDS = frozenset(TransportParams.model_fields)
_FOOTPRINT_FIELDS = frozenset(FootprintConfig.model_fields)


def expand_variants(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: Mapping[str, MetConfig],
) -> dict[str, VariantConfig]:
    """
    Return one :class:`VariantConfig` per simulation name, in declared order.

    Parameters
    ----------
    declared : dict
        ``{name: overrides}`` as written in ``config.yaml``. Besides
        overrides of the defaults, each may set ``met`` and ``realizations``.
    defaults : dict
        The top-level transport and footprint fields.
    mets : mapping of str to MetConfig
        The config's mets. A variant without ``met`` uses the met with its
        own name, or the only met.

    Raises
    ------
    ValueError
        For a bad name, an unknown met or setting, a bad ``realizations``,
        or footprint settings without a grid.
    """
    variants: dict[str, VariantConfig] = {}
    for group, spec in declared.items():
        if not VARIANT_NAME_RE.fullmatch(group):
            raise ValueError(
                f"Variant name {group!r} must match {VARIANT_NAME_RE.pattern}"
            )
        spec = dict(spec or {})
        if "from" in spec:
            raise ValueError(
                f"Variant {group!r} uses 'from:', which is no longer needed: a "
                "variant with the same transport settings as another shares its "
                "particles. Give the variant the transport overrides of "
                f"{spec['from']!r} (if any) and its own footprint settings."
            )
        met = _met_name(group, spec, mets)
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"Variant {group!r}: realizations must be >= 1")
        transport_fields, footprint_fields = _split(
            group, _override(group, defaults, spec)
        )
        transport = TransportParams(**transport_fields)
        footprint = _footprint(group, footprint_fields)
        if realizations is None:
            variants[group] = VariantConfig(
                name=group,
                group=group,
                met=met,
                transport=transport,
                footprint=footprint,
                realization=None,
            )
            continue
        _check_realizations(group, transport, realizations, declared)
        seed = transport.seed
        for k in range(realizations):
            # Realization k runs with seed + k, so realization 0 uses the
            # configured seed, as STILT-R's single error run does. A group of
            # one is still <group>-0, so raising realizations only adds runs.
            fields = {**transport_fields, "seed": None if seed is None else seed + k}
            variants[f"{group}-{k}"] = VariantConfig(
                name=f"{group}-{k}",
                group=group,
                met=met,
                transport=TransportParams(**fields),
                footprint=footprint,
                realization=k,
            )
    return variants


def _met_name(group: str, spec: dict[str, Any], mets: Mapping[str, MetConfig]) -> str:
    """Return the met a variant runs with, taking ``met`` out of *spec*."""
    met = spec.pop("met", None)
    if met is None:
        if group in mets:
            met = group
        elif len(mets) == 1:
            met = next(iter(mets))
        else:
            raise ValueError(
                f"Variant {group!r} must name its met (one of {sorted(mets)}) "
                "or be named after one"
            )
    if met not in mets:
        raise ValueError(f"Variant {group!r} names unknown met {met!r}")
    return met


def _override(group: str, base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a variant's overrides to ``base``.

    A ``grid`` mapping updates the base grid field by field, so a variant can
    change only the resolution, and ``grid: null`` removes the footprint. A
    variant that sets its own ``geometry`` drops the inherited ``grid``
    unless it sets ``grid`` too, so the grid is derived from its geometry.

    Raises
    ------
    ValueError
        If a variant changes part of a grid that is derived from the
        default ``geometry``. That grid is not known until the geometry is
        built.
    """
    merged = {**base, **spec}
    if isinstance(spec.get("grid"), dict):
        if isinstance(base.get("grid"), dict):
            merged["grid"] = {**base["grid"], **spec["grid"]}
        elif base.get("geometry") is not None and "geometry" not in spec:
            raise ValueError(
                f"Variant {group!r} changes part of a grid that is derived from "
                "the default geometry. Give a full grid, or set cells_per_target "
                "to change its resolution."
            )
    if spec.get("geometry") is not None and "grid" not in spec:
        merged.pop("grid", None)
    return merged


def _split(group: str, merged: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the flat settings of a variant into its transport and footprint parts."""
    unknown = set(merged) - _TRANSPORT_FIELDS - _FOOTPRINT_FIELDS
    if unknown:
        raise ValueError(f"Variant {group!r} has unknown settings {sorted(unknown)}")
    transport = {k: v for k, v in merged.items() if k in _TRANSPORT_FIELDS}
    footprint = {k: v for k, v in merged.items() if k in _FOOTPRINT_FIELDS}
    return transport, footprint


def _footprint(name: str, fields: dict[str, Any]) -> FootprintConfig | None:
    """
    Return the footprint settings, or ``None`` when they give no grid or geometry.

    The settings are not resolved here, so the geometry is not read.

    Raises
    ------
    ValueError
        If footprint settings other than the grid are set without a grid,
        since they would otherwise be dropped without a word.
    """
    config = FootprintConfig(**fields)
    if config.grid is not None or config.geometry is not None:
        return config
    given = config.model_dump()
    default = FootprintConfig.model_validate({}).model_dump()
    stray = sorted(field for field in given if given[field] != default[field])
    if stray:
        raise ValueError(
            f"Variant {name!r} sets footprint settings ({', '.join(stray)}) "
            "but no grid. Add a grid or geometry, or remove them."
        )
    return None


def _check_realizations(
    group: str,
    params: TransportParams,
    realizations: int,
    declared: dict[str, dict[str, Any]],
) -> None:
    """Raise unless *realizations* runs of *params* would differ and their names are free."""
    if realizations > 1 and not (
        params.krand == 4 or (params.krand == 2 and params.seed is not None)
    ):
        raise ValueError(
            f"Variant {group!r}: realizations={realizations} requires krand=4 "
            f"or krand=2 with a seed (got krand={params.krand}, seed={params.seed}): "
            "under krand=4 HYSPLIT seeds each run from the clock; under krand=2 "
            "PYSTILT gives each realization its own seed. Any other mode would "
            "repeat the same perturbation."
        )
    for k in range(realizations):
        name = f"{group}-{k}"
        if name in declared:
            raise ValueError(
                f"Variant {name!r} collides with realization {k} of {group!r}"
            )


__all__ = [
    "VARIANT_NAME_RE",
    "VariantConfig",
    "expand_variants",
]
