"""
Variants: the complete settings a receptor is run under.

A project runs every receptor under every variant. ``config.yaml`` declares
variants as overrides of its defaults, and
:meth:`ProjectConfig.resolve_variants` turns them into one
:class:`VariantConfig` per simulation name, each split into the transport
settings that decide its particles and the footprint settings applied to
them. Variants whose transport settings are equal share one run of HYSPLIT
per receptor and differ only in the footprint made from its particles.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import TransportParams
from .transport import UNRECORDED_FIELDS, TransportSettings

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class VariantConfig(BaseModel):
    """
    One variant: its met, its transport settings, and its footprint settings.

    Built by :meth:`~stilt.config.ProjectConfig.resolve_variants`. ``name`` is
    the name its simulations run under and ``group`` the name declared in
    ``config.yaml``. They differ only for realizations: ``hrrr-err`` with
    ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Variant name its simulations run under.")
    group: str = Field(description="Variant name as declared in ``config.yaml``.")
    met: str = Field(description="Name of the meteorology this variant runs with.")
    transport: TransportSettings = Field(
        description=(
            "Everything that decides the particles. Its hash names the run in "
            "the output directory; variants with equal transport settings "
            "share one run."
        )
    )
    footprint: FootprintConfig | None = Field(
        None,
        description="Footprint settings, or ``None`` for particles only (``grid: null``).",
    )


_TRANSPORT_FIELDS = frozenset(TransportParams.model_fields)
_FOOTPRINT_FIELDS = frozenset(FootprintConfig.model_fields)


@dataclass(frozen=True)
class _Plan:
    """One declared variant, checked: its met, transport parameters, footprint, and realizations."""

    group: str
    met_name: str
    met: MetConfig
    transport_fields: dict[str, Any]
    params: TransportParams
    footprint: FootprintConfig | None
    realizations: int | None


def check_variants(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: Mapping[str, MetConfig],
) -> None:
    """
    Raise if a variant declaration is invalid, without building any variant.

    It checks names, mets, settings, and realizations, but does not look up
    the HYSPLIT build, so a config can be loaded on a machine where
    ``exe_dir`` is not reachable. Takes the arguments of
    :func:`expand_variants`.
    """
    _plan(declared, defaults, mets)


def expand_variants(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: Mapping[str, MetConfig],
) -> dict[str, VariantConfig]:
    """
    Return one :class:`VariantConfig` per simulation name.

    Parameters
    ----------
    declared : dict
        ``{name: overrides}`` as written in ``config.yaml``. Besides
        parameter overrides, each may set ``met`` and ``realizations``.
    defaults : dict
        The top-level transport and footprint parameters.
    mets : mapping of str to MetConfig
        The config's meteorology entries. A variant without ``met`` uses the
        met with its own name, or the only met.

    Returns
    -------
    dict
        Variants by simulation name, in declared order. Their footprint
        settings are not resolved, so a footprint given by ``geometry`` has
        no grid yet. :meth:`~stilt.config.ProjectConfig.resolve_variants`
        resolves them.
    """
    return {v.name: v for plan in _plan(declared, defaults, mets) for v in _build(plan)}


def _plan(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: Mapping[str, MetConfig],
) -> list[_Plan]:
    """Check every declared variant and return what each one builds from."""
    plans = []
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
        met_name = _met_name(group, spec, mets)
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"Variant {group!r}: realizations must be >= 1")
        merged = _override(group, defaults, spec)
        transport_fields, footprint_fields = _split(group, merged)
        params = TransportParams(**transport_fields)
        if realizations is not None:
            _check_realizations(group, params, realizations, declared)
        plans.append(
            _Plan(
                group=group,
                met_name=met_name,
                met=mets[met_name],
                transport_fields=transport_fields,
                params=params,
                footprint=_footprint(group, footprint_fields),
                realizations=realizations,
            )
        )
    return plans


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
    variant that sets its own ``geometry`` drops the inherited
    ``geometry_hash``, and the inherited ``grid`` unless it sets ``grid`` too,
    so both are derived from its geometry.

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
    if "geometry" in spec:
        merged.pop("geometry_hash", None)
        if spec["geometry"] is not None and "grid" not in spec:
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


def _build(plan: _Plan) -> list[VariantConfig]:
    """
    Return the variant, or its realizations when ``realizations`` is declared.

    Realization ``k`` is named ``<group>-k`` and runs with ``seed + k``, so
    realization 0 uses the configured seed, as STILT-R's single error run
    does. A group of one is still ``<group>-0``, so raising ``realizations``
    later only adds simulations.
    """

    def variant(
        name: str, realization: int | None, params: TransportParams
    ) -> VariantConfig:
        return VariantConfig(
            name=name,
            group=plan.group,
            met=plan.met_name,
            transport=TransportSettings.build(
                params, plan.met, realization=realization
            ),
            footprint=plan.footprint,
        )

    if plan.realizations is None:
        return [variant(plan.group, None, plan.params)]
    seed = plan.params.seed
    return [
        variant(
            f"{plan.group}-{k}",
            k,
            TransportParams(
                **{**plan.transport_fields, "seed": None if seed is None else seed + k}
            ),
        )
        for k in range(plan.realizations)
    ]


__all__ = [
    "UNRECORDED_FIELDS",
    "VARIANT_NAME_RE",
    "VariantConfig",
    "check_variants",
    "expand_variants",
]
