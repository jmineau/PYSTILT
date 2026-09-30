"""
Variants: the complete settings a receptor is run under.

A project runs every receptor under every variant. A variant names its met
and holds a full set of transport and footprint settings. ``config.yaml``
declares variants as overrides of its defaults, and
:meth:`ModelConfig.resolve_variants` turns them into one
:class:`VariantConfig` per simulation name. Variants whose transport
settings are equal share one run of HYSPLIT per receptor and differ only in
the footprint made from its particles.
"""

from __future__ import annotations

import re
from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator

from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import STILTParams
from .transport import UNRECORDED_FIELDS, TransportSettings

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class VariantConfig(STILTParams, FootprintConfig):
    """
    The full settings of one variant: its met, transport, and footprint.

    Built by :meth:`~stilt.config.ModelConfig.resolve_variants`. ``name`` is
    the name its simulations run under and ``group`` the name declared in
    ``config.yaml``. They differ only for realizations: ``hrrr-err`` with
    ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Variant name its simulations run under, also their directory name."
    )
    group: str = Field(description="Variant name as declared in ``config.yaml``.")
    met: str = Field(description="Name of the meteorology this variant runs with.")
    realization: int | None = Field(
        None,
        description="Realization number within ``group``. ``None`` for a single run.",
    )

    @model_validator(mode="after")
    def _validate_name(self) -> Self:
        """Require names that are safe as directory names."""
        for value in (self.name, self.group):
            if not VARIANT_NAME_RE.fullmatch(value):
                raise ValueError(
                    f"Variant name {value!r} must match {VARIANT_NAME_RE.pattern}"
                )
        return self

    def stilt_params(self) -> STILTParams:
        """Return the transport parameters alone, as stored with a trajectory."""
        return STILTParams(**self.model_dump(include=set(STILTParams.model_fields)))

    def transport_settings(self, met: MetConfig) -> TransportSettings:
        """
        Return the settings that identify this variant's run.

        Its transport fields, the content of *met* (its met entry in the
        config), and the engine. Variants that differ only in footprint
        fields give equal settings, and so share one run.
        """
        return TransportSettings.build(
            self.stilt_params(), met, realization=self.realization
        )


def expand_variants(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: list[str],
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
    mets : list of str
        Met names in the config. A variant without ``met`` uses the met with
        its own name, or the only met.

    Returns
    -------
    dict
        Variants by simulation name, in declared order.
    """
    for group in declared:
        if not VARIANT_NAME_RE.fullmatch(group):
            raise ValueError(
                f"Variant name {group!r} must match {VARIANT_NAME_RE.pattern}"
            )

    runs: dict[str, list[VariantConfig]] = {}
    for group, spec in declared.items():
        spec = dict(spec or {})
        if "from" in spec:
            raise ValueError(
                f"Variant {group!r} uses 'from:', which is no longer needed: a "
                "variant with the same transport settings as another shares its "
                "particles. Give the variant the transport overrides of "
                f"{spec['from']!r} (if any) and its own footprint settings."
            )
        merged, realizations = _merge_transport(group, spec, defaults, mets)
        runs[group] = _expand_realizations(group, merged, realizations, declared)

    return {v.name: v for group in declared for v in runs[group]}


def _merge_transport(
    group: str, spec: dict[str, Any], defaults: dict[str, Any], mets: list[str]
) -> tuple[dict[str, Any], int | None]:
    """
    Merge a variant that runs HYSPLIT onto the defaults.

    Returns the merged parameters and the declared realization count, which
    is ``None`` when the variant does not declare ``realizations``.
    """
    met = spec.pop("met", None)
    if met is None:
        if group in mets:
            met = group
        elif len(mets) == 1:
            met = mets[0]
        else:
            raise ValueError(
                f"Variant {group!r} must name its met (one of {sorted(mets)}) "
                "or be named after one"
            )
    if met not in mets:
        raise ValueError(f"Variant {group!r} names unknown met {met!r}")
    realizations = spec.pop("realizations", None)
    if realizations is not None:
        realizations = int(realizations)
        if realizations < 1:
            raise ValueError(f"Variant {group!r}: realizations must be >= 1")
    return {**_override(defaults, spec), "met": met}, realizations


def _override(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a variant's overrides to ``base``.

    A ``grid`` mapping updates the base grid field by field, so a variant can
    change only the resolution, and ``grid: null`` removes the footprint. A
    variant that sets its own ``geometry`` drops the inherited
    ``geometry_hash``, and the inherited ``grid`` unless it sets ``grid`` too,
    so both are derived from its geometry.
    """
    merged = {**base, **spec}
    if isinstance(spec.get("grid"), dict) and isinstance(base.get("grid"), dict):
        merged["grid"] = {**base["grid"], **spec["grid"]}
    if "geometry" in spec:
        merged.pop("geometry_hash", None)
        if spec["geometry"] is not None and "grid" not in spec:
            merged.pop("grid", None)
    return merged


def _expand_realizations(
    group: str,
    merged: dict[str, Any],
    realizations: int | None,
    declared: dict[str, dict[str, Any]],
) -> list[VariantConfig]:
    """
    Return the variant, or its realizations when ``realizations`` is declared.

    Realization ``k`` is named ``<group>-k`` and runs with ``seed + k``. A
    group of one is still ``<group>-0``, so raising ``realizations`` later
    only adds simulations.
    """
    base = VariantConfig(name=group, group=group, **merged)
    if realizations is None:
        return [base]
    if realizations > 1 and not (
        base.krand == 4 or (base.krand == 2 and base.seed is not None)
    ):
        raise ValueError(
            f"Variant {group!r}: realizations={realizations} requires krand=4 "
            f"or krand=2 with a seed (got krand={base.krand}, seed={base.seed}): "
            "under krand=4 HYSPLIT seeds each run from the clock; under krand=2 "
            "PYSTILT gives each realization its own seed. Any other mode would "
            "repeat the same perturbation."
        )
    out = []
    for k in range(realizations):
        name = f"{group}-{k}"
        if name in declared:
            raise ValueError(
                f"Variant {name!r} collides with realization {k} of {group!r}"
            )
        out.append(
            VariantConfig(
                name=name,
                group=group,
                realization=k,
                **{**merged, "seed": base.realization_seed(k)},
            )
        )
    return out


__all__ = [
    "UNRECORDED_FIELDS",
    "VARIANT_NAME_RE",
    "VariantConfig",
    "expand_variants",
]
