"""
A variant: one complete configuration a receptor is run under.

A project is receptors crossed with variants. Each variant names its met,
carries a full :class:`~stilt.config.STILTParams` and the footprint settings,
and is one HYSPLIT call per receptor. ``config.yaml`` declares variants as
overrides of its flat defaults; :meth:`ModelConfig.resolve_variants` turns
them into :class:`VariantConfig` objects, one per simulation name.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from .footprint import FootprintParams
from .params import STILTParams

#: Variant (and met) names: lowercase, digits and hyphens; they become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: Keys a variant declaration may carry besides parameter overrides.
VARIANT_KEYS = frozenset({"met", "from", "realizations"})

#: Fields that change no output, so the record does not compare them.
UNRECORDED_FIELDS = frozenset({"timeout", "rm_dat", "exe_dir"})


class VariantConfig(STILTParams, FootprintParams):
    """
    One resolved variant: met, transport params, and footprint settings.

    Built by :meth:`~stilt.config.ModelConfig.resolve_variants`; not read
    from YAML directly. ``name`` is the simulation-level name (a realization
    group ``hrrr-err`` with ``realizations: 3`` yields ``hrrr-err-0`` to
    ``hrrr-err-2``), ``group`` the name as declared.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Simulation-level variant name (directory name).")
    group: str = Field(description="Variant name as declared in config.yaml.")
    met: str = Field(description="Name of the met stream this variant runs with.")
    realization: int | None = Field(
        None,
        description="Realization index within ``group``; ``None`` for a single run.",
    )
    derived_from: str | None = Field(
        None,
        description=(
            "Variant whose trajectory this one rasterizes instead of running "
            "HYSPLIT itself (``from:`` in config.yaml). Only footprint fields differ."
        ),
    )

    @model_validator(mode="after")
    def _validate_name(self) -> Self:
        """Names become directory names, so keep them plain."""
        for value in (self.name, self.group):
            if not VARIANT_NAME_RE.fullmatch(value):
                raise ValueError(
                    f"Variant name {value!r} must match {VARIANT_NAME_RE.pattern}"
                )
        return self

    @property
    def is_derived(self) -> bool:
        """Whether this variant reuses another variant's trajectory."""
        return self.derived_from is not None

    def stilt_params(self) -> STILTParams:
        """The transport parameters alone, as stored with a trajectory."""
        return STILTParams(**self.model_dump(include=set(STILTParams.model_fields)))

    def record(self) -> dict[str, Any]:
        """This variant as stored in the project's record (a full JSON dump)."""
        return self.model_dump(mode="json")

    def differences(self, recorded: dict[str, Any]) -> list[str]:
        """
        Names of the fields on which this variant differs from *recorded*.

        *recorded* is an entry of the project's record (:meth:`record`).
        Fields that change no output (:data:`UNRECORDED_FIELDS`) are skipped.
        """
        mine = self.record()
        return sorted(
            k for k in mine if k not in UNRECORDED_FIELDS and mine[k] != recorded.get(k)
        )


def expand_variants(
    declared: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    mets: list[str],
) -> dict[str, VariantConfig]:
    """
    Resolve declared variants into one :class:`VariantConfig` per simulation name.

    Parameters
    ----------
    declared
        ``{name: overrides}`` as written in ``config.yaml``. Each may carry
        ``met``, ``realizations`` and ``from`` besides parameter overrides.
    defaults
        The flat top-level parameters (transport and footprint fields).
    mets
        Configured met names; a variant's ``met`` defaults to the only one.
    """
    for group in declared:
        if not VARIANT_NAME_RE.fullmatch(group):
            raise ValueError(
                f"Variant name {group!r} must match {VARIANT_NAME_RE.pattern}"
            )

    # Transport variants first, so a derived one can sit anywhere in the file.
    runs: dict[str, list[VariantConfig]] = {}
    merged_by_group: dict[str, dict[str, Any]] = {}
    for group, spec in declared.items():
        spec = dict(spec or {})
        if "from" in spec:
            continue
        merged, realizations = _merge_transport(group, spec, defaults, mets)
        merged_by_group[group] = merged
        runs[group] = _expand_realizations(group, merged, realizations, declared)

    for group, spec in declared.items():
        spec = dict(spec or {})
        parent = spec.pop("from", None)
        if parent is None:
            continue
        _check_derived(group, parent, spec, declared)
        merged = _override(merged_by_group[parent], spec)
        runs[group] = [
            VariantConfig(name=group, group=group, derived_from=parent, **merged)
        ]

    return {v.name: v for group in declared for v in runs[group]}


def _merge_transport(
    group: str, spec: dict[str, Any], defaults: dict[str, Any], mets: list[str]
) -> tuple[dict[str, Any], int]:
    """Merge a transport variant onto the defaults; return it and its realization count."""
    met = spec.pop("met", None)
    if met is None:
        if len(mets) != 1:
            raise ValueError(
                f"Variant {group!r} must name its met (one of {sorted(mets)})"
            )
        met = mets[0]
    if met not in mets:
        raise ValueError(f"Variant {group!r} names unknown met {met!r}")
    realizations = int(spec.pop("realizations", 1))
    if realizations < 1:
        raise ValueError(f"Variant {group!r}: realizations must be >= 1")
    return {**_override(defaults, spec), "met": met}, realizations


def _override(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a variant's overrides to *base*.

    A ``grid`` mapping updates the base grid field by field, so a variant can
    change only the resolution; ``grid: null`` still removes the footprint.
    """
    merged = {**base, **spec}
    if isinstance(spec.get("grid"), dict) and isinstance(base.get("grid"), dict):
        merged["grid"] = {**base["grid"], **spec["grid"]}
    return merged


def _expand_realizations(
    group: str,
    merged: dict[str, Any],
    realizations: int,
    declared: dict[str, dict[str, Any]],
) -> list[VariantConfig]:
    """One variant, or ``group-0 .. group-(N-1)`` each with ``seed + k``."""
    base = VariantConfig(name=group, group=group, **merged)
    if realizations == 1:
        return [base]
    if not (base.krand == 4 or (base.krand == 2 and base.seed is not None)):
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


def _check_derived(
    name: str, parent: str, spec: dict[str, Any], declared: dict[str, dict[str, Any]]
) -> None:
    """Reject a ``from:`` variant that would change more than the footprint."""
    if parent not in declared:
        raise ValueError(f"Variant {name!r} derives from unknown variant {parent!r}")
    parent_spec = declared[parent] or {}
    if "from" in parent_spec:
        raise ValueError(
            f"Variant {name!r} derives from {parent!r}, which is itself derived"
        )
    if int(parent_spec.get("realizations", 1)) != 1:
        raise ValueError(
            f"Variant {name!r} derives from realization group {parent!r}; "
            "derive from a single run"
        )
    extra = set(spec) - FootprintParams.FIELDS
    if extra:
        raise ValueError(
            f"Variant {name!r} derives from {parent!r} and may only override "
            f"footprint settings {sorted(FootprintParams.FIELDS)}; got {sorted(extra)}"
        )


__all__ = [
    "UNRECORDED_FIELDS",
    "VARIANT_KEYS",
    "VARIANT_NAME_RE",
    "VariantConfig",
    "expand_variants",
]
