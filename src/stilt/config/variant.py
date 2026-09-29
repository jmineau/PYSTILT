"""
Variants: the complete settings a receptor is run under.

A project runs every receptor under every variant. A variant names its met
and holds a full set of transport and footprint settings, and it is one
HYSPLIT run per receptor. ``config.yaml`` declares variants as overrides of
its defaults, and :meth:`ModelConfig.resolve_variants` turns them into one
:class:`VariantConfig` per simulation name.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from .footprint import FootprintParams
from .params import STILTParams

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: Fields that change no output, so the record does not compare them.
UNRECORDED_FIELDS = frozenset({"timeout", "rm_dat", "exe_dir"})


class VariantConfig(STILTParams, FootprintParams):
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
    derived_from: str | None = Field(
        None,
        description=(
            "Variant whose trajectories this one computes its footprint from, "
            "instead of running HYSPLIT (``from:`` in ``config.yaml``). Only "
            "footprint fields may differ from it."
        ),
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

    @property
    def is_derived(self) -> bool:
        """Whether this variant reuses another variant's trajectories."""
        return self.derived_from is not None

    def stilt_params(self) -> STILTParams:
        """Return the transport parameters alone, as stored with a trajectory."""
        return STILTParams(**self.model_dump(include=set(STILTParams.model_fields)))

    def record(self) -> dict[str, Any]:
        """
        Return this variant as stored in the project's record.

        ``maxpar`` is stored as HYSPLIT receives it, so an unset ``maxpar`` is
        stored as ``numpar``.
        """
        data = self.model_dump(mode="json")
        if data["maxpar"] is None:
            data["maxpar"] = self.numpar
        return data

    def differences(self, recorded: dict[str, Any]) -> list[str]:
        """
        Return the names of the fields that differ from ``recorded``.

        ``recorded`` is this variant's entry in the project's record (see
        :meth:`record`). Fields that change no output
        (:data:`UNRECORDED_FIELDS`) are skipped.
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
    Return one :class:`VariantConfig` per simulation name.

    Parameters
    ----------
    declared : dict
        ``{name: overrides}`` as written in ``config.yaml``. Besides
        parameter overrides, each may set ``met``, ``realizations``, and
        ``from``.
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
    if "realizations" in parent_spec:
        raise ValueError(
            f"Variant {name!r} derives from realization group {parent!r}; "
            "derive from a single run"
        )
    footprint_fields = set(FootprintParams.model_fields)
    extra = set(spec) - footprint_fields
    if extra:
        raise ValueError(
            f"Variant {name!r} derives from {parent!r} and may only override "
            f"footprint settings {sorted(footprint_fields)}; got {sorted(extra)}"
        )


__all__ = [
    "UNRECORDED_FIELDS",
    "VARIANT_NAME_RE",
    "VariantConfig",
    "expand_variants",
]
