"""The project config: defaults, meteorology, and variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import STILTParams
from .variant import VARIANT_NAME_RE, VariantConfig, expand_variants

#: ModelConfig fields that are not parameters a variant inherits.
_PROJECT_FIELDS = frozenset({"mets", "variants", "execution"})


class ModelConfig(STILTParams, FootprintConfig):
    """
    A project's configuration, as read from ``config.yaml``.

    The transport and footprint fields are the defaults for every variant.
    Each variant names a met and overrides some of the defaults. Without
    ``variants``, each met runs as one variant with the met's name.
    """

    model_config = ConfigDict(extra="forbid")

    mets: dict[str, MetConfig] = Field(
        default_factory=dict,
        description="Meteorology streams by name. At least one is required.",
    )
    variants: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Variants by name, each a set of overrides of the defaults. A "
            "variant may also set ``met`` (needed with several mets unless the "
            "variant has a met's name), ``realizations`` (run N times, "
            "realization k with ``seed + k``), or ``from`` (compute a footprint "
            "from another variant's trajectories, changing only footprint "
            "fields). Unset runs one variant per met."
        ),
    )
    execution: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution backend settings, such as ``backend: slurm`` and its options.",
    )

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Require at least one met, each with a valid variant name."""
        if not self.mets:
            raise ValueError(
                "ModelConfig.mets must contain at least one meteorology configuration"
            )
        bad = [k for k in self.mets if not VARIANT_NAME_RE.fullmatch(k)]
        if bad:
            raise ValueError(
                f"Met names must match {VARIANT_NAME_RE.pattern}, got: {bad}"
            )
        return self

    @model_validator(mode="after")
    def _validate_variants(self) -> Self:
        """Resolve the variants so a bad declaration fails when the config loads."""
        from stilt.transforms import UnresolvedTransform

        for name, variant in self.resolve_variants().items():
            for t in variant.transforms:
                if isinstance(t, UnresolvedTransform):
                    raise ValueError(
                        f"Variant {name!r} transform {t.kind!r} could not be "
                        f"imported: {t.reason}"
                    )
        return self

    def defaults(self) -> dict[str, Any]:
        """Return the default parameters every variant starts from."""
        return self.model_dump(exclude=set(_PROJECT_FIELDS))

    def resolve_variants(self) -> dict[str, VariantConfig]:
        """
        Return one :class:`VariantConfig` per simulation name, in declared order.

        A variant with ``realizations`` becomes several, so ``hrrr-err`` with
        ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.
        """
        declared = self.variants or {met: {"met": met} for met in self.mets}
        return expand_variants(declared, self.defaults(), list(self.mets))

    def to_yaml(self, path: str | Path | None = None) -> str:
        """
        Return the config as YAML, and write it to ``path`` when given.

        The file is meant to be edited by hand, so it holds only the settings
        that were given, top-level and for each met. ``mets`` and ``variants``
        come first. ``variants`` is always written, with one entry per met when
        none were declared, so the file lists the variants that run. The full
        settings of each variant are kept in the project's record
        (:meth:`stilt.Project.load_record`).
        """
        data = self.model_dump(mode="json", include=self.model_fields_set)
        data["mets"] = {
            name: met.model_dump(mode="json", exclude_unset=True)
            for name, met in self.mets.items()
        }
        if not data.get("variants"):
            data["variants"] = {met: {} for met in self.mets}
        head = {k: data.pop(k) for k in ("mets", "variants")}
        text = yaml.safe_dump(
            {**head, **data}, default_flow_style=False, sort_keys=False
        )
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a model config from a YAML file."""
        path = Path(path)
        with path.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


__all__ = ["ModelConfig"]
