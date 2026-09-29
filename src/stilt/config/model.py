"""Project-level config: flat defaults, met streams, and variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .footprint import FootprintParams
from .meteorology import MetConfig
from .params import STILTParams
from .variant import VARIANT_NAME_RE, VariantConfig, expand_variants

T = TypeVar("T", bound=BaseModel)

#: ModelConfig fields that are not parameters a variant inherits.
_PROJECT_FIELDS = frozenset({"mets", "variants", "execution"})


class ModelConfig(STILTParams, FootprintParams):
    """
    Project-level config.

    The flat transport and footprint fields are the **defaults**; a variant is
    ``{met: ..., <overrides>}`` merged onto them. With no ``variants`` given,
    one variant per met is generated, named after the met.
    """

    model_config = ConfigDict(extra="forbid")

    mets: dict[str, MetConfig] = Field(
        default_factory=dict,
        description="Named meteorology streams available to the model.",
    )
    variants: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Named variants as overrides of the defaults. Each may set ``met`` "
            "(required with several mets), ``realizations`` (run N times with "
            "seed + k), or ``from`` (rasterize another variant's trajectory; "
            "footprint fields only). Absent: one variant per met."
        ),
    )
    execution: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution backend settings such as local, Slurm, or Kubernetes options.",
    )

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """At least one met, each named so it can also name a variant."""
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
        """Resolve the variants once so a bad declaration fails at load time."""
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
        """The flat default parameters every variant starts from."""
        return self.model_dump(exclude=set(_PROJECT_FIELDS))

    def resolve_variants(self) -> dict[str, VariantConfig]:
        """
        One :class:`VariantConfig` per simulation name, in declaration order.

        Realization groups are expanded (``hrrr-err`` with ``realizations: 3``
        gives ``hrrr-err-0`` .. ``hrrr-err-2``).
        """
        declared = self.variants or {met: {"met": met} for met in self.mets}
        return expand_variants(declared, self.defaults(), list(self.mets))

    def to_stilt_params(self) -> STILTParams:
        """The default transport parameters alone."""
        return STILTParams(**self.model_dump(include=set(STILTParams.model_fields)))

    def to_yaml(self, path: str | Path) -> None:
        """
        Write the config to a YAML file, leaving out settings at their defaults.

        The file is meant to be read and edited by hand, so it holds only the
        top-level settings that were given; what was given is written in full
        (a transform or geometry keeps its ``kind``). The resolved settings of
        each variant are kept in the project's record
        (:meth:`stilt.Project.load_record`), not here.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", include=self.model_fields_set)
        with path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a model config from a YAML file."""
        path = Path(path)
        with path.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


def _config_or_kwargs(
    config: T | None,
    kwargs: dict,
    cls: type[T],
) -> T | None:
    """Resolve a config-or-kwargs pair."""
    if config is not None and kwargs:
        raise TypeError(
            f"Cannot pass both a {cls.__name__} instance and keyword arguments."
        )
    if kwargs:
        return cls(**kwargs)
    return config


__all__ = ["ModelConfig"]
