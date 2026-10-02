"""The project config: defaults, meteorology, and variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import ConfigDict, Field, model_validator

from .execution import ExecutionConfig
from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import STILTParams
from .variant import VARIANT_NAME_RE, VariantConfig, check_variants, expand_variants


class ProjectConfig(STILTParams, FootprintConfig):
    """
    A project's configuration, as read from ``config.yaml``.

    The transport and footprint fields are the defaults for every variant.
    Each variant names a met and overrides some of the defaults. Without
    ``variants``, each met runs as one variant with the met's name.
    """

    model_config = ConfigDict(extra="forbid")

    mets: dict[str, MetConfig] = Field(
        default_factory=dict,
        description="Meteorology by name, such as ``hrrr``. At least one is required.",
    )
    variants: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Variants by name, each a set of overrides of the defaults. A "
            "variant may also set ``met`` (needed with several mets unless the "
            "variant has a met's name) and ``realizations`` (run N times, "
            "realization k with ``seed + k``). Variants with the same transport "
            "settings share one run of HYSPLIT and differ in the footprint made "
            "from it. Unset runs one variant per met."
        ),
    )
    execution: ExecutionConfig = Field(
        default_factory=lambda: ExecutionConfig.model_validate({}),
        description=(
            "Where the receptors run and with what resources, such as "
            "``backend: slurm`` and its options."
        ),
    )
    output: str = Field(
        "output",
        description=(
            "Directory the results go to, relative to the project directory "
            "unless absolute. Several projects can name the same directory and "
            "share runs."
        ),
    )

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Require at least one met, each with a valid variant name."""
        if not self.mets:
            raise ValueError(
                "ProjectConfig.mets must contain at least one meteorology configuration"
            )
        bad = [k for k in self.mets if not VARIANT_NAME_RE.fullmatch(k)]
        if bad:
            raise ValueError(
                f"Met names must match {VARIANT_NAME_RE.pattern}, got: {bad}"
            )
        return self

    @model_validator(mode="after")
    def _validate_variants(self) -> Self:
        """
        Check the variants so a bad declaration fails when the config loads.

        Nothing is built: loading a config reads no geometry file and does
        not look up the HYSPLIT build. :meth:`resolve_variants` builds them.
        """
        check_variants(self._declared(), self.defaults(), self.mets)
        return self

    def defaults(self) -> dict[str, Any]:
        """Return the default transport and footprint parameters every variant starts from."""
        parameters = set(STILTParams.model_fields) | set(FootprintConfig.model_fields)
        values = self.model_dump(include=parameters)
        # Keep the geometry spec itself, so the variants that inherit it share
        # one spec and build its mesh once.
        values["geometry"] = self.geometry
        return values

    def resolve_variants(self) -> dict[str, VariantConfig]:
        """
        Return one :class:`VariantConfig` per simulation name, in declared order.

        A variant with ``realizations`` becomes several, so ``hrrr-err`` with
        ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.

        The footprint settings are resolved
        (:meth:`~stilt.config.FootprintConfig.resolve`). A footprint given by
        ``geometry`` gets its grid and geometry hash here, and each geometry
        is built once however many variants inherit it.
        """
        return {
            name: variant
            if variant.footprint is None
            else variant.model_copy(update={"footprint": variant.footprint.resolve()})
            for name, variant in expand_variants(
                self._declared(), self.defaults(), self.mets
            ).items()
        }

    def _declared(self) -> dict[str, dict[str, Any]]:
        """Return the declared variants, one per met when none are declared."""
        return self.variants or {met: {"met": met} for met in self.mets}

    def to_yaml(self, path: str | Path | None = None) -> str:
        """
        Return the config as YAML, and write it to ``path`` when given.

        The file is meant to be edited by hand, so it holds only the settings
        that were given, top-level and for each met. ``mets`` and ``variants``
        come first. ``variants`` is always written, with one entry per met when
        none were declared, so the file lists the variants that run. The full
        settings of each variant are in the output directory, in the
        ``_settings.yaml`` of each ``settings=`` folder.
        """
        data = self.model_dump(mode="json", include=self.model_fields_set)
        data["mets"] = {
            name: met.model_dump(mode="json", exclude_unset=True)
            for name, met in self.mets.items()
        }
        if "execution" in data:
            data["execution"] = self.execution.model_dump(
                mode="json", exclude_unset=True
            )
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
        """Load a project config from a YAML file."""
        path = Path(path)
        with path.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


__all__ = ["ProjectConfig"]
