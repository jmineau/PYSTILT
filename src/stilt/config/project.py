"""The project config: defaults, meteorology, and variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .execution import ExecutionConfig
from .footprint import FootprintConfig
from .meteorology import MetConfig
from .variant import VARIANT_NAME_RE, VariantConfig, expand_variants, transport_config


class ProjectConfig(FootprintConfig):
    """
    A project's configuration, as read from ``config.yaml``.

    The transport model's parameters and the footprint fields are flat,
    top-level keys, and are the defaults for every variant. ``model`` names
    the transport model, HYSPLIT unless set; its parameters are checked by
    its own config class (:class:`stilt.transport.hysplit.HysplitConfig`)
    and are in :attr:`transport`. Each variant names a met and overrides
    some of the defaults. A variant that names another ``model`` gives that
    model's parameters itself. Without ``variants``, each met runs as one
    variant with the met's name.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        "hysplit",
        description=(
            "Transport model the variants run with, unless a variant names "
            "another. Its parameters are top-level keys of the config."
        ),
    )

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

    _transport: Any = PrivateAttr(None)
    _variant_configs: dict[str, VariantConfig] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Require at least one met, each with a valid name and a directory."""
        if not self.mets:
            raise ValueError(
                "ProjectConfig.mets must contain at least one meteorology configuration"
            )
        bad = [k for k in self.mets if not VARIANT_NAME_RE.fullmatch(k)]
        if bad:
            raise ValueError(
                f"Met names must match {VARIANT_NAME_RE.pattern}, got: {bad}"
            )
        for name, met in self.mets.items():
            if met.directory is None:
                raise ValueError(
                    f"Met {name!r} needs a directory: where its files are, or "
                    "where downloaded files are saved."
                )
            if met.subgrid_enable and met.download is None and met.subgrid_dir is None:
                raise ValueError(
                    f"Met {name!r}: subgrid_dir is required when subgrid_enable=True "
                    "without download. Set it to a directory for the cropped "
                    "files, outside the met archive."
                )
        return self

    @model_validator(mode="after")
    def _expand_variants(self) -> Self:
        """
        Merge each declared variant with the defaults, so a bad declaration fails when the config loads.

        No file is read: the grid of a footprint given by a geometry, and
        the transport model build, are found by
        :func:`stilt.variants.resolve`.
        """
        self._transport = transport_config(
            self.model, dict(self.model_extra or {}), f"config ({self.model})"
        )
        self._variant_configs = expand_variants(
            self._declared(),
            self.model,
            self._transport.model_dump(),
            self.model_dump(include=set(FootprintConfig.model_fields)),
            self.mets,
        )
        return self

    @property
    def transport(self) -> Any:
        """The transport model's config, from the top-level keys: the defaults every variant starts from."""
        return self._transport

    @property
    def variant_configs(self) -> dict[str, VariantConfig]:
        """
        Each variant merged with the defaults, by simulation name, in declared order.

        A variant with ``realizations`` becomes several, so ``hrrr-err`` with
        ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.
        ``project.variants`` resolves them.
        """
        return self._variant_configs

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
