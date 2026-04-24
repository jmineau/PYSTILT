"""Project-level config models and YAML/doc helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from .fields import T, _field_meta, cfg_field
from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import ErrorParams, ModelParams, STILTParams, TransportParams
from .spatial import Bounds, Grid


class ModelConfig(STILTParams):
    """Project-level config: STILT params plus met and footprint definitions."""

    model_config = ConfigDict(extra="forbid")

    footprints: dict[str, FootprintConfig] = cfg_field(
        default_factory=dict,
        description="Named footprint products available for this model configuration.",
    )
    grids: dict[str, Grid] = cfg_field(
        default_factory=dict,
        description="Named grids referenced by footprint definitions.",
    )
    mets: dict[str, MetConfig] = cfg_field(
        default_factory=dict,
        description="Named meteorology streams available to the model.",
    )
    execution: dict[str, Any] = cfg_field(
        default_factory=dict,
        description="Execution backend settings such as local, Slurm, or Kubernetes options.",
        visibility="advanced",
    )
    skip_existing: bool = cfg_field(
        True,
        description=(
            "Skip simulations that already have output. "
            "Set False to force re-run all simulations. "
            "Can be overridden at call time via model.run(skip_existing=...)."
        ),
    )

    @classmethod
    def basic(
        cls,
        *,
        mets: dict[str, MetConfig],
        n_hours: int = -24,
        numpar: int = 200,
        footprints: dict[str, FootprintConfig] | None = None,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Build a science-facing config with the most common controls."""
        return cls(
            mets=mets,
            n_hours=n_hours,
            numpar=numpar,
            footprints=footprints or {},
            skip_existing=skip_existing,
            **kwargs,
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_nested_configs(cls, data: dict) -> dict:
        """Expand named grid references in footprint configs before validation."""
        if not isinstance(data, dict):
            return data
        grids_raw = data.get("grids") or {}
        fp_raw = data.get("footprints") or {}
        if fp_raw:
            resolved = {}
            for name, cfg in fp_raw.items():
                if isinstance(cfg, dict):
                    cfg = dict(cfg)
                    grid_ref = cfg.get("grid")
                    if isinstance(grid_ref, str):
                        if grid_ref not in grids_raw:
                            raise ValueError(
                                f"Footprint '{name}' references unknown grid '{grid_ref}'"
                            )
                        cfg["grid"] = grids_raw[grid_ref]
                    elif grid_ref is None:
                        raise ValueError(f"Footprint '{name}' is missing a 'grid' key.")
                resolved[name] = cfg
            data = {**data, "footprints": resolved}
        return data

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Ensure each configured meteorology stream has a unique name."""
        if not self.mets:
            raise ValueError(
                "ModelConfig.mets must contain at least one meteorology configuration"
            )
        bad_keys = [k for k in self.mets if not k.isalnum()]
        if bad_keys:
            raise ValueError(
                f"Met keys must be alphanumeric (no underscores or special chars), got: {bad_keys}"
            )
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Write the model config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", exclude=set())
        for key in ("mets", "grids", "footprints", "execution"):
            if not data.get(key):
                del data[key]
        with path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def to_stilt_params(self) -> STILTParams:
        """Project this model config onto the pure STILT run-parameter surface."""
        data = self.model_dump(
            exclude={"footprints", "grids", "mets", "execution", "skip_existing"}
        )
        return STILTParams(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a model config from a YAML file."""
        path = Path(path)
        with path.open() as f:
            raw: dict = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


CONFIG_DOC_MODELS: tuple[type[BaseModel], ...] = (
    Bounds,
    Grid,
    MetConfig,
    FootprintConfig,
    ModelParams,
    TransportParams,
    ErrorParams,
    ModelConfig,
)


def iter_documented_config_fields(
    *models: type[BaseModel],
    include_internal: bool = False,
) -> Iterator[tuple[type[BaseModel], str, Any]]:
    """Yield config fields in declaration order for docs or UI generation."""
    if not models:
        models = CONFIG_DOC_MODELS
    for model in models:
        for name, field in model.model_fields.items():
            visibility = _field_meta(field).get("visibility", "public")
            if visibility == "internal" and not include_internal:
                continue
            yield model, name, field


def _collect_target_entries(
    params: BaseModel,
    model: type[BaseModel],
    *,
    target: str,
) -> dict[str, Any]:
    """Collect config fields whose metadata routes them to one output target."""
    entries: dict[str, Any] = {}
    default_target = getattr(model, "DEFAULT_TARGET", None)
    for name, field in model.model_fields.items():
        meta = _field_meta(field)
        field_target = meta.get("target", default_target)
        if field_target != target:
            continue
        value = getattr(params, name)
        if value is None:
            continue
        key = meta.get("namelist", name)
        entries[key] = value
    return entries


def build_setup_entries(params: STILTParams) -> dict[str, Any]:
    """Collect fields that belong in HYSPLIT ``SETUP.CFG``."""
    entries: dict[str, Any] = {}
    entries.update(_collect_target_entries(params, ModelParams, target="setup"))
    entries.update(_collect_target_entries(params, TransportParams, target="setup"))
    return entries


def build_control_entries(params: STILTParams) -> dict[str, Any]:
    """Collect fields that belong in HYSPLIT ``CONTROL``."""
    return _collect_target_entries(params, TransportParams, target="control")


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


__all__ = [
    "CONFIG_DOC_MODELS",
    "ModelConfig",
    "build_control_entries",
    "build_setup_entries",
    "iter_documented_config_fields",
]
