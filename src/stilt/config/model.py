"""Project-level config models and YAML/doc helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .footprint import FootprintConfig
from .meteorology import MetConfig
from .params import STILTParams
from .spatial import Grid

T = TypeVar("T", bound=BaseModel)

_GRID_KEYS = frozenset({"xmin", "xmax", "ymin", "ymax", "xres", "yres", "projection"})
_REQUIRED_GRID_KEYS = frozenset({"xmin", "xmax", "ymin", "ymax", "xres", "yres"})


class ModelConfig(STILTParams):
    """Project-level config: STILT params plus met and footprint definitions."""

    model_config = ConfigDict(extra="forbid")

    footprints: dict[str, FootprintConfig] = Field(
        default_factory=dict,
        description="Named footprint products available for this model configuration.",
    )
    grids: dict[str, Grid] = Field(
        default_factory=dict,
        description="Named grids referenced by footprint definitions.",
    )
    mets: dict[str, MetConfig] = Field(
        default_factory=dict,
        description="Named meteorology streams available to the model.",
    )
    execution: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution backend settings such as local, Slurm, or Kubernetes options.",
    )
    skip_existing: bool = Field(
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
                        shorthand_keys = _REQUIRED_GRID_KEYS & set(cfg)
                        if shorthand_keys == _REQUIRED_GRID_KEYS:
                            cfg["grid"] = {
                                key: cfg.pop(key) for key in _GRID_KEYS if key in cfg
                            }
                        elif cfg.get("geometry") is None:
                            raise ValueError(
                                f"Footprint '{name}' is missing a 'grid' key "
                                "(or a 'geometry' to derive one from)."
                            )
                resolved[name] = cfg
            data = {**data, "footprints": resolved}
        return data

    @model_validator(mode="after")
    def _reject_unresolved_transforms(self) -> Self:
        """A project config must be runnable: every transform class must import."""
        from stilt.transforms import UnresolvedTransform

        for name, cfg in self.footprints.items():
            for t in cfg.transforms:
                if isinstance(t, UnresolvedTransform):
                    raise ValueError(
                        f"Footprint '{name}' transform {t.kind!r} could not be "
                        f"imported: {t.reason}"
                    )
        return self

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
