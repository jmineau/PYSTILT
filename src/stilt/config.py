"""
The project config: what ``config.yaml`` holds.

``config.yaml`` names the transport model, the mets, and the variants. The
model's parameters and the footprint settings are flat, top-level keys: the
defaults every variant starts from. Each part has its own config class,
next to the code that uses it: the transport model's (such as
:class:`~stilt.transport.hysplit.HysplitConfig`),
:class:`~stilt.footprint.config.FootprintConfig`,
:class:`~stilt.meteorology.MetConfig`, and
:class:`~stilt.execution.ExecutionConfig`. :class:`ProjectConfig` reads the
file and checks it, without reading any other file.
:func:`stilt.variants.resolve` turns each declared variant into what its
simulations run with.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from stilt.execution.config import ExecutionConfig
from stilt.footprint.config import FootprintConfig
from stilt.meteorology import MetConfig
from stilt.transport import MODELS, get_model

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: The top-level keys that belong to the footprint; every other unknown key belongs to the model.
_FOOTPRINT_FIELDS = frozenset(FootprintConfig.model_fields)


def transport_config(model: str, fields: dict[str, Any], where: str) -> Any:
    """
    Return *fields* validated by the config class of the transport model called *model*.

    Raises
    ------
    ValueError
        If there is no such model, or a field is unknown to it or invalid.
    """
    config_class = get_model(model).config_class
    try:
        return config_class.model_validate(fields)
    except ValidationError as error:
        raise ValueError(f"{where}: {error}") from None


class Declared(NamedTuple):
    """
    One variant as declared in ``config.yaml``, merged with the defaults.

    Its transport settings are not validated yet, and a ``realizations``
    group is one entry; :func:`stilt.variants.resolve` does both.
    """

    met: str
    model: str
    realizations: int | None
    transport: dict[str, Any]
    footprint: FootprintConfig | None


class ProjectConfig(BaseModel):
    """
    A project's configuration, as read from ``config.yaml``.

    The transport model's parameters and the footprint fields are flat,
    top-level keys, and are the defaults for every variant. ``model`` names
    the transport model, HYSPLIT unless set; its parameters are checked by
    its own config class and are in :attr:`transport`. The footprint fields
    are in :attr:`footprint`. Each variant names a met and overrides some of
    the defaults. A variant that names another ``model`` gives that model's
    parameters itself. Without ``variants``, each met runs as one variant
    with the met's name.
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
    _footprint: FootprintConfig = PrivateAttr(
        default_factory=lambda: FootprintConfig.model_validate({})
    )

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
    def _validate_defaults_and_variants(self) -> Self:
        """
        Check the defaults and each declared variant, so a bad config fails when it loads.

        A variant's own transport settings are checked when it is resolved
        (:func:`stilt.variants.resolve`), by the transport model's config
        class; everything else is checked here.
        """
        extra = dict(self.model_extra or {})
        footprint = {k: extra.pop(k) for k in list(extra) if k in _FOOTPRINT_FIELDS}
        try:
            self._footprint = FootprintConfig.model_validate(footprint)
        except ValidationError as error:
            raise ValueError(f"config footprint settings: {error}") from None
        self._transport = transport_config(self.model, extra, f"config ({self.model})")
        declared = self.declared()
        for group in declared:
            for k in range(self.variant(group).realizations or 0):
                if f"{group}-{k}" in declared:
                    raise ValueError(
                        f"Variant '{group}-{k}' collides with realization {k} of {group!r}"
                    )
        return self

    @property
    def transport(self) -> Any:
        """The transport model's config, from the top-level keys: the defaults every variant starts from."""
        return self._transport

    @property
    def footprint(self) -> FootprintConfig:
        """The footprint settings, from the top-level keys: the defaults every variant starts from."""
        return self._footprint

    def declared(self) -> dict[str, dict[str, Any]]:
        """Return the variants as written, one per met when none are declared."""
        return self.variants or {met: {"met": met} for met in self.mets}

    def variant(self, group: str) -> Declared:
        """
        Return one declared variant merged with the defaults.

        A variant that names another ``model`` inherits only the met and
        the footprint fields; it gives that model's fields itself. Its
        transport fields are merged but not validated.

        Raises
        ------
        ValueError
            For a bad name, an unknown met or model, a bad ``realizations``,
            ``from:``, or footprint settings without a grid.
        """
        if not VARIANT_NAME_RE.fullmatch(group):
            raise ValueError(
                f"Variant name {group!r} must match {VARIANT_NAME_RE.pattern}"
            )
        spec = dict(self.declared()[group] or {})
        if "from" in spec:
            raise ValueError(
                f"Variant {group!r} uses 'from:', which is no longer needed: a "
                "variant with the same transport settings as another shares its "
                "particles. Give the variant the transport overrides of "
                f"{spec['from']!r} (if any) and its own footprint settings."
            )
        met = _met_name(group, spec, self.mets)
        model = spec.pop("model", self.model)
        if model not in MODELS:
            raise ValueError(
                f"Unknown transport model {model!r}. The models are {sorted(MODELS)}."
            )
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"Variant {group!r}: realizations must be >= 1")
        base = {
            **(self._transport.model_dump() if model == self.model else {}),
            **self._footprint.model_dump(),
        }
        merged = _override(group, base, spec)
        return Declared(
            met=met,
            model=model,
            realizations=realizations,
            transport={k: v for k, v in merged.items() if k not in _FOOTPRINT_FIELDS},
            footprint=_footprint(
                group, {k: v for k, v in merged.items() if k in _FOOTPRINT_FIELDS}
            ),
        )

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
        given = set(self.model_extra or {})
        # Footprint keys as their class writes them (a transform by its kind).
        data.update(
            self._footprint.model_dump(mode="json", include=given & _FOOTPRINT_FIELDS)
        )
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


#: The commented ``config.yaml`` that ``Project.init(path, starter=True)`` and
#: ``stilt init`` write, to edit by hand.
STARTER_CONFIG = """# PYSTILT project configuration
# See docs for details: https://jmineau.github.io/PYSTILT


# Meteorology, by name. Edit directory to point to your ARL files, or replace
# file_format and file_tres with "download: hrrr" to download them from NOAA.
mets:
  hrrr:  # Unique name for this met.
    directory: /path/to/arl/meteorology
    file_format: "%Y%m%d_%H"
    file_tres: 6h  # Hours each met file covers; the docs' HRRR files hold six.


# Footprint grid. Remove it (or set grid: null) to make particles only.
grid:
  xmin: -113.0
  xmax: -110.5
  ymin: 40.0
  ymax: 42.0
  xres: 0.01
  yres: 0.01


# Variants. Every receptor runs once per variant, with the settings in this
# file as the defaults. An entry with no overrides runs the defaults as they
# are; add others to run the same receptors under other settings, e.g. a
# mixed-layer bracket or a wind-error ensemble (see the docs). Only the
# variants listed here run.
variants:
  hrrr: {}
#  hrrr-zi08: {ziscale: 0.8}


# Common run controls. Negative n_hours means backward in time.
n_hours: -24
numpar: 1000
varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, dens, samt, sigw, tlgr]
hnf_plume: true  # rescale footprints via a gaussian plume model in the hyper-near field


# Results go to this directory, relative to the project unless absolute.
# Several projects can name the same directory and share runs.
output: ./output


# Execution is optional. Local execution is the default.
# execution:
#   backend: local  # or "slurm"
#   cpus: 1         # receptors at once (per array task on Slurm)
#   n_workers: 1    # Slurm array tasks
"""


def _met_name(group: str, spec: dict[str, Any], mets: dict[str, MetConfig]) -> str:
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
    variant that sets its own ``geometry`` drops the inherited ``grid``
    unless it sets ``grid`` too, so the grid is derived from its geometry.

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
    if spec.get("geometry") is not None and "grid" not in spec:
        merged.pop("grid", None)
    return merged


def _footprint(name: str, fields: dict[str, Any]) -> FootprintConfig | None:
    """
    Return the footprint settings, or ``None`` when they give no grid or geometry.

    The geometry is not read here.

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


__all__ = [
    "STARTER_CONFIG",
    "VARIANT_NAME_RE",
    "Declared",
    "ProjectConfig",
    "transport_config",
]
