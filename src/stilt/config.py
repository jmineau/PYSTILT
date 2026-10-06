"""
The project config: what ``config.yaml`` holds, and the variants it declares.

``config.yaml`` names the transport model, the mets, and the variants. The
model's parameters and the footprint settings are flat, top-level keys: the
defaults every variant starts from. Each part has its own config class,
next to the code that uses it: the transport model's (such as
:class:`~stilt.transport.hysplit.HysplitConfig`),
:class:`~stilt.footprint.config.FootprintConfig`,
:class:`~stilt.meteorology.MetConfig`, and
:class:`~stilt.execution.ExecutionConfig`. :class:`ProjectConfig` reads the
file and checks it without reading any other file, so a ``config.yaml``
loads offline. :meth:`ProjectConfig.resolve` turns each declared variant
into a :class:`Variant`, what its simulations run with; that reads each
footprint geometry and asks the transport model for its build.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
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

from stilt._paths import absolute
from stilt.execution.config import ExecutionConfig
from stilt.footprint.config import FileGeometrySpec, FootprintConfig
from stilt.footprint.targets import Mesh
from stilt.identity import (
    footprint_hash,
    footprint_settings,
    run_settings,
    settings_hash,
)
from stilt.meteorology import MetConfig
from stilt.transport import MODELS, ModelInfo, TransportConfig, get_model

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


class _Declared(NamedTuple):
    """
    One variant as declared in ``config.yaml``, merged with the defaults.

    Its transport settings are not validated yet, and a ``realizations``
    group is one entry; :meth:`ProjectConfig.resolve` does both.
    """

    met: str
    model: str
    realizations: int | None
    transport: dict[str, Any]
    footprint: FootprintConfig | None


@dataclass(frozen=True)
class Variant:
    """
    One variant, resolved: its configs, its met, the model build, and its hashes.

    Get one from ``project.variants`` or ``sim.variant``. A variant with
    ``realizations: N`` is an ensemble: its simulations run N times,
    realization ``k`` with ``seed + k`` (:meth:`transport_for`), and its
    results share one folder, a ``realization=k`` partition each.

    Variants with equal :attr:`particles_hash` share one run of the
    transport model per receptor (and realization) and differ only in their
    footprints. The hashes are computed once. Change a variant with
    :func:`dataclasses.replace`, which computes them again.

    Attributes
    ----------
    name : str
        Name as declared in ``config.yaml``.
    met : str
        Name of the met it runs with.
    met_config : MetConfig
        That met's config.
    transport : TransportConfig
        The transport model's config, such as a
        :class:`~stilt.transport.hysplit.HysplitConfig`, with the base seed.
    model : ModelInfo
        The transport model build that runs it.
    realizations : int or None
        Number of realizations of an ensemble, or ``None`` for a single run.
    footprint : FootprintConfig or None
        Footprint config, with its grid, or ``None`` for particles only.
    geometry_hash : str or None
        Hash of the geometry the footprint grid was derived for
        (``Mesh.hash``), or ``None`` when the footprint has no geometry.
    """

    name: str
    met: str
    met_config: MetConfig
    transport: TransportConfig
    model: ModelInfo
    realizations: int | None = None
    footprint: FootprintConfig | None = None
    geometry_hash: str | None = None

    @property
    def realization_numbers(self) -> list[int | None]:
        """The realizations its simulations run as: ``0`` to ``N - 1``, or ``[None]`` for a single run."""
        if self.realizations is None:
            return [None]
        return list(range(self.realizations))

    def transport_for(self, realization: int | None) -> TransportConfig:
        """
        Return the transport config realization *realization* runs with.

        Realization ``k`` of an ensemble has ``seed + k``; a single run
        (``None``) has the variant's own.

        Raises
        ------
        ValueError
            If *realization* is not one of :attr:`realization_numbers`.
        """
        if realization not in self.realization_numbers:
            raise ValueError(
                f"Variant {self.name!r} runs as realizations "
                f"{self.realization_numbers}, not {realization!r}."
            )
        if realization is None:
            return self.transport
        return self.transport.realizations(realization + 1)[realization]

    @cached_property
    def run_settings(self) -> dict[str, Any]:
        """The settings that identify this variant's particles, as ``_settings.yaml`` records them."""
        return run_settings(
            self.transport,
            self.met_config,
            self.model,
            ensemble=self.realizations is not None,
        )

    @cached_property
    def particles_hash(self) -> str:
        """Hash of :attr:`run_settings`, which finds the particles folder."""
        return settings_hash(self.run_settings)

    @cached_property
    def footprint_settings(self) -> dict[str, Any] | None:
        """The settings that identify this variant's footprints, or ``None`` for particles only."""
        if self.footprint is None:
            return None
        return footprint_settings(self.footprint, self.geometry_hash)

    @cached_property
    def footprint_hash(self) -> str | None:
        """Hash of the particles and :attr:`footprint_settings`, which finds the footprint folder."""
        if self.footprint_settings is None:
            return None
        return footprint_hash(self.particles_hash, self.footprint_settings)


class ProjectConfig(BaseModel):
    """
    A project's configuration, as read from ``config.yaml``.

    The transport model's parameters and the footprint fields are flat,
    top-level keys, and are the defaults for every variant. ``model`` names
    the transport model, HYSPLIT unless set; its parameters are checked by
    its own config class and are in :attr:`transport`. The footprint fields
    are in :attr:`footprint`. Each variant names a met and overrides some of
    the defaults; only the declared variants run. A variant that names
    another ``model`` gives that model's own parameters itself, and
    inherits the parameters every model shares (:class:`TransportConfig`).
    An unknown key is an error that names the nearest setting.
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
            "realization k with ``seed + k``, as one ensemble). Variants with "
            "the same transport "
            "settings share one run of HYSPLIT and differ in the footprint made "
            "from it. At least one is required: ``hrrr: {}`` runs the defaults."
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
        (:meth:`resolve`), by the transport model's config class; everything
        else is checked here.
        """
        if not self.variants:
            first = next(iter(self.mets))
            raise ValueError(
                "The config has no variants. Add\n"
                "  variants:\n"
                f"    {first}: {{}}\n"
                f"to run the defaults with met {first}, and more entries for "
                "other settings."
            )
        extra = dict(self.model_extra or {})
        _check_keys(extra, self.model, "config", own=set(type(self).model_fields))
        footprint = {k: extra.pop(k) for k in list(extra) if k in _FOOTPRINT_FIELDS}
        try:
            self._footprint = FootprintConfig.model_validate(footprint)
        except ValidationError as error:
            raise ValueError(f"config footprint settings: {error}") from None
        self._transport = transport_config(self.model, extra, f"config ({self.model})")
        for group in self.declared():
            self._declared(group)
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
        """Return the variants as written."""
        return self.variants

    def _declared(self, group: str) -> _Declared:
        """
        Return one declared variant merged with the defaults.

        A variant that names another ``model`` inherits the met, the
        footprint fields, and the transport parameters every model shares
        (:class:`~stilt.transport.TransportConfig`); it gives that model's
        own fields itself. Its transport fields are merged but not
        validated.

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
        _check_keys(spec, model, f"Variant {group!r}", own={"realizations"})
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"Variant {group!r}: realizations must be >= 1")
        shared = set(TransportConfig.model_fields)
        defaults = self._transport.model_dump(
            include=None if model == self.model else shared
        )
        base = {**defaults, **self._footprint.model_dump()}
        merged = _override(group, base, spec)
        return _Declared(
            met=met,
            model=model,
            realizations=realizations,
            transport={k: v for k, v in merged.items() if k not in _FOOTPRINT_FIELDS},
            footprint=_footprint(
                group, {k: v for k, v in merged.items() if k in _FOOTPRINT_FIELDS}
            ),
        )

    def resolve(self, directory: str | Path | None = None) -> dict[str, Variant]:
        """
        Return one :class:`Variant` per declared variant, in declared order.

        Each variant's transport settings are validated by its model's config
        class, as are its realizations (``realizations: N`` runs it N
        times, realization ``k`` with ``seed + k``). Each geometry is read
        once, however many variants use it, and the grid of a footprint
        given only by a geometry is derived from it
        (:meth:`stilt.Mesh.to_grid`). The transport model's version and
        data files are read once for each distinct build: the fields of its
        config that change no particle, such as ``exe_dir``.
        ``project.variants`` holds the result.

        Parameters
        ----------
        directory : str or Path, optional
            Where a relative geometry file starts: the project directory.
            Without it, the working directory.

        Raises
        ------
        ValueError
            If a variant's transport settings are invalid for its model, or its
            realizations would repeat one another.
        """
        meshes: dict[str, Mesh] = {}
        builds: dict[tuple[str, str], ModelInfo] = {}
        variants: dict[str, Variant] = {}
        for group in self.declared():
            declared = self._declared(group)
            transport = transport_config(
                declared.model,
                declared.transport,
                f"Variant {group!r} ({declared.model})",
            )
            footprint, geometry_hash = declared.footprint, None
            if footprint is not None and footprint.geometry is not None:
                mesh = _mesh(footprint.geometry, directory, meshes)
                geometry_hash = mesh.hash
                if footprint.grid is None:
                    grid = mesh.to_grid(cells_per_target=footprint.cells_per_target)
                    footprint = footprint.model_copy(update={"grid": grid})
            where = transport.model_dump_json(include=set(transport.UNRECORDED))
            build = (declared.model, where)
            if build not in builds:
                model = get_model(declared.model)
                builds[build] = ModelInfo(
                    name=model.name,
                    version=model.version(transport),
                    data_files=model.data_files(transport),
                )
            if declared.realizations is not None:
                try:
                    transport.realizations(declared.realizations)
                except ValueError as error:
                    raise ValueError(f"Variant {group!r}: {error}") from None
            variants[group] = Variant(
                name=group,
                met=declared.met,
                met_config=self.mets[declared.met],
                transport=transport,
                model=builds[build],
                realizations=declared.realizations,
                footprint=footprint,
                geometry_hash=geometry_hash,
            )
        return variants

    def to_yaml(self, path: str | Path | None = None) -> str:
        """
        Return the config as YAML, and write it to ``path`` when given.

        The file is meant to be edited by hand, so it holds only the settings
        that were given, top-level and for each met. ``mets`` and ``variants``
        come first. The full settings of each variant are in the output
        directory, in the ``_settings.yaml`` of each ``settings=`` folder.
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


def _mesh(spec: Any, directory: str | Path | None, meshes: dict[str, Mesh]) -> Mesh:
    """
    Return the mesh a geometry spec describes, reading each spec once.

    A relative file starts from *directory*. The spec in the settings keeps
    the path as written, and a mesh's hash is of its polygons, so moving
    the project changes no hash.
    """
    key = spec.model_dump_json()
    if key not in meshes:
        if isinstance(spec, FileGeometrySpec) and directory is not None:
            spec = spec.model_copy(update={"path": str(absolute(spec.path, directory))})
        meshes[key] = Mesh.from_spec(spec)
    return meshes[key]


def _check_keys(keys: Iterable[str], model: str, where: str, own: set[str]) -> None:
    """
    Raise for the first key that is no setting, naming the nearest one and what it is.

    A key may be a footprint setting, a setting of the transport *model*,
    or one of *own*: the config's own keys at the top, or a variant's.

    Raises
    ------
    ValueError
        If a key is none of these.
    """
    parts = {name: "a footprint setting" for name in _FOOTPRINT_FIELDS}
    config_class = get_model(model).config_class
    parts.update({name: f"a {model} setting" for name in config_class.model_fields})
    parts.update({name: f"a {where.split()[0].lower()} key" for name in own})
    for key in keys:
        if key in parts or key == "from":  # "from:" has its own message
            continue
        close = difflib.get_close_matches(key, list(parts), n=1)
        hint = f" Did you mean {close[0]!r}, {parts[close[0]]}?" if close else ""
        raise ValueError(f"{where}: {key!r} is not a setting.{hint}")


__all__ = [
    "STARTER_CONFIG",
    "VARIANT_NAME_RE",
    "ProjectConfig",
    "Variant",
    "transport_config",
]
