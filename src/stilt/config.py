"""
The project config: what ``config.yaml`` holds, and the variants it declares.

``config.yaml`` is blocks: ``mets`` names the meteorology, ``models`` the
transport models, ``stages`` what is made from each simulation's particles
(the footprint), and ``variants`` the settings every receptor runs under.
A model's or a stage's block holds its settings, the defaults of every
variant that uses it, and no setting sits at the top level. Each part has
its own config class, next to the code that uses it: the transport model's
and its met config (such as :class:`~stilt.transport.hysplit.HysplitConfig`
and :class:`~stilt.transport.hysplit.MetConfig`),
:class:`~stilt.footprint.config.FootprintConfig`, and
:class:`~stilt.execution.ExecutionConfig`. :class:`ProjectConfig` reads the
file and checks it without reading any other file, so a ``config.yaml``
loads offline. :meth:`ProjectConfig.resolve` turns each declared variant
into a :class:`Variant`, what its simulations run with; that reads each
footprint geometry and asks the transport model for its build. Which
weather each met is, read from its files, is read when a variant's
hash is first needed.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePath
from typing import Any, NamedTuple, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from stilt._paths import absolute
from stilt.execution.config import ExecutionConfig
from stilt.footprint.config import (
    TIME_INTEGRATE_REMOVED,
    FileGeometrySpec,
    FootprintConfig,
)
from stilt.footprint.targets import Mesh
from stilt.identity import (
    footprint_hash,
    footprint_settings,
    run_settings,
    settings_hash,
)
from stilt.particles import HNF_PLUME_COLUMNS
from stilt.transport import (
    MODELS,
    VEGHT_RENAMED,
    ModelInfo,
    TransportConfig,
    get_model,
)

#: Pattern for variant and met names, which become directory names.
VARIANT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: The import path of the footprint stage, the ``class`` of its ``stages:`` block.
FOOTPRINT_STAGE = "stilt.footprint.FootprintStage"

#: The settings of the footprint stage.
_FOOTPRINT_FIELDS = frozenset(FootprintConfig.model_fields)

#: Writes a block of settings given in Python (a path, a config object) as YAML can hold it.
_JSON = TypeAdapter(dict[str, Any])

#: A variant's own keys; every other key of a variant is a setting of its model.
_VARIANT_KEYS = ("met", "model", "realizations", "stages")


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
    block: str  # the name of its model's block under models:
    model: str  # the model, as stilt.transport.get_model names it
    realizations: int | None
    transport: dict[str, Any]
    footprint: FootprintConfig | None


@dataclass(frozen=True)
class Variant:
    """
    One variant, resolved: its configs, its met, its transport model, and its hashes.

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
    met_config : BaseModel
        That met's config, of the transport model's ``met_config_class``,
        with absolute directories when resolved for a project.
    transport : TransportConfig
        The transport model's config, such as a
        :class:`~stilt.transport.hysplit.HysplitConfig`, with the base seed.
    model : str
        The transport model that runs it, as ``config.yaml`` names it: a
        built-in name such as ``hysplit``, or a model class's import path.
        Its build is :attr:`build`.
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
    met_config: Any
    transport: TransportConfig
    model: str
    realizations: int | None = None
    footprint: FootprintConfig | None = None
    geometry_hash: str | None = None

    @property
    def realization_numbers(self) -> list[int | None]:
        """The realizations its simulations run as, from ``0`` to ``N - 1``, or ``[None]`` for a single run."""
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
    def met_settings(self) -> dict[str, Any]:
        """
        What a run records of the met, its config's ``settings()``.

        For HYSPLIT, the weather product and the crop. Read on first use,
        once for all the variants of a met: the product comes from the
        header of the met's first file.

        Raises
        ------
        FileNotFoundError
            If the met's directory holds no file to read its product from.
        """
        try:
            return self.met_config.settings()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Met {self.met!r}: {error}") from None

    @cached_property
    def build(self) -> ModelInfo:
        """
        The transport model build that runs it.

        For HYSPLIT, the SHA-256 of the ``hycs_std`` it runs and the version
        that ``hycs_std`` prints. Read on first use, as :attr:`met_settings`
        is, so a project opens on a machine without the model's executable.

        Raises
        ------
        FileNotFoundError
            If the model's executable cannot be found here, such as a
            machine without ``hycs_std`` that only reports status.
        """
        model = get_model(self.model)
        return ModelInfo(
            name=self.model,
            sha256=model.sha256(self.transport),
            data_files=model.data_files(self.transport),
            version=model.version(self.transport),
        )

    @cached_property
    def run_settings(self) -> dict[str, Any]:
        """The settings that identify this variant's particles, as ``_settings.yaml`` records them."""
        return run_settings(
            self.transport,
            self.met_settings,
            self.build,
            ensemble=self.realizations is not None,
        )

    @cached_property
    def readable_run_settings(self) -> dict[str, Any]:
        """
        The settings that identify this variant's particles, without what cannot be read here.

        ``met`` is left out when the met's files cannot be read, and
        ``model`` keeps only its ``name`` when the model's executable cannot
        be found. What can be compared with a stored folder when
        :attr:`run_settings` cannot be made; equal to it when it can.
        """
        try:
            met: dict[str, Any] | None = self.met_settings
        except FileNotFoundError:
            met = None
        try:
            build: ModelInfo | None = self.build
        except FileNotFoundError:
            build = None
        settings = run_settings(
            self.transport,
            met or {},
            build or ModelInfo(name=self.model),
            ensemble=self.realizations is not None,
        )
        if met is None:
            del settings["met"]
        if build is None:
            settings["model"] = {"name": self.model}
        return settings

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


class _UniqueKeyLoader(yaml.SafeLoader):
    """
    PyYAML's safe loader, refusing a key written twice in one mapping.

    PyYAML keeps the last value of a repeated key, so ``n_hours: -24`` and
    then ``n_hours: 24`` would run forward with no word.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Build a mapping, raising for its first key seen twice."""
        seen = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue  # PyYAML refuses a key that is not a plain value itself
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"{key!r} is written twice", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class ProjectConfig(BaseModel):
    """
    A project's configuration, as read from ``config.yaml``.

    The file is blocks. ``mets`` names the meteorology, ``models`` the
    transport models, ``stages`` what is made from each simulation's
    particles (the footprint), and ``variants`` what runs. A model's or a
    stage's block holds its settings, checked by its own config class: the
    defaults of every variant that uses it. Each variant names a met and a
    model, when there are several, and overrides some of the defaults; only
    the declared variants run. No setting sits at the top level: one written
    there is refused with a message that names its block. An unknown key is
    an error that names the nearest setting.

    Examples
    --------
    >>> config = ProjectConfig(
    ...     mets={"hrrr": "/path/to/hrrr_arl"},
    ...     models={"hysplit": {"n_hours": -24, "numpar": 200}},
    ...     stages={
    ...         "footprint": {
    ...             "class": "stilt.footprint.FootprintStage",
    ...             "grid": {
    ...                 "xmin": -113.0,
    ...                 "xmax": -110.5,
    ...                 "ymin": 40.0,
    ...                 "ymax": 42.0,
    ...                 "xres": 0.01,
    ...                 "yres": 0.01,
    ...             },
    ...         }
    ...     },
    ...     variants={"hrrr": {}, "hrrr-long": {"n_hours": -72}},
    ... )
    >>> list(config.variants)
    ['hrrr', 'hrrr-long']
    """

    model_config = ConfigDict(extra="forbid")

    mets: dict[str, dict[str, Any] | str] = Field(
        default_factory=dict,
        description=(
            "Meteorology by name, such as ``hrrr``: the folder of its files, or "
            "a mapping of its settings. At least one is required. Each is "
            "checked by the met config of the transport model that reads it "
            "(:class:`stilt.transport.hysplit.MetConfig` for HYSPLIT)."
        ),
    )
    models: dict[str, dict[str, Any] | None] = Field(
        default_factory=lambda: {"hysplit": {}},
        description=(
            "Transport models by name, each a block of its settings: the "
            "defaults of every variant that runs it. A block named "
            "``hysplit`` or ``stilt-r`` (runs imported from STILT-R, which "
            "cannot run) is that built-in model. A model class in another "
            "package gets a block of any name that gives its import path as "
            "``class``, such as ``class: mypkg.models.MyModel``. A model has "
            "one block. Left out, the project runs HYSPLIT with its defaults."
        ),
    )
    stages: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "What is made from each simulation's particles, by name, each a "
            "block that gives the stage's import path as ``class`` and then "
            "its settings. The footprint is ``class: "
            "stilt.footprint.FootprintStage``, with the footprint settings "
            "(:class:`~stilt.footprint.FootprintConfig`). A stage runs for "
            "every variant, unless the variant sets it to null. Left out, the "
            "project makes particles only."
        ),
    )
    variants: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description=(
            "Variants by name, each a set of overrides of the defaults. A "
            "variant's own keys are ``met`` (needed with several mets unless "
            "the variant has a met's name), ``model`` (the name of its "
            "``models`` block, needed with several), ``realizations`` (run N "
            "times, realization k with ``seed + k``, as one ensemble), and "
            "``stages`` (a stage's settings to change, by the stage's name, "
            "or null for a variant without it: ``stages: {footprint: null}`` "
            "makes particles only). Every other key is a setting of its "
            "model. Variants with the same transport settings share one "
            "transport model run and differ in the footprint made from it. At "
            "least one is required: ``hrrr: {}`` runs the defaults."
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
            "unless absolute. A URL such as ``s3://bucket/output`` puts it on "
            "an object store. Several projects can name the same directory and "
            "share runs."
        ),
    )

    # Each models: block by its name: the model, as get_model names it, and
    # the block's settings.
    _models: dict[str, tuple[str, dict[str, Any]]] = PrivateAttr(default_factory=dict)
    # The name of the footprint's stages: block and its checked settings.
    _footprint: tuple[str, FootprintConfig] | None = PrivateAttr(None)
    # Each met checked by the met config of a model that reads it, by
    # (met, model).
    _met_configs: dict[tuple[str, str], Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _no_setting_at_the_top(cls, data: Any) -> Any:
        """Refuse a top-level key that is not one of the config's, naming the block a setting belongs in."""
        if isinstance(data, dict):
            _check_keys(
                data,
                "config",
                allowed=dict.fromkeys(cls.model_fields, "a config key"),
                elsewhere=_blocks_of_settings(data.get("models")),
            )
        return data

    @field_validator("mets", mode="before")
    @classmethod
    def _met_mappings(cls, mets: Any) -> Any:
        """Accept a met config object or a path in Python, kept as the settings it was given."""
        if isinstance(mets, dict):
            return {
                name: met.model_dump(mode="json", exclude_unset=True)
                if isinstance(met, BaseModel)
                else str(met)
                if isinstance(met, PurePath)
                else met
                for name, met in mets.items()
            }
        return mets

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Require at least one met, each with a valid name."""
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
    def _validate_blocks_and_variants(self) -> Self:
        """
        Check each model's and stage's block and each declared variant, so a bad config fails when it loads.

        A model's block is checked by name, and by value as the variants
        that change none of its settings run it. A block need not be
        complete (a model may have a setting with no default), so the
        settings of a variant that adds its own are checked when it is
        resolved (:meth:`resolve`), as its realizations and the model's
        build are.
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
        self._models = _model_blocks(self.models)
        self._footprint = _footprint_block(self.stages, self._models)
        used = set()
        for group in self.variants:
            declared = self._declared(group)
            if declared.transport == self._models[declared.block][1]:
                where = f"models: {declared.block} ({declared.model})"
                transport_config(declared.model, declared.transport, where)
            self.met_config(declared.met, declared.block)
            used.add(declared.met)
        # With several models, a met no variant uses has no model to check it.
        if len(self._models) == 1:
            for name in self.mets.keys() - used:
                self.met_config(name)
        return self

    def met_config(self, name: str, model: str | None = None) -> Any:
        """
        Return the met called *name*, checked by the met config of the transport model that reads it.

        *model* is the name of a ``models`` block, needed when there are
        several. A relative directory stays as written; :meth:`resolve`
        makes it absolute.

        Raises
        ------
        ValueError
            If the met is not one the model can read, there is no such
            model, or there are several models and none is named.
        """
        if model is None:
            if len(self._models) > 1:
                raise ValueError(
                    f"Name the model that reads met {name!r}: one of "
                    f"{sorted(self._models)}."
                )
            model = next(iter(self._models))
        if model not in self._models:
            raise ValueError(
                f"There is no model {model!r} under models: (it has "
                f"{sorted(self._models)})."
            )
        model = self._models[model][0]
        key = (name, model)
        if key not in self._met_configs:
            met_class = get_model(model).met_config_class
            try:
                self._met_configs[key] = met_class.model_validate(self.mets[name])
            except ValidationError as error:
                raise ValueError(f"Met {name!r} ({model}): {error}") from None
        return self._met_configs[key]

    def _declared(self, group: str) -> _Declared:
        """
        Return one declared variant merged with the defaults of its model's block and the footprint's.

        Its transport fields are merged but not validated.

        Raises
        ------
        ValueError
            For a bad name, an unknown met, model or stage, a bad
            ``realizations``, ``from:``, a setting written outside its
            block, or a footprint without a grid.
        """
        if not VARIANT_NAME_RE.fullmatch(group):
            raise ValueError(
                f"Variant name {group!r} must match {VARIANT_NAME_RE.pattern}"
            )
        where = f"Variant {group!r}"
        spec = dict(self.variants[group] or {})
        if "from" in spec:
            raise ValueError(
                f"{where} uses 'from:', which is no longer needed: a "
                "variant with the same transport settings as another shares its "
                "particles. Give the variant the transport overrides of "
                f"{spec['from']!r} (if any) and its own footprint settings."
            )
        met = _met_name(group, spec, self.mets)
        block = _model_name(group, spec, self._models)
        model, defaults = self._models[block]
        stage = self._footprint[0] if self._footprint else "footprint"
        _check_keys(
            spec,
            where,
            allowed={
                **_model_settings(model),
                **dict.fromkeys(_VARIANT_KEYS, "a variant key"),
            },
            elsewhere=dict.fromkeys(
                _FOOTPRINT_FIELDS,
                f"a footprint setting. Write it under the variant's stages: {stage}:.",
            ),
        )
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"{where}: realizations must be >= 1")
        footprint = self._variant_footprint(group, spec.pop("stages", None))
        transport = {**defaults, **spec}
        _check_near_field_columns(group, transport, footprint)
        return _Declared(
            met=met,
            block=block,
            model=model,
            realizations=realizations,
            transport=transport,
            footprint=footprint,
        )

    def _variant_footprint(self, group: str, stages: Any) -> FootprintConfig | None:
        """
        Return the footprint settings of a variant: the footprint's block with the variant's ``stages`` overrides.

        ``None`` when the config has no footprint stage, or the variant sets
        it to null.

        Raises
        ------
        ValueError
            If the variant names a stage the config does not have, sets what
            is no footprint setting, or ends with neither a grid nor a
            geometry.
        """
        where = f"Variant {group!r}"
        stages = {} if stages is None else stages
        if not isinstance(stages, dict):
            raise ValueError(
                f"{where}: stages must map a stage's name to the settings it "
                "changes, or to null."
            )
        for name in stages:
            if name not in self.stages:
                raise ValueError(
                    f"{where} names the stage {name!r}, which stages: does not "
                    f"have (it has {sorted(self.stages)})."
                )
        if self._footprint is None:
            return None
        name, defaults = self._footprint
        fields = defaults.model_dump()
        if name in stages:
            own = stages[name]
            if own is None:
                return None
            _check_keys(
                own,
                f"{where}, stages: {name}",
                allowed=dict.fromkeys(_FOOTPRINT_FIELDS, "a footprint setting"),
                elsewhere={
                    "class": "the stage itself, set in the stages: block and "
                    "not by a variant."
                },
            )
            fields = _override(group, fields, dict(own))
        try:
            config = FootprintConfig(**fields)
        except ValidationError as error:
            raise ValueError(f"{where}, stages: {name}: {error}") from None
        if config.grid is None and config.geometry is None:
            raise ValueError(
                f"{where}: its footprint has no grid. Give stages: {name}: a "
                f"grid or a geometry, or set the variant's stages: {{{name}: null}} "
                "to make particles only."
            )
        return config

    def resolve(self, directory: str | Path | None = None) -> dict[str, Variant]:
        """
        Return one :class:`Variant` per declared variant, in declared order.

        Each variant's transport settings are validated by its model's config
        class, as are its realizations (``realizations: N`` runs it N
        times, realization ``k`` with ``seed + k``). Each met's directories
        are made absolute; what a run records of the met is read on first
        use (:attr:`Variant.met_settings`). Each geometry is read
        once, however many variants use it, and the grid of a footprint
        given only by a geometry is derived from it
        (:meth:`stilt.Mesh.to_grid`). The transport model's build is read
        on first use too (:attr:`Variant.build`).
        ``project.variants`` holds the result.

        Parameters
        ----------
        directory : str or Path, optional
            Where a relative geometry file or met directory starts: the
            project directory. Without it, the working directory.

        Raises
        ------
        ValueError
            If a variant's transport settings are invalid for its model, or its
            realizations would repeat one another.
        """
        meshes: dict[str, Mesh] = {}
        mets: dict[tuple[str, str], Any] = {}
        variants: dict[str, Variant] = {}
        for group in self.variants:
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
            if declared.realizations is not None:
                try:
                    transport.realizations(declared.realizations)
                except ValueError as error:
                    raise ValueError(f"Variant {group!r}: {error}") from None
            met_key = (declared.met, declared.model)
            if met_key not in mets:
                mets[met_key] = _located(
                    self.met_config(declared.met, declared.block), directory
                )
            variants[group] = Variant(
                name=group,
                met=declared.met,
                met_config=mets[met_key],
                transport=transport,
                model=declared.model,
                realizations=declared.realizations,
                footprint=footprint,
                geometry_hash=geometry_hash,
            )
        return variants

    def to_yaml(self, path: str | Path | None = None) -> str:
        """
        Return the config as YAML, and write it to ``path`` when given.

        The file is meant to be edited by hand, so it holds only the settings
        that were given, in each block. ``mets``, ``models``, ``stages`` and
        ``variants`` come first. The full settings of each variant are in
        the output directory, in the ``_settings.yaml`` of each ``settings=``
        folder.
        """
        given = self.model_fields_set
        data = self.model_dump(mode="json", include=given - {"models", "stages"})
        # A block's settings as its class writes them (a transform by its kind).
        if "models" in given:
            data["models"] = _JSON.dump_python(self.models, mode="json")
        if "stages" in given and self._footprint is not None:
            name, config = self._footprint
            block = self.stages[name]
            written = config.model_dump(mode="json", include=set(block) - {"class"})
            data["stages"] = {name: {"class": block["class"], **written}}
        if "execution" in data:
            data["execution"] = self.execution.model_dump(
                mode="json", exclude_unset=True
            )
        order = ("mets", "models", "stages", "variants")
        head = {k: data.pop(k) for k in order if k in data}
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
        """
        Load a project config from a YAML file.

        Raises
        ------
        ValueError
            If the file is not valid YAML, writes one key twice in a section
            (YAML would keep the last value), or holds a wrong setting.
        """
        path = Path(path)
        try:
            with path.open() as f:
                raw: dict = yaml.load(f, Loader=_UniqueKeyLoader) or {}
        except yaml.YAMLError as error:
            # PyYAML's own message spans several lines; keep where and what.
            mark = getattr(error, "problem_mark", None)
            where = (
                ""
                if mark is None
                else f" at line {mark.line + 1}, column {mark.column + 1}"
            )
            problem = getattr(error, "problem", None) or error
            raise ValueError(
                f"{path.name} is not valid YAML{where}: {problem}."
            ) from None
        return cls.model_validate(raw)


#: STILT-R's HYSPLIT settings where they differ from what HYSPLIT does on its
#: own: ``Project.init`` and ``stilt init`` write them into a new project's
#: ``models: hysplit:`` block, so it runs HYSPLIT as STILT-R does. The values
#: are those of STILT-R's ``run_stilt.r``. ``kblt``, ``kmixd`` and ``vscales``
#: are what HYSPLIT chooses itself in STILT mode, and are written so that
#: ``SETUP.CFG`` matches STILT-R's line for line. ``varsiwant`` is STILT-R's
#: list with ``pres`` added, for pressure weighting.
STILT_R_HYSPLIT: dict[str, Any] = {
    "numpar": 1000,
    "capemin": -1.0,
    "kblt": 5,
    "kmixd": 3,
    "krand": 4,
    "vscales": -1.0,
    "wvert": False,
    "varsiwant": [
        "time",
        "indx",
        "long",
        "lati",
        "zagl",
        "foot",
        "mlht",
        "pres",
        "dens",
        "samt",
        "sigw",
        "tlgr",
    ],
}

#: The commented ``config.yaml`` that ``Project.init(path, starter=True)`` and
#: ``stilt init`` write, to edit by hand.
STARTER_CONFIG = """# PYSTILT project configuration
# See docs for details: https://jmineau.github.io/PYSTILT


# Meteorology, by name: the folder of your ARL files. To download them from NOAA
# instead, write hrrr: {download: hrrr, directory: /path/to/keep/them}.
mets:
  hrrr: /path/to/arl/meteorology


# Transport models, by name, each with its settings: the defaults of every
# variant that runs it. Negative n_hours means backward in time.
models:
  hysplit:
    n_hours: -24
    # STILT-R's values where HYSPLIT's own defaults differ. A setting left
    # out runs with HYSPLIT's default.
    numpar: 1000
    capemin: -1     # no convection
    kblt: 5         # Hanna turbulence
    kmixd: 3        # mixed-layer height from the bulk Richardson number
    krand: 4        # random numbers seeded from the clock
    vscales: -1     # Hanna timescale in a stable boundary layer
    wvert: false
    varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, dens, samt, sigw, tlgr]


# Stages, by name: what is made from each simulation's particles. The footprint
# stage grids them. Remove the stages block to make particles only.
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid:
      xmin: -113.0
      xmax: -110.5
      ymin: 40.0
      ymax: 42.0
      xres: 0.01
      yres: 0.01
    hnf_plume: true  # correct foot for the thin plume near the receptor before gridding


# Variants. Every receptor runs once per variant, with the blocks above as
# the defaults. An entry with no overrides runs the defaults as they are; add
# others to run the same receptors under other settings, e.g. a mixed-layer
# bracket or a wind-error ensemble (see the docs). A variant changes a
# footprint setting under its own stages: key. Only the variants listed here
# run.
variants:
  hrrr: {}
#  hrrr-zi08: {ziscale: 0.8}
#  hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}


# Results go to this directory, relative to the project unless absolute.
# Several projects can name the same directory and share runs.
output: ./output


# Execution is optional. Local execution is the default. Each backend's
# settings are in the block of its name, and only the backend's block is read.
# execution:
#   backend: local    # or "slurm"
#   local:
#     cpus: 1         # receptors at once
#   slurm:
#     n_workers: 1    # array tasks
#     cpus: 1         # receptors at once in each task
#     time: "02:00:00"
"""


def _model_blocks(
    models: dict[str, dict[str, Any] | None],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """
    Return each ``models`` block's model and settings, by the block's name.

    The model is as :func:`stilt.transport.get_model` names it: the block's
    ``class``, or the block's own name for a built-in model. The settings
    are checked by name here, and by value as a variant runs them.

    Raises
    ------
    ValueError
        If there is no block, a block names no model, two blocks name one
        model, or a block holds what is no setting of its model.
    """
    if not models:
        raise ValueError(
            "models: has no block. Leave it out to run HYSPLIT with its "
            "defaults, or write models: {hysplit: {}}."
        )
    blocks: dict[str, tuple[str, dict[str, Any]]] = {}
    block_of: dict[str, str] = {}
    for name, block in models.items():
        if not VARIANT_NAME_RE.fullmatch(name):
            raise ValueError(
                f"Model name {name!r} must match {VARIANT_NAME_RE.pattern}"
            )
        where = f"models: {name}"
        settings = dict(block or {})
        model = settings.pop("class", name)
        if "class" in (block or {}) and "." not in str(model):
            raise ValueError(
                f"{where}: class must be the import path of a model class, "
                f"such as mypkg.models.MyModel, not {model!r}."
            )
        if "." not in model and model not in MODELS:
            raise ValueError(
                f"{where}: {model!r} is not a built-in transport model (they "
                f"are {sorted(MODELS)}). A model in another package gets "
                "class: with its import path, such as class: mypkg.models.MyModel."
            )
        if model in block_of:
            raise ValueError(
                f"models: {block_of[model]} and {name} are both {model}. A "
                "model has one block, its defaults: write a second variant to "
                "run it with other settings."
            )
        block_of[model] = name
        _check_keys(
            settings,
            where,
            allowed=_model_settings(model),
            elsewhere=dict.fromkeys(
                _FOOTPRINT_FIELDS,
                "a footprint setting. Write it under stages:, in the footprint's block.",
            ),
        )
        blocks[name] = (model, settings)
    return blocks


def _footprint_block(
    stages: dict[str, dict[str, Any]], models: dict[str, tuple[str, dict[str, Any]]]
) -> tuple[str, FootprintConfig] | None:
    """
    Return the name of the footprint's ``stages`` block and its checked settings, or ``None`` without one.

    The footprint is the one stage there is, so each block must name it as
    its ``class`` (:data:`FOOTPRINT_STAGE`), and only one can. *models* are
    the ``models`` blocks, to say where a model's setting belongs.

    Raises
    ------
    ValueError
        If a block names no class or another one, two blocks name one
        class, or a block holds a wrong setting.
    """
    elsewhere = {
        field: f"a {model} setting. Write it under models: {name}:."
        for name, (model, _) in models.items()
        for field in _model_settings(model)
    }
    found: tuple[str, FootprintConfig] | None = None
    for name, block in stages.items():
        if not VARIANT_NAME_RE.fullmatch(name):
            raise ValueError(
                f"Stage name {name!r} must match {VARIANT_NAME_RE.pattern}"
            )
        where = f"stages: {name}"
        settings = dict(block or {})
        stage = settings.pop("class", None)
        if stage is None:
            raise ValueError(
                f"{where} names no class. The footprint is class: {FOOTPRINT_STAGE}."
            )
        if stage != FOOTPRINT_STAGE:
            raise ValueError(
                f"{where}: {stage!r} is not a stage. The one stage is the "
                f"footprint, class: {FOOTPRINT_STAGE}."
            )
        if found is not None:
            raise ValueError(
                f"stages: {found[0]} and {name} are both {stage}. A stage has "
                "one block, its defaults: write a second variant for a second "
                "footprint."
            )
        _check_keys(
            settings,
            where,
            allowed=dict.fromkeys(_FOOTPRINT_FIELDS, "a footprint setting"),
            elsewhere=elsewhere,
        )
        try:
            found = (name, FootprintConfig.model_validate(settings))
        except ValidationError as error:
            raise ValueError(f"{where}: {error}") from None
    return found


def _model_settings(model: str) -> dict[str, str]:
    """Return the settings of the transport model *model*, each with what to call it in a message."""
    fields = get_model(model).config_class.model_fields
    return dict.fromkeys(fields, f"a {model} setting")


def _blocks_of_settings(models: Any) -> dict[str, str]:
    """
    Return where each setting that may not sit at the top of the config is written, by the setting's name.

    *models* is the ``models`` block as written, not checked yet; HYSPLIT's
    settings are named when it gives no model that can be read.
    """
    blocks = models if isinstance(models, dict) and models else {"hysplit": {}}
    places: dict[str, str] = {}
    for name, block in blocks.items():
        model = block.get("class", name) if isinstance(block, dict) else name
        try:
            fields = _model_settings(str(model))
        except (ImportError, ValueError):
            continue  # reported when the models block itself is checked
        for field in fields:
            places.setdefault(
                field, f"a {model} setting. Write it under models: {name}:."
            )
    places.update(
        dict.fromkeys(
            _FOOTPRINT_FIELDS,
            "a footprint setting. Write it under stages:, in the footprint's "
            f"block (class: {FOOTPRINT_STAGE}).",
        )
    )
    places["model"] = (
        "no longer a key. Name each transport model as a block under "
        "models:, such as models: {hysplit: {}}."
    )
    return places


def _met_name(group: str, spec: dict[str, Any], mets: dict[str, Any]) -> str:
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


def _model_name(group: str, spec: dict[str, Any], models: dict[str, Any]) -> str:
    """Return the name of the ``models`` block a variant runs with, taking ``model`` out of *spec*."""
    name = spec.pop("model", None)
    if name is None:
        if len(models) > 1:
            raise ValueError(
                f"Variant {group!r} must name its model (one of {sorted(models)})"
            )
        return next(iter(models))
    if name not in models:
        raise ValueError(
            f"Variant {group!r} names the model {name!r}, which models: does "
            f"not have (it has {sorted(models)})."
        )
    return name


def _located(met: Any, directory: str | Path | None) -> Any:
    """
    Return *met* with its ``directory`` and ``subgrid_dir`` absolute, when it has them.

    A relative path starts from *directory*, or the working directory, and
    ``~`` and ``$VARIABLES`` are expanded.
    """
    paths = {
        field: absolute(value, directory)
        for field in ("directory", "subgrid_dir")
        if (value := getattr(met, field, None)) is not None
    }
    return met.model_copy(update=paths)


def _override(group: str, base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a variant's overrides to ``base``.

    *base* is the footprint's block and *spec* what the variant changes of
    it. A ``grid`` mapping updates the base grid field by field, so a
    variant can change only the resolution. A variant that sets its own
    ``geometry`` drops the inherited ``grid`` unless it sets ``grid`` too,
    so the grid is derived from its geometry.

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


def _check_near_field_columns(
    name: str, transport: dict[str, Any], footprint: FootprintConfig | None
) -> None:
    """
    Refuse a variant whose footprint has the near-field correction and whose model is not asked for its columns.

    Only a model with ``varsiwant`` (HYSPLIT's list of particle columns) is
    checked; another model's particles are checked when the footprint is
    made.

    Raises
    ------
    ValueError
        If ``hnf_plume`` is set and ``varsiwant`` lacks a column it reads.
    """
    varsiwant = transport.get("varsiwant")
    if footprint is None or not footprint.hnf_plume or varsiwant is None:
        return
    missing = sorted(set(HNF_PLUME_COLUMNS) - set(varsiwant))
    if missing:
        raise ValueError(
            f"Variant {name!r}: the near-field correction (hnf_plume) reads the "
            f"particle columns {missing}. Add them to varsiwant, or set "
            "hnf_plume: false."
        )


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


def _check_keys(
    keys: Iterable[str], where: str, allowed: dict[str, str], elsewhere: dict[str, str]
) -> None:
    """
    Raise for the first key that may not be written at *where*, saying where it belongs or naming the nearest key.

    *allowed* are the keys that may be written there, each with what it is
    (``a hysplit setting``). *elsewhere* are settings that belong in another
    block, each with what it is and where to write it, as a sentence.

    Raises
    ------
    ValueError
        If a key is not one of *allowed*.
    """
    for key in keys:
        if key in allowed or key == "from":  # "from:" has its own message
            continue
        if key == "time_integrate":
            raise ValueError(f"{where}: {TIME_INTEGRATE_REMOVED}")
        if key == "veght":
            raise ValueError(f"{where}: {VEGHT_RENAMED}")
        if key in elsewhere:
            raise ValueError(f"{where}: {key!r} is {elsewhere[key]}")
        close = difflib.get_close_matches(str(key), [*allowed, *elsewhere], n=1)
        hint = ""
        if close and close[0] in allowed:
            hint = f" Did you mean {close[0]!r}, {allowed[close[0]]}?"
        elif close:
            hint = f" Did you mean {close[0]!r}? It is {elsewhere[close[0]]}"
        raise ValueError(f"{where}: {key!r} is not a setting.{hint}")


__all__ = [
    "FOOTPRINT_STAGE",
    "STARTER_CONFIG",
    "STILT_R_HYSPLIT",
    "VARIANT_NAME_RE",
    "ProjectConfig",
    "Variant",
    "transport_config",
]
