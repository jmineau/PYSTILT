"""
The project config: what ``config.yaml`` holds, and the variants it declares.

``config.yaml`` is blocks: ``mets`` names the meteorology, ``models`` the
transport models, ``stages`` what is made from each simulation's results
(the footprint), and ``variants`` the settings every receptor runs under.
A model's or a stage's block holds its settings, the defaults of every
variant that uses it, and no setting sits at the top level. Each part has
its own config class, next to the code that uses it: the transport model's
and its met config (such as :class:`~stilt.transport.hysplit.HysplitConfig`
and :class:`~stilt.transport.hysplit.MetConfig`), a stage's (the
footprint's :class:`~stilt.footprint.config.FootprintConfig`), and
:class:`~stilt.execution.ExecutionConfig`. A stage is named by the import
path of its class (:mod:`stilt.stages`), so this module imports none.
:class:`ProjectConfig` reads the file and checks it without reading any
other file, so a ``config.yaml`` loads offline.
:meth:`ProjectConfig.resolve` turns each declared variant into a
:class:`Variant`, what its simulations run with; that has each stage read
the files its settings name (a footprint geometry) and check that it fits
the variant's model, and asks the transport model for its build. Which
weather each met is, read from its files, is read when a variant's hash is
first needed.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
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
from stilt.identity import canonical, run_settings, settings_hash, stage_hash
from stilt.stages import Stage, get_stage, ordered, reads
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
    stages: dict[str, Any]  # each stage's checked settings, by the stage's name


class _StageBlock(NamedTuple):
    """One ``stages`` block of ``config.yaml``: the stage its ``class`` names, and its settings, checked."""

    path: str  # the import path of the stage's class, as written
    stage: Stage
    config: BaseModel


@dataclass(frozen=True)
class Variant:
    """
    One variant, resolved: its configs, its met, its transport model, and its hashes.

    Get one from ``project.variants`` or ``sim.variant``. A variant with
    ``realizations: N`` is an ensemble: its simulations run N times,
    realization ``k`` with ``seed + k`` (:meth:`transport_for`), and its
    results share one folder, a ``realization=k`` partition each.

    Variants with equal :attr:`particles_hash` share one run of the
    transport model per receptor (and realization) and differ only in what
    their stages make of it, such as their footprints. The hashes are
    computed once. Change a variant with :func:`dataclasses.replace`, which
    computes them again.

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
    stages : dict
        The settings of each stage the variant uses, by the stage's name, as
        the stage resolved them: ``stages["footprint"]`` is a
        :class:`~stilt.footprint.FootprintConfig` with its grid. Empty for a
        variant that makes particles only.
    stage_classes : dict of str to str
        The import path of each stage's class, by the stage's name.
    """

    name: str
    met: str
    met_config: Any
    transport: TransportConfig
    model: str
    realizations: int | None = None
    stages: dict[str, Any] = field(default_factory=dict)
    stage_classes: dict[str, str] = field(default_factory=dict)

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

    def stage(self, name: str) -> Stage:
        """
        Return the stage called *name*, one of :attr:`stages`.

        Raises
        ------
        KeyError
            If the variant does not use a stage of that name.
        """
        return get_stage(self.stage_classes[name])

    @cached_property
    def stage_order(self) -> list[str]:
        """The names of the variant's stages in the order they run, each after the stages it reads."""
        return ordered({name: self.stage(name) for name in self.stages})

    @cached_property
    def stage_settings(self) -> dict[str, dict[str, Any]]:
        """The settings that identify each stage's results, by the stage's name, as its ``_settings.yaml`` records them."""
        return {
            name: canonical(self.stage(name).settings(self.stages[name]))
            for name in self.stage_order
        }

    @cached_property
    def stage_hashes(self) -> dict[str, str]:
        """
        The hash that finds each stage's settings folder, by the stage's name.

        A stage's hash is of the particles' hash, the hash of each stage it
        reads, and its own settings (:func:`stilt.identity.stage_hash`).
        """
        if not self.stages:
            return {}  # without the particles' hash, which reads the met and the build
        return self.stage_hashes_on(self.particles_hash)

    def stage_hashes_on(self, particles_hash: str) -> dict[str, str]:
        """
        Return the hash of each stage's settings folder for particles hashed *particles_hash*.

        :attr:`stage_hashes` for the variant's own particles. Given the hash
        of a stored particles folder, it finds the stage folders made from
        it where :attr:`particles_hash` cannot be computed, as on a machine
        without the model's executable.
        """
        hashes: dict[str, str] = {}
        for name in self.stage_order:
            read = {r: hashes[r] for r in reads(self.stage(name))}
            hashes[name] = stage_hash(
                name, self.stage_settings[name], particles_hash, read
            )
        return hashes


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
    results (the footprint), and ``variants`` what runs. A model's or a
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
            "What is made from each simulation's results after the transport "
            "model runs, by name, each a block that gives the stage's import "
            "path as ``class`` and then its settings. A block is named for "
            "its stage: the footprint's is ``footprint``, with ``class: "
            "stilt.footprint.FootprintStage`` and the footprint settings "
            "(:class:`~stilt.footprint.FootprintConfig`). A stage runs for "
            "every variant, unless the variant sets it to null. Left out, "
            "the project makes particles only."
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
    # Each stages: block by its stage's name, in the order the stages run.
    _stages: dict[str, _StageBlock] = PrivateAttr(default_factory=dict)
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
                elsewhere=_blocks_of_settings(data.get("models"), data.get("stages")),
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
        self._models = _model_blocks(self.models, self.stages)
        self._stages = _stage_blocks(self.stages, self._models)
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
        Return one declared variant merged with the defaults of its model's block and of each stage's.

        Its transport fields are merged but not validated.

        Raises
        ------
        ValueError
            For a bad name, an unknown met, model or stage, a bad
            ``realizations``, ``from:``, a setting written outside its
            block, or a stage set to null that another of its stages reads.
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
        _check_keys(
            spec,
            where,
            allowed={
                **_model_settings(model),
                **dict.fromkeys(_VARIANT_KEYS, "a variant key"),
            },
            elsewhere={
                setting: f"a {name} setting. Write it under the variant's "
                f"stages: {name}:."
                for setting, (name, _) in _stage_settings(self.stages).items()
            },
        )
        realizations = spec.pop("realizations", None)
        if realizations is not None:
            realizations = int(realizations)
            if realizations < 1:
                raise ValueError(f"{where}: realizations must be >= 1")
        stages = self._variant_stages(group, spec.pop("stages", None))
        transport = {**defaults, **spec}
        return _Declared(
            met=met,
            block=block,
            model=model,
            realizations=realizations,
            transport=transport,
            stages=stages,
        )

    def _variant_stages(self, group: str, own: Any) -> dict[str, Any]:
        """
        Return the settings of each stage a variant uses, by the stage's name, in the order the stages run.

        Each is the stage's block with the variant's ``stages`` overrides:
        the variant's keys replace the block's, unless the stage's config
        class merges them itself (its ``override``, as the footprint's does
        for a grid). A stage the variant sets to null is left out.

        Raises
        ------
        ValueError
            If the variant names a stage the config does not have, sets what
            is no setting of a stage, or sets a stage to null that another
            of its stages reads.
        """
        where = f"Variant {group!r}"
        own = {} if own is None else own
        if not isinstance(own, dict):
            raise ValueError(
                f"{where}: stages must map a stage's name to the settings it "
                "changes, or to null."
            )
        for name in own:
            if name not in self.stages:
                raise ValueError(
                    f"{where} names the stage {name!r}, which stages: does not "
                    f"have (it has {sorted(self.stages)})."
                )
        configs: dict[str, Any] = {}
        for name, block in self._stages.items():
            config = block.config
            if name in own:
                changes = own[name]
                if changes is None:
                    continue
                if not isinstance(changes, dict):
                    raise ValueError(
                        f"{where}, stages: {name}: give the settings the "
                        "variant changes as a mapping, or null for a variant "
                        "without the stage."
                    )
                config = _stage_config(
                    block.stage,
                    _overridden(block, dict(changes), f"{where}, stages: {name}"),
                    f"{where}, stages: {name}",
                    elsewhere={
                        "class": "the stage itself, set in the stages: block and "
                        "not by a variant."
                    },
                )
            configs[name] = config
        for name, block in self._stages.items():
            for read in reads(block.stage):
                if name in configs and read not in configs:
                    raise ValueError(
                        f"{where} sets the stage {read!r} to null, and its stage "
                        f"{name!r} reads that stage's files. Set {name!r} to "
                        f"null too, or keep {read!r}."
                    )
        return configs

    def resolve(self, directory: str | Path | None = None) -> dict[str, Variant]:
        """
        Return one :class:`Variant` per declared variant, in declared order.

        Each variant's transport settings are validated by its model's config
        class, as are its realizations (``realizations: N`` runs it N
        times, realization ``k`` with ``seed + k``). Each met's directories
        are made absolute; what a run records of the met is read on first
        use (:attr:`Variant.met_settings`). Each stage resolves the settings
        a variant gives it (:meth:`stilt.stages.Stage.resolve`): it reads the
        files they name and checks that they fit the variant's model. The
        footprint stage reads each geometry once, however many variants use
        it, and derives the grid of a footprint given only by a geometry
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
            If a variant's transport settings are invalid for its model, its
            realizations would repeat one another, or a stage's settings do
            not fit the variant's model.
        """
        start = None if directory is None else Path(directory)
        mets: dict[tuple[str, str], Any] = {}
        variants: dict[str, Variant] = {}
        for group in self.variants:
            declared = self._declared(group)
            transport = transport_config(
                declared.model,
                declared.transport,
                f"Variant {group!r} ({declared.model})",
            )
            stages: dict[str, Any] = {}
            for name, config in declared.stages.items():
                stage = self._stages[name].stage
                try:
                    stages[name] = stage.resolve(
                        config, declared.model, transport.settings(), start
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Variant {group!r}, stages: {name}: {error}"
                    ) from None
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
                stages=stages,
                stage_classes={name: self._stages[name].path for name in stages},
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
        if "stages" in given:
            data["stages"] = {
                name: {
                    "class": self._stages[name].path,
                    **self._stages[name].config.model_dump(
                        mode="json", include=set(block) - {"class"}
                    ),
                }
                for name, block in self.stages.items()
            }
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
    models: dict[str, dict[str, Any] | None], stages: Any = None
) -> dict[str, tuple[str, dict[str, Any]]]:
    """
    Return each ``models`` block's model and settings, by the block's name.

    *stages* is the ``stages`` block as written, to say where a stage's
    setting belongs.

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
            elsewhere=_stage_places(stages),
        )
        blocks[name] = (model, settings)
    return blocks


def _stage_blocks(
    stages: dict[str, dict[str, Any]], models: dict[str, tuple[str, dict[str, Any]]]
) -> dict[str, _StageBlock]:
    """
    Return each ``stages`` block's stage and checked settings, by the stage's name, in the order the stages run.

    A block names its stage by the import path of the stage's class
    (``class``), which is imported here, and is itself named for the stage:
    the footprint's block is ``footprint``. *models* are the ``models``
    blocks, to say where a model's setting belongs.

    Raises
    ------
    MissingClass
        If a block's class cannot be imported on this machine.
    ValueError
        If a block names no class or one that is not a stage, is not named
        for its stage, holds a wrong setting, or names a stage that reads
        one the config does not have.
    """
    elsewhere = {
        setting: f"a {model} setting. Write it under models: {name}:."
        for name, (model, _) in models.items()
        for setting in _model_settings(model)
    }
    blocks: dict[str, _StageBlock] = {}
    for name, block in stages.items():
        where = f"stages: {name}"
        settings = dict(block or {})
        path = settings.pop("class", None)
        if path is None:
            raise ValueError(
                f"{where} names no class. The footprint is class: {FOOTPRINT_STAGE}."
            )
        try:
            stage = get_stage(str(path))
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        if stage.name != name:
            # One name for the block, the stage's folder and its status column.
            raise ValueError(
                f"{where}: {path} is the {stage.name} stage, and a stage's "
                f"block is named for its stage. Write it as stages: {stage.name}:."
            )
        blocks[name] = _StageBlock(
            str(path), stage, _stage_config(stage, settings, where, elsewhere)
        )
    order = ordered({name: block.stage for name, block in blocks.items()})
    return {name: blocks[name] for name in order}


def _stage_config(
    stage: Stage, settings: dict[str, Any], where: str, elsewhere: dict[str, str]
) -> Any:
    """
    Return *settings* checked by the config class of *stage*.

    The class checks the values first, so that it can refuse a setting it
    once had with what to do instead; then a key that is no setting of the
    stage is refused, saying where it belongs (*elsewhere*) or naming the
    nearest setting.

    Raises
    ------
    ValueError
        If a setting is unknown to the stage or invalid.
    """
    try:
        config = stage.config_class.model_validate(settings)
    except ValidationError as error:
        raise ValueError(f"{where}: {error}") from None
    fields = stage.config_class.model_fields
    _check_keys(
        settings,
        where,
        allowed=dict.fromkeys(fields, f"a {stage.name} setting"),
        elsewhere=elsewhere,
    )
    return config


def _overridden(block: _StageBlock, own: dict[str, Any], where: str) -> dict[str, Any]:
    """
    Return a variant's settings of a stage: the block's, changed by the variant's *own*.

    The variant's keys replace the block's. A stage whose config class has
    an ``override(block, own)`` classmethod merges them itself: the
    footprint's changes a grid field by field.

    Raises
    ------
    ValueError
        If the config class refuses the variant's changes.
    """
    base = block.config.model_dump()
    override = getattr(block.stage.config_class, "override", None)
    if override is None:
        return {**base, **own}
    try:
        return override(base, own)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _stage_settings(stages: Any) -> dict[str, tuple[str, bool]]:
    """
    Return each stage setting a config could hold, with its stage's name and whether the config has that stage.

    *stages* is the ``stages`` block as written, not checked yet. A config
    without one gets the footprint's settings, so that a footprint setting
    written elsewhere is still recognized. A stage that cannot be imported
    names none: its own block reports that.
    """
    written = isinstance(stages, dict) and bool(stages)
    blocks = stages.values() if written else [{"class": FOOTPRINT_STAGE}]
    found: dict[str, tuple[str, bool]] = {}
    for block in blocks:
        try:
            stage = get_stage(str(block["class"]))
        except (ImportError, KeyError, TypeError, ValueError):
            continue
        for setting in stage.config_class.model_fields:
            found.setdefault(setting, (stage.name, written))
    return found


def _stage_places(stages: Any) -> dict[str, str]:
    """Return where each stage setting is written, as a sentence, by the setting's name."""
    return {
        setting: (
            f"a {name} setting. Write it under stages: {name}:."
            if written
            else f"a {name} setting. Write it under stages:, in the {name}'s "
            f"block (class: {FOOTPRINT_STAGE})."
        )
        for setting, (name, written) in _stage_settings(stages).items()
    }


def _model_settings(model: str) -> dict[str, str]:
    """Return the settings of the transport model *model*, each with what to call it in a message."""
    fields = get_model(model).config_class.model_fields
    return dict.fromkeys(fields, f"a {model} setting")


def _blocks_of_settings(models: Any, stages: Any = None) -> dict[str, str]:
    """
    Return where each setting that may not sit at the top of the config is written, by the setting's name.

    *models* and *stages* are those blocks as written, not checked yet;
    HYSPLIT's settings are named when the config gives no model that can be
    read, and the footprint's when it gives no stage.
    """
    blocks = models if isinstance(models, dict) and models else {"hysplit": {}}
    places: dict[str, str] = {}
    for name, block in blocks.items():
        model = block.get("class", name) if isinstance(block, dict) else name
        try:
            fields = _model_settings(str(model))
        except (ImportError, ValueError):
            continue  # reported when the models block itself is checked
        for setting in fields:
            places.setdefault(
                setting, f"a {model} setting. Write it under models: {name}:."
            )
    places.update(_stage_places(stages))
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
