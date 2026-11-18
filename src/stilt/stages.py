"""
The stage interface: what is made from each simulation's results after the transport model runs.

A stage makes one file per release from the particles, from the files of
other stages, or both, and writes that file itself. The footprint is one
(:class:`stilt.footprint.FootprintStage`). A project lists its stages under
``stages:`` in ``config.yaml``, each block naming the stage's class by its
import path, so the worker runs a stage without importing it by name and a
stage in another package needs no registration:

.. code-block:: yaml

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}

:class:`Stage` is what such a class provides. :func:`get_stage` imports
one, :func:`ordered` puts the stages of a config in the order they run, and
:func:`check_stage` is for the tests of a stage of your own.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from stilt._plugins import import_class
from stilt.identity import canonical

if TYPE_CHECKING:
    import pandas as pd

    from stilt._paths import Location
    from stilt.simulation import Simulation


#: Pattern for a stage's name, which is a folder in the output directory and a column of ``project.status()``.
STAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Names a stage may not take: the output directory's own trees, and the columns of ``project.status()``.
RESERVED_NAMES = frozenset(
    {
        "particles",
        "logs",
        "scratch",
        "release",
        "variant",
        "realization",
        "model",
        "state",
        "stage",
        "reason",
        "message",
    }
)


class Stage(Protocol):
    """
    What is made from each simulation's results: one file per release.

    A class with these members is a stage; it subclasses nothing. It is made
    with no arguments, and holds no state: everything a stage needs is in
    its config and the simulation it is handed.

    Attributes
    ----------
    name : str
        What the stage's results are called, such as ``footprint``: the
        name of its block under ``stages:``, its tree in the output
        directory, its column in ``project.status()``, the ``stage`` of a
        failure record, and a part of the hash that names its settings
        folders. Lowercase letters, digits and underscores.
    config_class : type of pydantic.BaseModel
        Checks the stage's block of ``config.yaml`` when the file loads, and
        reads no file. It must also read back what :meth:`settings` wrote,
        since a stored settings folder is found by validating its record
        with this class and hashing :meth:`settings` again
        (:func:`check_stage` holds a stage to that). Two more members are
        optional. A classmethod ``override(block, own)`` returns a variant's
        settings from the block's, as the class dumps them, and the
        variant's own, as written; without it the variant's keys replace the
        block's. A property ``zsfc_reasons`` lists what goes wrong for
        releases above sea level whose particles lack the terrain height
        ``zsfc``, a clause each, for ``stilt check`` to report when the
        transport model is not asked for it.
    suffix : str
        The extension of the stage's files, such as ``.parquet``, with its
        dot. The file of a release is ``<release id><suffix>``, and
        completion looks for it in a listing.
    reads : tuple of str, optional
        The names of the stages whose files this one reads, none when the
        stage does not say. A stage runs after the ones it reads, and again
        whenever one of them was made again.
    """

    name: str
    config_class: type[BaseModel]
    suffix: str

    def resolve(
        self,
        config: Any,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> Any:
        """
        Return the config a variant's simulations run with, from the one ``config.yaml`` gives.

        Called once for each variant when a project reads its variants. This
        is where the stage reads the files its settings name, and where it
        refuses a transport model whose particles it cannot use.

        Parameters
        ----------
        config : config_class
            The variant's settings of this stage, checked.
        model : str
            The variant's transport model, as its ``models`` block names it.
        model_settings : mapping
            The model's settings as a run records them.
        directory : Path or None
            The project directory, where a relative file name starts.

        Raises
        ------
        ValueError
            If the settings do not fit the model, or name a file that
            cannot be used.
        """
        ...

    def settings(self, config: Any) -> dict[str, Any]:
        """
        Return what identifies the stage's results, besides what it reads: the record of its settings folder.

        Plain values (mappings, lists, strings, numbers), which
        :attr:`config_class` reads back. The folder's hash is of this
        record, the particles' hash, and the hash of each stage in
        :attr:`reads`.
        """
        ...

    def check(
        self, config: Any, releases: pd.DataFrame, directory: Path | None
    ) -> list[tuple[str, str | None, str]]:
        """
        Return what would stop this stage for the project's releases, found without running anything.

        ``project.check()`` and ``stilt check`` call it. *releases* has one
        row per release point (``release``, ``time``, ``longitude``,
        ``latitude``, ``altitude``, ``altitude_ref``, ``kind``). Each thing
        found is a tuple of a short name for the check, the release id (or
        ``None`` when it is not about one release), and a message.
        """
        ...

    def run(self, config: Any, sim: Simulation, path: Location) -> None:
        """
        Make the stage's file for one simulation, and write it at *path*.

        The worker calls it for each simulation whose file is missing, after
        the particles and the files of the stages in :attr:`reads` exist.
        Read the particles from ``sim.particles`` (a pandas DataFrame, which
        simulations that share it hold in common: do not change it in
        place), and another stage's file from ``sim.path("<its name>")``
        with that stage's own reader. Write one file at *path* and nothing
        else, whatever there is to say: a result with nothing in it is still
        a file, as an empty footprint is. On a filesystem *path* is a
        temporary name beside the file's own, ending in :attr:`suffix`,
        which the worker moves into place when this returns; a file left
        by a run that raises is removed. So a run cut off part-way leaves
        no file that counts as complete. A Parquet file may carry
        :func:`stilt.identity.file_stamp` in its metadata, as PYSTILT's own
        files do.

        Parameters
        ----------
        config : config_class
            The variant's settings of this stage, as :meth:`resolve`
            returned them.
        sim : Simulation
            The simulation: its release, its variant, its particles, and
            the project directory (``sim.directory``), where a relative file
            name in the settings starts.
        path : Path or UPath
            Where to write the file.

        Raises
        ------
        Exception
            Any error fails the simulation at this stage: the worker records
            it (:attr:`stilt.Simulation.failure`) and goes on. Raise a
            :class:`~stilt.exceptions.SimulationError` with a ``reason`` for
            a failure you expect.
        """
        ...


@functools.cache
def get_stage(path: str) -> Stage:
    """
    Return the stage whose class is at the import path *path*, made with no arguments.

    Raises
    ------
    MissingClass
        If the class cannot be imported on this machine.
    ValueError
        If *path* is not an import path, or the class is not a stage: it
        lacks a member of :class:`Stage`, or its ``name``, ``suffix`` or
        ``config_class`` cannot be used.
    """
    if "." not in path:
        raise ValueError(
            f"A stage is named by the import path of its class, such as "
            f"stilt.footprint.FootprintStage, not {path!r}."
        )
    stage = import_class(path)()
    members = ("name", "config_class", "suffix", "resolve", "settings", "check", "run")
    missing = [member for member in members if not hasattr(stage, member)]
    if missing:
        raise ValueError(
            f"{path} is not a stage: it has no {', '.join(missing)} "
            "(stilt.stages.Stage lists what a stage provides)."
        )
    name = stage.name
    if not isinstance(name, str) or not STAGE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{path}: a stage's name must match {STAGE_NAME_RE.pattern}, not {name!r}."
        )
    if name in RESERVED_NAMES:
        raise ValueError(
            f"{path}: a stage cannot be named {name!r}, which is a folder of "
            "the output directory or a column of project.status()."
        )
    suffix = stage.suffix
    if not isinstance(suffix, str) or not suffix.startswith(".") or len(suffix) < 2:
        raise ValueError(
            f"{path}: a stage's suffix must be a file extension such as "
            f"'.parquet', not {suffix!r}."
        )
    config_class = stage.config_class
    if not (isinstance(config_class, type) and issubclass(config_class, BaseModel)):
        raise ValueError(
            f"{path}: a stage's config_class must be a pydantic model, not "
            f"{config_class!r}."
        )
    return stage


def reads(stage: Stage) -> tuple[str, ...]:
    """Return the names of the stages whose files *stage* reads, none when it does not say."""
    return tuple(getattr(stage, "reads", ()))


def ordered(stages: Mapping[str, Stage]) -> list[str]:
    """
    Return the names of *stages* in the order they run: each after the stages it reads.

    Stages that do not read one another keep the order they were given in.

    Raises
    ------
    ValueError
        If a stage reads one that is not among *stages*, or stages read
        each other in a circle.
    """
    for name, stage in stages.items():
        for read in reads(stage):
            if read not in stages:
                raise ValueError(
                    f"The stage {name!r} reads the stage {read!r}, which "
                    f"stages: does not have (it has {sorted(stages)})."
                )
    order: list[str] = []
    waiting = list(stages)
    while waiting:
        ready = [n for n in waiting if all(r in order for r in reads(stages[n]))]
        if not ready:
            raise ValueError(
                f"The stages {sorted(waiting)} read each other in a circle, so "
                "none of them can run first."
            )
        order.append(ready[0])
        waiting.remove(ready[0])
    return order


def check_stage(stage: Stage, config: Any) -> None:
    """
    Check that *stage* keeps the promises the worker and the output directory rely on, for one *config*.

    Call it from the tests of a stage of your own, with a config as
    :meth:`Stage.resolve` returns it, once for each kind of setting the
    stage has. It checks that the record :meth:`Stage.settings` writes is
    plain data, and that :attr:`Stage.config_class` reads it back to a
    config with the same record: a stored settings folder is found that
    way, so a stage that breaks it finds none of its results.

    Raises
    ------
    AssertionError
        Saying which promise is broken.
    """
    assert STAGE_NAME_RE.fullmatch(stage.name), (
        f"name {stage.name!r} must match {STAGE_NAME_RE.pattern}"
    )
    assert stage.name not in RESERVED_NAMES, f"name {stage.name!r} is reserved"
    assert stage.suffix.startswith("."), (
        f"suffix {stage.suffix!r} must be a file extension, such as '.parquet'"
    )
    assert issubclass(stage.config_class, BaseModel), (
        "config_class must be a pydantic model"
    )
    record = canonical(stage.settings(config))
    try:
        json.dumps(record)
    except TypeError as error:
        raise AssertionError(
            f"settings() must return plain data (mappings, lists, strings, "
            f"numbers): {error}"
        ) from None
    read_back = stage.config_class.model_validate(record)
    again = canonical(stage.settings(read_back))
    assert again == record, (
        f"{type(stage).__name__}.config_class does not read back what settings() "
        f"wrote: {record} became {again}"
    )


__all__ = [
    "RESERVED_NAMES",
    "STAGE_NAME_RE",
    "Stage",
    "check_stage",
    "get_stage",
    "ordered",
    "reads",
]
