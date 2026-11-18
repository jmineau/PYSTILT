"""
The check before a run: what would stop a project's simulations, found without running one.

:func:`check_project` looks at the inputs of a project, never at its results:
that it has releases, that the transport model's
executable is there (HYSPLIT's ``hycs_std``), the met files each release needs,
whether each release is inside the met's crop, whether ``varsiwant`` holds
``zsfc`` where the run needs it, and what each stage of a variant finds
wrong with its own inputs (:meth:`stilt.stages.Stage.check`): for the
footprint, a release outside the grid, or an averaging-kernel table that
does not match the releases. ``Project.check`` and ``stilt check`` call it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd

from stilt.particles.table import needs_height_to_match
from stilt.transport import get_model

if TYPE_CHECKING:
    from stilt.config import Variant
    from stilt.project import Project

#: The columns of the table :func:`check_project` returns.
CHECK_COLUMNS = ("check", "variant", "release", "message")

#: One thing found, a tuple of its check, the release it is about (``None`` when it is not about one), and what is wrong.
_Finding = tuple[str, str | None, str]

#: A check of one variant, a tuple of what it looks at (variants that look at the same thing share one result) and how to make it.
_Check = tuple[Any, Callable[[], list[_Finding]]]


def check_project(project: Project) -> pd.DataFrame:
    """
    Return what would stop the simulations of *project*, one row for each thing found.

    The columns are ``check``, ``variant``, ``release`` (``NA`` when the
    finding is not about one release), and ``message``; the table is empty
    when nothing is found. Variants with the same finding share its row:
    ``variant`` then lists them, separated by commas.

    Raises
    ------
    ValueError
        If ``config.yaml`` or ``releases.csv`` is invalid.
    """
    rows = project._rows
    # Each finding with the variants it holds for, in the order found.
    found: dict[_Finding, list[str]] = {}
    if rows.empty:
        found[_NO_RELEASES] = list(project.variants)
    made: dict[Any, list[_Finding]] = {}  # each check's findings, made once
    configs: list[Any] = []  # the distinct stage configs, which may not hash
    for name, variant in project.variants.items():
        checks = (
            *_build_checks(variant),
            *_met_checks(variant, rows),
            *_stage_checks(variant, rows, project, configs),
            *_zsfc_checks(variant, rows),
        )
        for key, make in checks:
            if key not in made:
                made[key] = make()
            for finding in made[key]:
                names = found.setdefault(finding, [])
                if name not in names:
                    names.append(name)
    table = [
        (check, ", ".join(names), release, message)
        for (check, release, message), names in found.items()
    ]
    return pd.DataFrame(table, columns=list(CHECK_COLUMNS)).astype("string")


#: The finding of a project with no releases, which has nothing to run.
_NO_RELEASES: _Finding = (
    "releases",
    None,
    "The project has no releases: releases.csv is missing or holds only its "
    "header. Add them with project.add_releases, or write rows to releases.csv.",
)


def _build_checks(variant: Variant) -> Iterator[_Check]:
    """Yield the variant's transport model build when it cannot be found here, such as no ``hycs_std``."""
    # Variants that differ only in what changes no particle (directory) share it.
    where = variant.transport.model_dump_json(include=set(variant.transport.UNRECORDED))

    def make() -> list[_Finding]:
        """Return the model's message when its executable cannot be found, running nothing."""
        model = get_model(variant.model)
        try:
            model.sha256(variant.transport)
            model.data_files(variant.transport)
        except FileNotFoundError as error:
            return [("model", None, str(error))]
        return []

    yield ("model", variant.model, where), make


def _met_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """Yield what the variant's transport model finds wrong with its met, for the project's releases."""
    check_met = getattr(get_model(variant.model), "check_met", None)
    if check_met is None:
        return
    n_hours = variant.transport.n_hours

    def make() -> list[_Finding]:
        """Return the model's findings for the project's releases as plain tuples."""
        table = check_met(variant.met_config, rows, n_hours)
        return [
            (str(check), None if pd.isna(release) else str(release), str(message))
            for check, release, message in table.itertuples(index=False)
        ]

    yield ("met", variant.met, variant.model, n_hours), make


def _stage_checks(
    variant: Variant, rows: pd.DataFrame, project: Project, configs: list[Any]
) -> Iterator[_Check]:
    """
    Yield what each of the variant's stages finds wrong for the project's releases.

    A stage checks its own inputs (:meth:`stilt.stages.Stage.check`): the
    footprint stage lists the releases outside its grid and what each of
    its transforms finds. Variants with equal settings of a stage share its
    check; *configs* holds the settings seen so far.
    """
    for name, config in variant.stages.items():
        if not any(known == config for known in configs):
            configs.append(config)
        seen = next(i for i, known in enumerate(configs) if known == config)
        stage = variant.stage(name)
        yield (
            ("stage", name, seen),
            lambda stage=stage, config=config: [
                (str(check), None if release is None else str(release), str(message))
                for check, release, message in stage.check(
                    config, rows, project.directory
                )
            ],
        )


def _zsfc_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """
    Yield the missing ``zsfc`` of a variant with releases above sea level that need it.

    The run reads ``zsfc`` to match the release heights of a multipoint
    release's particles. A stage that reads it too says what goes wrong
    without it, in the ``zsfc_reasons`` of its settings, as the footprint's
    do for the near-field correction.
    """
    varsiwant = getattr(variant.transport, "varsiwant", None)
    if varsiwant is None or "zsfc" in varsiwant:
        return
    above_sea_level = rows[rows["altitude_ref"] == "msl"]
    n_msl = above_sea_level["release"].nunique()
    if n_msl == 0:
        return
    stage_reasons = tuple(
        str(reason)
        for config in variant.stages.values()
        for reason in getattr(config, "zsfc_reasons", ())
    )
    n_by_height = sum(
        needs_height_to_match(g["longitude"], g["latitude"], g["altitude"])
        for _, g in above_sea_level[above_sea_level["kind"] == "multipoint"].groupby(
            "release"
        )
    )
    if not (n_by_height or stage_reasons):
        return
    key = ("zsfc", tuple(varsiwant), n_msl, n_by_height, stage_reasons)
    message = _zsfc_message(varsiwant, n_msl, n_by_height, stage_reasons)
    yield key, lambda: [("zsfc", None, message)]


def _zsfc_message(
    varsiwant: list[str], n_msl: int, n_by_height: int, stage_reasons: tuple[str, ...]
) -> str:
    """Say what needs ``zsfc`` and the line that adds it to ``varsiwant``."""
    reasons = []
    if n_by_height:
        reasons.append(
            f"{n_by_height} of them {'is a multipoint release' if n_by_height == 1 else 'are multipoint releases'} "
            "whose points are less than 1 km apart and at different altitudes, "
            "so the release heights of the particles are matched by position "
            "and can come out wrong"
        )
    reasons += stage_reasons
    releases = f"{n_msl} release is" if n_msl == 1 else f"{n_msl} releases are"
    fix = ", ".join([*varsiwant, "zsfc"])
    return (
        f"zsfc is not in varsiwant, and {releases} above sea level: "
        f"{'; '.join(reasons)}. Add it in config.yaml with varsiwant: [{fix}]. "
        "That changes the settings folder, so the particles are made again"
    )


__all__ = ["CHECK_COLUMNS", "check_project"]
