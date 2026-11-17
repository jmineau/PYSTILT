"""
The check before a run: what would stop a project's simulations, found without running one.

:func:`check_project` looks at the inputs of a project, never at its results
or at HYSPLIT: that it has receptors, the met files each receptor needs,
whether each receptor is inside the footprint grid and the met's crop,
whether an averaging-kernel table matches the receptors, and whether
``varsiwant`` holds ``zsfc`` where the project needs it. ``Project.check`` and ``stilt check`` call it.
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
    from stilt.spatial import Grid

#: The columns of the table :func:`check_project` returns.
CHECK_COLUMNS = ("check", "variant", "receptor", "message")

#: One thing found, a tuple of its check, the receptor it is about (``None`` when it is not about one), and what is wrong.
_Finding = tuple[str, str | None, str]

#: A check of one variant, a tuple of what it looks at (variants that look at the same thing share one result) and how to make it.
_Check = tuple[Any, Callable[[], list[_Finding]]]


def check_project(project: Project) -> pd.DataFrame:
    """
    Return what would stop the simulations of *project*, one row for each thing found.

    The columns are ``check``, ``variant``, ``receptor`` (``NA`` when the
    finding is not about one receptor), and ``message``; the table is empty
    when nothing is found. Variants that share the thing a check looks at,
    such as a met and a run length, share its rows: ``variant`` then lists
    them, separated by commas.

    Raises
    ------
    ValueError
        If ``config.yaml`` or ``receptors.csv`` is invalid.
    """
    rows = project._rows
    ids = [str(r) for r in project.receptors["receptor"]]
    found: dict[Any, tuple[list[_Finding], list[str]]] = {}
    if not ids:
        found["receptors"] = ([_NO_RECEPTORS], list(project.variants))
    for name, variant in project.variants.items():
        checks = (
            *_met_checks(variant, rows),
            *_grid_checks(variant, rows),
            *_transform_checks(variant, project, ids),
            *_zsfc_checks(variant, rows),
        )
        for key, make in checks:
            if key not in found:
                found[key] = (make(), [])
            if name not in found[key][1]:
                found[key][1].append(name)
    table = [
        (check, ", ".join(names), receptor, message)
        for findings, names in found.values()
        for check, receptor, message in findings
    ]
    return pd.DataFrame(table, columns=list(CHECK_COLUMNS)).astype("string")


#: The finding of a project with no receptors, which has nothing to run.
_NO_RECEPTORS: _Finding = (
    "receptors",
    None,
    "The project has no receptors: receptors.csv is missing or holds only its "
    "header. Add them with project.add_receptors, or write rows to receptors.csv.",
)


def _met_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """Yield what the variant's transport model finds wrong with its met, for the project's receptors."""
    check_met = getattr(get_model(variant.model.name), "check_met", None)
    if check_met is None:
        return
    n_hours = variant.transport.n_hours

    def make() -> list[_Finding]:
        """Return the model's findings for the project's receptors as plain tuples."""
        table = check_met(variant.met_config, rows, n_hours)
        return [
            (str(check), None if pd.isna(receptor) else str(receptor), str(message))
            for check, receptor, message in table.itertuples(index=False)
        ]

    yield ("met", variant.met, variant.model.name, n_hours), make


def _grid_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """Yield the receptors with a release point outside the variant's footprint grid."""
    if variant.footprint is None or variant.footprint.grid is None:
        return
    grid = variant.footprint.grid
    yield ("grid", grid), lambda: _outside(rows, grid)


def _outside(rows: pd.DataFrame, grid: Grid) -> list[_Finding]:
    """List the receptors with a release point outside *grid*, which is bounded in longitude and latitude."""
    outside = rows[~grid.contains(rows["longitude"], rows["latitude"])]
    box = (
        f"{grid.xmin:g} to {grid.xmax:g} longitude, "
        f"{grid.ymin:g} to {grid.ymax:g} latitude"
    )
    return [
        (
            "grid",
            str(row.receptor),
            f"longitude {row.longitude:g} and latitude {row.latitude:g} are "
            f"outside the footprint grid ({box})",
        )
        for row in outside.drop_duplicates("receptor").itertuples()
    ]


def _transform_checks(
    variant: Variant, project: Project, ids: list[str]
) -> Iterator[_Check]:
    """
    Yield what each of the variant's transforms finds wrong for the project's receptors.

    A transform with a ``check(receptors, directory)`` method checks its own
    inputs, such as an averaging kernel's table; the others have nothing to
    check. Variants with an equal transform share its findings.
    """
    if variant.footprint is None:
        return
    for transform in variant.footprint.transforms:
        check = getattr(transform, "check", None)
        if check is None:
            continue
        yield (
            ("transform", repr(transform)),
            lambda check=check: [
                (str(name), None if receptor is None else str(receptor), str(message))
                for name, receptor, message in check(ids, project.directory)
            ],
        )


def _zsfc_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """Yield the missing ``zsfc`` of a variant with receptors above sea level, which the run reads for them."""
    varsiwant = getattr(variant.transport, "varsiwant", None)
    if varsiwant is None or "zsfc" in varsiwant:
        return
    above_sea_level = rows[rows["altitude_ref"] == "msl"]
    n_msl = above_sea_level["receptor"].nunique()
    if n_msl == 0:
        return
    near_field = variant.footprint is not None and variant.footprint.hnf_plume
    zsfc_transforms = tuple(
        _kind(t)
        for t in (() if variant.footprint is None else variant.footprint.transforms)
        if getattr(t, "reads_zsfc", False)
    )
    n_by_height = sum(
        needs_height_to_match(g["longitude"], g["latitude"], g["altitude"])
        for _, g in above_sea_level[above_sea_level["kind"] == "multipoint"].groupby(
            "receptor"
        )
    )
    if not (near_field or n_by_height or zsfc_transforms):
        return
    key = ("zsfc", tuple(varsiwant), n_msl, n_by_height, near_field, zsfc_transforms)
    message = _zsfc_message(varsiwant, n_msl, n_by_height, near_field, zsfc_transforms)
    yield key, lambda: [("zsfc", None, message)]


def _kind(transform: Any) -> str:
    """Return the ``kind`` a config names *transform* by, as :func:`stilt.transforms.transform_kind` does."""
    kind = getattr(transform, "kind", None)
    if isinstance(kind, str):
        return kind
    cls = type(transform)
    return f"{cls.__module__}.{cls.__qualname__}"


def _zsfc_message(
    varsiwant: list[str],
    n_msl: int,
    n_by_height: int,
    near_field: bool,
    zsfc_transforms: tuple[str, ...],
) -> str:
    """Say what needs ``zsfc`` and the line that adds it to ``varsiwant``."""
    reasons = []
    if near_field:
        reasons.append("the near-field correction (hnf_plume) is skipped for them")
    if n_by_height:
        reasons.append(
            f"{n_by_height} of them {'is a multipoint receptor' if n_by_height == 1 else 'are multipoint receptors'} "
            "whose points are less than 1 km apart and at different altitudes, "
            "so the release heights of the particles are matched by position "
            "and can come out wrong"
        )
    if zsfc_transforms:
        kinds = " and ".join(dict.fromkeys(zsfc_transforms))
        noun = (
            "transform fails" if len(set(zsfc_transforms)) == 1 else "transforms fail"
        )
        reasons.append(f"the {kinds} {noun} their footprints")
    receptors = f"{n_msl} receptor is" if n_msl == 1 else f"{n_msl} receptors are"
    fix = ", ".join([*varsiwant, "zsfc"])
    return (
        f"zsfc is not in varsiwant, and {receptors} above sea level: "
        f"{'; '.join(reasons)}. Add it in config.yaml with varsiwant: [{fix}]. "
        "That changes the settings folder, so the particles are made again"
    )


__all__ = ["CHECK_COLUMNS", "check_project"]
