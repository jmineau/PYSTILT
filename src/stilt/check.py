"""
The check before a run: what would stop a project's simulations, found without running one.

:func:`check_project` looks at the inputs of a project, never at its results
or at HYSPLIT: the met files each receptor needs, whether each receptor is
inside the footprint grid and the met's crop, whether an averaging-kernel
table matches the receptors, and whether ``varsiwant`` holds ``zsfc`` where
the project needs it. ``Project.check`` and ``stilt check`` call it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd

from stilt.transforms.averaging_kernel import AveragingKernel
from stilt.transforms.pressure_weighting import PressureWeighting
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
    for name, variant in project.variants.items():
        checks = (
            *_met_checks(variant, rows),
            *_grid_checks(variant, rows),
            *_kernel_checks(variant, project, ids),
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
    lon, lat = rows["longitude"], rows["latitude"]
    outside = rows[
        (lon < grid.xmin) | (lon > grid.xmax) | (lat < grid.ymin) | (lat > grid.ymax)
    ]
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


def _kernel_checks(
    variant: Variant, project: Project, ids: list[str]
) -> Iterator[_Check]:
    """Yield, for each averaging-kernel table the variant names, the receptors with no kernel in it and the kernels with no receptor."""
    if variant.footprint is None:
        return
    for transform in variant.footprint.transforms:
        if isinstance(transform, AveragingKernel) and transform.table is not None:
            yield (
                ("kernel", transform.table),
                lambda t=transform: _compare_kernel_table(t, project, ids),
            )


def _compare_kernel_table(
    transform: AveragingKernel, project: Project, ids: list[str]
) -> list[_Finding]:
    """Compare the receptor ids of a kernel table with the project's."""
    table = transform.table
    try:
        held = transform.table_receptors(project.directory)
    except FileNotFoundError as error:
        return [
            ("kernel_table", None, f"table {table!r} does not exist: {error.filename}")
        ]
    except ValueError as error:
        return [("kernel_table", None, str(error))]
    return [
        ("kernel_missing", rid, f"table {table!r} has no kernel for this receptor")
        for rid in ids
        if rid not in held
    ] + [
        (
            "kernel_unused",
            rid,
            f"table {table!r} has a kernel for a receptor the project does not have",
        )
        for rid in sorted(held - set(ids))
    ]


def _zsfc_checks(variant: Variant, rows: pd.DataFrame) -> Iterator[_Check]:
    """Yield the missing ``zsfc`` of a variant with receptors above sea level, which the run reads for them."""
    varsiwant = getattr(variant.transport, "varsiwant", None)
    if varsiwant is None or "zsfc" in varsiwant:
        return
    above_sea_level = rows[rows["altitude_ref"] == "msl"].drop_duplicates("receptor")
    if above_sea_level.empty:
        return
    n_multipoint = int((above_sea_level["kind"] == "multipoint").sum())
    near_field = variant.transport.hnf_plume
    weighting = variant.footprint is not None and any(
        isinstance(t, PressureWeighting) for t in variant.footprint.transforms
    )
    if not (near_field or n_multipoint or weighting):
        return
    key = (
        "zsfc",
        tuple(varsiwant),
        len(above_sea_level),
        n_multipoint,
        near_field,
        weighting,
    )
    yield (
        key,
        lambda: [
            (
                "zsfc",
                None,
                _zsfc_message(
                    varsiwant, len(above_sea_level), n_multipoint, near_field, weighting
                ),
            )
        ],
    )


def _zsfc_message(
    varsiwant: list[str],
    n_msl: int,
    n_multipoint: int,
    near_field: bool,
    weighting: bool,
) -> str:
    """Say what needs ``zsfc`` and the line that adds it to ``varsiwant``."""
    reasons = []
    if near_field:
        reasons.append("the near-field correction (hnf_plume) is skipped")
    if n_multipoint:
        reasons.append(
            "a multipoint receptor's release heights are matched by position, "
            "not height, which fails when its points are close together"
        )
    if weighting:
        reasons.append("the pressure_weighting transform fails every footprint")
    receptors = f"{n_msl} receptor is" if n_msl == 1 else f"{n_msl} receptors are"
    fix = ", ".join([*varsiwant, "zsfc"])
    return (
        f"zsfc is not in varsiwant, and {receptors} above sea level: "
        f"{'; '.join(reasons)}. Add it in config.yaml with varsiwant: [{fix}]. "
        "That changes the settings folder, so the particles are made again"
    )


__all__ = ["CHECK_COLUMNS", "check_project"]
