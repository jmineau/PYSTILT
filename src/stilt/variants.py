"""
Variants, resolved: everything a variant's results are made with, and the hashes that find them.

``config.yaml`` declares variants, and :class:`~stilt.config.ProjectConfig`
merges each with the defaults when it loads (a
:class:`~stilt.config.VariantConfig`). Two things are still unknown then,
because they need more than the file: the grid of a footprint given by a
geometry, which needs the geometry read, and the build of the transport
model. :func:`resolve` finds both and returns one :class:`Variant` per
simulation name. ``project.variants`` holds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from stilt.config import FootprintConfig, MetConfig, ProjectConfig
from stilt.footprint.targets import Mesh
from stilt.identity import (
    footprint_hash,
    footprint_settings,
    run_settings,
    settings_hash,
)
from stilt.transport import ModelInfo, TransportConfig, get_model


@dataclass(frozen=True)
class Variant:
    """
    One variant, resolved: its configs, its met, the model build, and its hashes.

    Get one from ``project.variants`` or ``sim.variant``. ``name`` is the
    name its simulations run under and ``group`` the name declared in
    ``config.yaml``. They differ only for realizations: ``hrrr-err`` with
    ``realizations: 3`` gives ``hrrr-err-0`` to ``hrrr-err-2``.

    Variants with equal :attr:`particles_hash` share one run of the
    transport model per receptor and differ only in their footprints. The
    hashes are computed once. Change a variant with
    :func:`dataclasses.replace`, which computes them again.

    Attributes
    ----------
    name : str
        Name its simulations run under.
    group : str
        Name as declared in ``config.yaml``.
    met : str
        Name of the met it runs with.
    met_config : MetConfig
        That met's config.
    transport : TransportConfig
        The transport model's config, such as a
        :class:`~stilt.transport.hysplit.HysplitConfig`.
    model : ModelInfo
        The transport model build that runs it.
    realization : int or None
        Realization number within an ensemble, or ``None`` for a single run.
    footprint : FootprintConfig or None
        Footprint config, with its grid, or ``None`` for particles only.
    geometry_hash : str or None
        Hash of the geometry the footprint grid was derived for
        (``Mesh.hash``), or ``None`` when the footprint has no geometry.
    """

    name: str
    group: str
    met: str
    met_config: MetConfig
    transport: TransportConfig
    model: ModelInfo
    realization: int | None = None
    footprint: FootprintConfig | None = None
    geometry_hash: str | None = None

    @cached_property
    def run_settings(self) -> dict[str, Any]:
        """The settings that identify this variant's particles, as ``_settings.yaml`` records them."""
        return run_settings(
            self.transport, self.met_config, self.model, self.realization
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


def resolve(config: ProjectConfig) -> dict[str, Variant]:
    """
    Return one :class:`Variant` per simulation name, in declared order.

    Each geometry is read once, however many variants use it, and the grid
    of a footprint given only by a geometry is derived from it
    (:meth:`stilt.Mesh.to_grid`). The transport model's version and data
    files are read once for each distinct build: the fields of its config
    that change no particle, such as ``exe_dir``.

    Parameters
    ----------
    config : ProjectConfig
        The project's config.
    """
    meshes: dict[str, Mesh] = {}
    builds: dict[tuple[str, str], ModelInfo] = {}
    variants: dict[str, Variant] = {}
    for spec in config.variant_configs.values():
        footprint, geometry_hash = spec.footprint, None
        if footprint is not None and footprint.geometry is not None:
            key = footprint.geometry.model_dump_json()
            if key not in meshes:
                meshes[key] = Mesh.from_spec(footprint.geometry)
            mesh = meshes[key]
            geometry_hash = mesh.hash
            if footprint.grid is None:
                grid = mesh.to_grid(cells_per_target=footprint.cells_per_target)
                footprint = footprint.model_copy(update={"grid": grid})
        where = spec.transport.model_dump_json(include=set(spec.transport.UNRECORDED))
        build = (spec.model, where)
        if build not in builds:
            model = get_model(spec.model)
            builds[build] = ModelInfo(
                name=model.name,
                version=model.version(spec.transport),
                data_files=model.data_files(spec.transport),
            )
        variants[spec.name] = Variant(
            name=spec.name,
            group=spec.group,
            met=spec.met,
            met_config=config.mets[spec.met],
            transport=spec.transport,
            model=builds[build],
            realization=spec.realization,
            footprint=footprint,
            geometry_hash=geometry_hash,
        )
    return variants


__all__ = ["Variant", "resolve"]
