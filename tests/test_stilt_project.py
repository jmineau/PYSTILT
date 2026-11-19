"""
stilt.Project and stilt.Simulation: the particle part's classes with STILT's reads added.

They open the same project as the particle part, change nothing they
inherit, and are what a STILT user holds.
"""

import inspect
from dataclasses import replace

import pytest
import yaml
from runtraj import project, simulation
from runtraj.spatial import Grid

import stilt
from stilt.project import STILT_R_HYSPLIT

from .fixtures.factories import blocks, make_met_config

_GRID = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)


def _own(cls: type) -> set[str]:
    """The public and private names a class sets in its own body, without Python's."""
    names = set(vars(cls))
    return {n for n in names if not (n.startswith("__") and n.endswith("__"))}


def _overridden(cls: type) -> set[str]:
    """The names *cls* sets that one of its bases already has."""
    return {n for n in _own(cls) if any(hasattr(b, n) for b in cls.__mro__[1:])}


def test_stilt_project_changes_only_which_simulation_it_returns_and_the_new_projects_values():
    # simulation returns stilt.Simulation; init writes STILT-R's HYSPLIT values.
    assert _overridden(stilt.Project) == {"simulation", "init"}


def test_stilt_simulation_changes_nothing_it_inherits():
    assert _overridden(stilt.Simulation) == set()


def test_the_subclasses_are_the_ones_stilt_exports():
    assert issubclass(stilt.Project, project.Project)
    assert issubclass(stilt.Simulation, simulation.Simulation)
    # The particle part has no hook a subclass sets.
    assert not hasattr(project.Project, "simulation_class")


def test_the_particle_part_has_none_of_stilts_reads():
    for name in ("footprints", "jacobian", "transport_error"):
        assert not hasattr(project.Project, name)
    for name in (
        "footprint",
        "footprint_path",
        "has_footprint",
        "calc_footprint",
        "background",
        "transport_error",
        "plot",
    ):
        assert not hasattr(simulation.Simulation, name)


def test_stilt_project_simulation_says_it_returns_stilts_simulation():
    # A real annotation, so editors and pyrefly see the reads.
    hints = inspect.get_annotations(stilt.Project.simulation, eval_str=True)
    assert hints["return"] is stilt.Simulation


@pytest.fixture
def made(tmp_path, point_release):
    """A STILT project with one release and a footprint grid."""
    config = blocks(
        mets={"hrrr": make_met_config(tmp_path / "met")},
        variants={"hrrr": {}},
        grid=_GRID,
    )
    return stilt.Project.init(tmp_path / "proj", [point_release], **config)


def test_both_classes_open_the_same_project(made, point_release):
    particles_only = project.Project(made.directory)

    assert particles_only.config == made.config
    assert particles_only.simulations.equals(made.simulations)
    assert particles_only.status().equals(made.status())


def test_each_project_returns_its_own_simulation_of_the_same_files(made, point_release):
    sim = made.simulation(point_release.id, "hrrr")
    plain = project.Project(made.directory).simulation(point_release.id, "hrrr")

    assert type(sim) is stilt.Simulation
    assert type(plain) is simulation.Simulation
    assert sim.id == plain.id
    assert sim.path("footprint") == plain.path("footprint") == sim.footprint_path
    assert sim.has("footprint") is plain.has("footprint") is sim.has_footprint


def test_stilt_project_simulation_refuses_what_the_particle_part_refuses(
    made, point_release
):
    with pytest.raises(KeyError, match="No variant 'nope'"):
        made.simulation(point_release.id, "nope")
    with pytest.raises(ValueError, match="realizations"):
        made.simulation(point_release.id, "hrrr", 3)


def test_one_simulation_is_equal_whichever_class_holds_it(made, point_release):
    # The workers build the particle class; stilt.Project the subclass.
    sim = made.simulation(point_release.id, "hrrr")
    plain = project.Project(made.directory).simulation(point_release.id, "hrrr")
    other = made.simulation(point_release.id, "hrrr")

    assert sim == plain and plain == sim and sim == other
    assert hash(sim) == hash(plain)
    assert len({sim, plain}) == 1
    assert plain != replace(plain, directory=None)


def test_the_particle_part_writes_the_settings_as_given(tmp_path):
    made = project.Project.init(
        tmp_path / "proj",
        mets={"hrrr": make_met_config(tmp_path / "met")},
        models={"hysplit": {"n_hours": -24}},
        variants={"hrrr": {}},
    )

    written = yaml.safe_load(made.config_path.read_text())
    assert written["models"] == {"hysplit": {"n_hours": -24}}
    assert type(made) is project.Project


def test_stilt_project_init_adds_stilt_rs_values_to_the_same_settings(tmp_path):
    made = stilt.Project.init(
        tmp_path / "proj",
        mets={"hrrr": make_met_config(tmp_path / "met")},
        models={"hysplit": {"n_hours": -24}},
        variants={"hrrr": {}},
    )

    written = yaml.safe_load(made.config_path.read_text())
    assert written["models"] == {"hysplit": {"n_hours": -24, **STILT_R_HYSPLIT}}
    assert type(made) is stilt.Project
