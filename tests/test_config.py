"""Tests for runtraj.config - models and YAML roundtrip."""

import re
import textwrap

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from runtraj.config import ProjectConfig
from runtraj.exceptions import MissingClass
from runtraj.spatial import Grid

from stilt.footprint import FOOTPRINT_STAGE
from stilt.footprint.config import (
    FootprintConfig,
    footprint_settings,
    read_footprint_settings,
)
from stilt.footprint.targets import Mesh
from stilt.transforms import AveragingKernel, FirstOrderLifetime, PressureWeighting
from tests.fixtures.factories import blocks

from .fixtures.factories import make_met_config, write_arl_file


@pytest.fixture(autouse=True)
def _met_header(tmp_path):
    """Put an ARL file in each test's met folder: resolving a project reads its met's source there."""
    write_arl_file(tmp_path / "met" / "header.arl")


class ScaleFoot(BaseModel):
    """A user transform, addressed in YAML by its dotted import path."""

    model_config = ConfigDict(frozen=True)

    factor: float = 1.0

    def apply(self, particles, release=None, directory=None):
        out = particles.copy()
        out["foot"] = out["foot"] * self.factor
        return out


SCALE_FOOT_KIND = f"{__name__}.ScaleFoot"

# ---------------------------------------------------------------------------
# winderrtf
# ---------------------------------------------------------------------------
# ProjectConfig - the blocks
# ---------------------------------------------------------------------------


def test_a_config_is_blocks(tmp_path, grid):
    """Mets, models, stages and variants each map a name to a block; a variant overrides its model's."""
    cfg = ProjectConfig(
        mets={"hrrr": make_met_config(directory=tmp_path / "met")},
        models={"hysplit": {"n_hours": -24, "numpar": 100, "seed": 42, "krand": 2}},
        stages={
            "footprint": {"class": FOOTPRINT_STAGE, "grid": grid, "smooth_factor": 0.5}
        },
        variants={"hrrr": {}, "hrrr-long": {"n_hours": -72}},
    )
    variants = cfg.resolve()
    assert variants["hrrr"].transport.n_hours == -24
    assert variants["hrrr"].transport.seed == 42
    assert variants["hrrr-long"].transport.n_hours == -72
    assert variants["hrrr-long"].transport.numpar == 100
    assert variants["hrrr"].stages.get("footprint").grid == grid
    assert variants["hrrr"].stages.get("footprint").smooth_factor == 0.5
    assert variants["hrrr-long"].stages.get("footprint") == variants["hrrr"].stages.get(
        "footprint"
    )


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (
            {"n_hours": -24},
            "config: 'n_hours' is a hysplit setting. Write it under models: hysplit:.",
        ),
        (
            {"model": "hysplit"},
            "config: 'model' is no longer a key. Name each transport model as a "
            "block under models:, such as models: {hysplit: {}}.",
        ),
    ],
    ids=["model setting", "model"],
)
def test_a_setting_at_the_top_is_refused_naming_its_block(tmp_path, setting, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        ProjectConfig(mets=_met_config(tmp_path), variants={"hrrr": {}}, **setting)


def test_a_setting_in_another_block_is_refused_naming_its_own(tmp_path, grid):
    mets = _met_config(tmp_path)
    stage = {"class": FOOTPRINT_STAGE, "grid": grid}
    with pytest.raises(
        ValueError,
        match="models: hysplit: 'grid' is a footprint setting. Write it under stages:",
    ):
        ProjectConfig(
            mets=mets,
            models={"hysplit": {"grid": grid}},
            stages={"footprint": stage},
            variants={"hrrr": {}},
        )
    with pytest.raises(
        ValueError,
        match="stages: footprint: 'numpar' is a hysplit setting. "
        "Write it under models: hysplit:.",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"footprint": {**stage, "numpar": 5}},
            variants={"hrrr": {}},
        )
    with pytest.raises(
        ValueError,
        match="Variant 'hrrr': 'smooth_factor' is a footprint setting. "
        "Write it under the variant's stages: footprint:.",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"footprint": stage},
            variants={"hrrr": {"smooth_factor": 2}},
        )
    with pytest.raises(
        ValueError,
        match="Variant 'hrrr', stages: footprint: 'class' is the stage itself",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"footprint": stage},
            variants={"hrrr": {"stages": {"footprint": {"class": FOOTPRINT_STAGE}}}},
        )


def test_a_stage_block_names_the_footprint_stage_as_its_class(tmp_path, grid):
    mets = _met_config(tmp_path)
    with pytest.raises(
        ValueError,
        match="stages: footprint names no class. Write class: with the import path",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"footprint": {"grid": grid}},
            variants={"hrrr": {}},
        )
    with pytest.raises(
        MissingClass,
        match="config.yaml names mypkg.WindRose, but mypkg is not installed here",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"wind": {"class": "mypkg.WindRose"}},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="stages: wind: A stage is named by the"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"wind": {"class": "WindRose"}},
            variants={"hrrr": {}},
        )


def test_a_stage_block_is_named_for_its_stage(tmp_path, grid):
    """One name for the block, the output folder and the status column: the stage's own."""
    stage = {"class": FOOTPRINT_STAGE, "grid": grid}
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={
            "hrrr": {},
            "smooth": {"stages": {"footprint": {"smooth_factor": 2}}},
            "particles": {"stages": {"footprint": None}},
        },
    )
    variants = config.resolve()
    assert variants["hrrr"].stages["footprint"].smooth_factor == 1
    assert variants["smooth"].stages["footprint"].smooth_factor == 2
    assert variants["smooth"].stages["footprint"].grid == grid
    assert variants["particles"].stages == {}
    for name in ("foot", "fine", "Not_Plain"):
        with pytest.raises(
            ValueError,
            match=f"stages: {name}: stilt.footprint.FootprintStage is the footprint "
            "stage, and a stage's block is named for its stage. Write it as "
            "stages: footprint:.",
        ):
            ProjectConfig(
                models={"hysplit": {"n_hours": -24}},
                mets=_met_config(tmp_path),
                stages={name: stage},
                variants={"hrrr": {}},
            )
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr' names the stage 'foot', which stages: does "
        r"not have \(it has \['footprint'\]\)",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            stages={"footprint": stage},
            variants={"hrrr": {"stages": {"foot": None}}},
        )


def test_a_variants_stages_is_a_mapping(tmp_path, grid):
    """A list of stage names is refused: a variant gets every stage, and says only what it changes."""
    with pytest.raises(
        ValueError,
        match="Variant 'hrrr': stages must map a stage's name to the settings it changes",
    ):
        ProjectConfig(
            mets=_met_config(tmp_path),
            models={"hysplit": {"n_hours": -24}},
            stages={"footprint": {"class": FOOTPRINT_STAGE, "grid": grid}},
            variants={"hrrr": {"stages": ["footprint"]}},
        )


def test_the_footprint_stage_is_at_the_path_a_config_names():
    import importlib

    module, _, name = FOOTPRINT_STAGE.rpartition(".")
    stage = getattr(importlib.import_module(module), name)
    assert stage.name == "footprint"
    assert stage.config_class is FootprintConfig


def _default_footprint(cfg):
    """The footprint settings of the first variant, which inherits the defaults."""
    return next(iter(cfg.resolve().values())).stages.get("footprint")


def test_without_a_stages_block_a_project_makes_particles_only(tmp_path, grid):
    mc = make_met_config(directory=tmp_path / "met")
    cfg = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets={"hrrr": mc},
        stages={"footprint": {"class": FOOTPRINT_STAGE, "grid": grid}},
        variants={"hrrr": {}},
    )
    assert _default_footprint(cfg).grid == grid
    assert (
        _default_footprint(
            ProjectConfig(
                models={"hysplit": {"n_hours": -24}},
                mets={"hrrr": mc},
                variants={"hrrr": {}},
            )
        )
        is None
    )


# ---------------------------------------------------------------------------
# ProjectConfig YAML roundtrip
# ---------------------------------------------------------------------------


def test_model_config_yaml_roundtrip_with_footprint(tmp_path, grid):
    """The footprint settings survive a to_yaml/from_yaml roundtrip."""
    cfg = ProjectConfig(
        **blocks(
            mets={
                "hrrr": make_met_config(
                    directory=tmp_path / "met",
                )
            },
            grid=grid,
            tres="15min",
            variants={"hrrr": {}},
        )
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert _default_footprint(loaded) is not None
    assert (
        _default_footprint(loaded).model_dump() == _default_footprint(cfg).model_dump()
    )


@pytest.mark.parametrize("where", ["block", "variant"])
def test_time_integrate_is_refused_saying_what_to_do_instead(tmp_path, grid, where):
    """The footprint's config class refuses the setting it once had, wherever its settings are written."""
    stage = {"class": FOOTPRINT_STAGE, "grid": grid}
    own = {}
    if where == "block":
        stage["time_integrate"] = True
    else:
        own = {"stages": {"footprint": {"time_integrate": True}}}
    with pytest.raises(ValueError, match=r"time_integrate was removed.*sum\('time'\)"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            models={"hysplit": {"n_hours": -24}},
            stages={"footprint": stage},
            variants={"hrrr": own},
        )


@pytest.mark.parametrize("where", ["block", "variant"])
def test_veght_is_hysplits_and_goes_to_setup_cfg(tmp_path, where):
    from runtraj.models.hysplit.driver import setup_entries

    block = {"n_hours": -24, "veght": 0.3} if where == "block" else {"n_hours": -24}
    variant = {} if where == "block" else {"veght": 0.3}
    config = ProjectConfig(
        mets=_met_config(tmp_path),
        models={"hysplit": block},
        variants={"hrrr": variant},
    )
    entries = setup_entries(config.resolve()["hrrr"].transport)
    assert entries["veght"] == 0.3
    assert "footprint_depth" not in entries


@pytest.mark.parametrize("setting", ["footprint_depth", "pbl_height", "pbl_height_min"])
def test_a_foot_setting_written_for_the_model_is_refused_naming_its_stage(
    tmp_path, setting, grid
):
    with pytest.raises(
        ValueError, match=rf"{setting}' is .*a footprint setting.*stages"
    ):
        ProjectConfig(
            mets=_met_config(tmp_path),
            models={"hysplit": {"n_hours": -24, setting: 100.0}},
            stages={"footprint": {"class": FOOTPRINT_STAGE, "grid": grid}},
            variants={"hrrr": {}},
        )


def test_veght_written_for_the_footprint_is_refused_naming_the_model(tmp_path, grid):
    with pytest.raises(ValueError, match=r"'veght' is a hysplit setting.*models"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            models={"hysplit": {"n_hours": -24}},
            stages={
                "footprint": {"class": FOOTPRINT_STAGE, "grid": grid, "veght": 0.3}
            },
            variants={"hrrr": {}},
        )


def test_a_footprint_depth_unlike_veght_is_refused_for_hysplits_foot(tmp_path, grid):
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr', stages: footprint: footprint_depth is 0.3.*veght 0.5.*pbl_height",
    ):
        ProjectConfig(
            mets=_met_config(tmp_path),
            models={"hysplit": {"n_hours": -24}},
            stages={
                "footprint": {
                    "class": FOOTPRINT_STAGE,
                    "grid": grid,
                    "footprint_depth": 0.3,
                }
            },
            variants={"hrrr": {}},
        ).resolve()


def test_variants_that_differ_in_the_footprints_foot_share_the_run(tmp_path, grid):
    config = ProjectConfig(
        mets=_met_config(tmp_path),
        models={"hysplit": {"n_hours": -24, "veght": 0.3}},
        stages={"footprint": {"class": FOOTPRINT_STAGE, "grid": grid}},
        variants={
            "hrrr": {},
            "rich": {
                "stages": {
                    "footprint": {"pbl_height": "richardson", "footprint_depth": 0.5}
                }
            },
            "fixed": {"stages": {"footprint": {"pbl_height": 800.0}}},
        },
    )
    variants = config.resolve()

    assert len({v.particles_hash for v in variants.values()}) == 1
    assert len({v.stage_hashes["footprint"] for v in variants.values()}) == 3
    # On HYSPLIT's foot the footprint takes the run's veght; computing its
    # own, the footprint's footprint_depth.
    depths = {n: v.stages["footprint"].footprint_depth for n, v in variants.items()}
    assert depths == {"hrrr": 0.3, "rich": 0.5, "fixed": 0.5}


def test_hnf_plume_needs_its_columns_in_varsiwant_only_with_a_footprint(tmp_path, grid):
    varsiwant = ["time", "indx", "long", "lati", "zagl", "foot"]
    # The stage finds that it does not fit the model when the variants are resolved.
    config = ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            grid=grid,
            varsiwant=varsiwant,
            variants={"hrrr": {}},
        )
    )
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr', stages: footprint: the near-field correction "
        r"\(hnf_plume\) reads",
    ):
        config.resolve()
    # Without a footprint, or with the correction off, nothing reads them.
    ProjectConfig(
        **blocks(mets=_met_config(tmp_path), varsiwant=varsiwant, variants={"hrrr": {}})
    ).resolve()
    ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            grid=grid,
            varsiwant=varsiwant,
            hnf_plume=False,
            variants={"hrrr": {}},
        )
    ).resolve()


def test_a_variant_turns_the_footprint_off_with_null(tmp_path, grid):
    stage = {
        "class": FOOTPRINT_STAGE,
        "grid": grid,
        "hnf_plume": False,
        "smooth_factor": 0.5,
    }
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={"hrrr": {}, "hrrr-err": {"stages": {"footprint": None}}},
    )
    variants = config.resolve()
    assert variants["hrrr"].stages.get("footprint") is not None
    assert variants["hrrr-err"].stages.get("footprint") is None
    # A footprint with its grid taken away has nothing to grid on: say how to turn it off.
    without_grid = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={
            "hrrr": {},
            "hrrr-err": {"stages": {"footprint": {"grid": None}}},
        },
    )
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr-err', stages: footprint: the footprint has no grid. "
        r"[\s\S]*set the stage to null under the variant's stages: to make "
        "particles only",
    ):
        without_grid.resolve()


def test_a_stored_record_with_time_integrate_true_is_refused():
    """A footprint summed over time is not read back as an hourly one."""
    with pytest.raises(ValueError, match="time_integrate was removed"):
        FootprintConfig.model_validate({"time_integrate": True})
    assert FootprintConfig.model_validate({"time_integrate": False}).tres == "1h"


def _met_config(tmp_path):
    return {
        "hrrr": make_met_config(
            directory=tmp_path / "met",
        )
    }


def test_model_config_yaml_roundtrip_with_footprint_transforms(tmp_path, grid):
    given = [
        AveragingKernel(
            levels=[0.0, 1000.0], values=[0.2, 0.8], coordinate="release_height"
        ),
        PressureWeighting(),
        FirstOrderLifetime(lifetime_hours=4.0),
    ]
    cfg = ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            grid=grid,
            transforms=given,
            variants={"hrrr": {}},
        )
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.resolve()["hrrr"].stages.get("footprint").transforms
    assert len(transforms) == 3
    assert transforms == given
    assert isinstance(transforms[0], AveragingKernel)
    assert isinstance(transforms[1], PressureWeighting)
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert loaded.resolve()["hrrr"].stages.get("footprint").transforms == given


def test_model_config_yaml_roundtrip_with_user_transform(tmp_path, grid):
    cfg = ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            grid=grid,
            transforms=[ScaleFoot(factor=2.5)],
            variants={"hrrr": {}},
        )
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)

    text = path.read_text()
    assert f"kind: {SCALE_FOOT_KIND}" in text
    assert "factor: 2.5" in text

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.resolve()["hrrr"].stages.get("footprint").transforms
    assert transforms == [ScaleFoot(factor=2.5)]


def test_model_config_variant_grid_in_yaml(tmp_path):
    """A variant may carry its own grid; the others inherit the default."""
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        models:
          hysplit:
            n_hours: -24
            numpar: 100
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.01
              yres: 0.01
        variants:
          hrrr: {{}}
          coarse:
            stages:
              footprint:
                grid:
                  xres: 0.05
                  yres: 0.05
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    variants = ProjectConfig.from_yaml(path).resolve()
    assert variants["hrrr"].stages.get("footprint").grid.xres == 0.01
    assert variants["coarse"].stages.get("footprint").grid.xres == 0.05
    assert variants["coarse"].stages.get("footprint").grid.xmin == -114.0


def test_model_config_inline_grid_in_yaml(tmp_path):
    """The default grid is a mapping under ``grid``."""
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        models:
          hysplit:
            n_hours: -24
            numpar: 100
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
            smooth_factor: 0.75
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    footprint = ProjectConfig.from_yaml(path).resolve()["hrrr"].stages.get("footprint")
    assert footprint.grid.xres == 0.05
    assert footprint.grid.xmin == -114.0
    assert footprint.smooth_factor == 0.75


def test_model_config_null_grid_means_trajectory_only(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        models:
          hysplit:
            n_hours: -24
            numpar: 100
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
        variants:
          hrrr: {{}}
          traj:
            stages:
              footprint: null
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    variants = ProjectConfig.from_yaml(path).resolve()
    assert variants["hrrr"].stages.get("footprint") is not None
    assert variants["traj"].stages.get("footprint") is None


def test_model_config_loads_footprint_transforms_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        models:
          hysplit:
            n_hours: -24
            numpar: 100
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
            transforms:
              - kind: averaging_kernel
                levels: [0.0, 1000.0]
                values: [0.3, 0.7]
                coordinate: release_height
              - kind: pressure_weighting
              - kind: first_order_lifetime
                lifetime_hours: 3.0
              - kind: {SCALE_FOOT_KIND}
                factor: 0.5
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.resolve()["hrrr"].stages.get("footprint").transforms

    assert len(transforms) == 4
    assert isinstance(transforms[0], AveragingKernel)
    assert transforms[0].levels == [0.0, 1000.0]
    assert transforms[0].values == [0.3, 0.7]
    assert transforms[0].coordinate == "release_height"
    assert isinstance(transforms[1], PressureWeighting)
    assert transforms[1].surface_pressure is None
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert transforms[2].lifetime_hours == pytest.approx(3.0)
    assert isinstance(transforms[3], ScaleFoot)
    assert transforms[3].factor == pytest.approx(0.5)


def test_model_config_rejects_unimportable_transform_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        models:
          hysplit:
            n_hours: -24
            numpar: 100
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
            transforms:
              - kind: no_such_pkg_for_stilt_tests.transforms.MyKernel
                levels: [0.0, 1000.0]
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ImportError, match="could not be imported"):
        ProjectConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# FootprintConfig.geometry
# ---------------------------------------------------------------------------


def _resolved_footprint(tmp_path, **footprint) -> FootprintConfig:
    """The resolved footprint of a one-met project with these footprint fields."""
    variant = (
        ProjectConfig(
            **blocks(
                mets=_met_config(tmp_path), **{"variants": {"hrrr": {}}, **footprint}
            )
        )
    ).resolve()["hrrr"]
    assert variant.stages.get("footprint") is not None
    return variant.stages.get("footprint")


def test_footprint_grid_is_derived_from_windows_geometry(tmp_path):
    spec = {
        "kind": "windows",
        "coords": [(-111.97, 40.515), (-112.015, 40.779)],
        "size": 0.01,
        "ids": ["landfill", "wwtp"],
    }
    assert FootprintConfig(geometry=spec).grid is None  # derived when resolved
    fc = _resolved_footprint(tmp_path, geometry=spec)
    assert fc.geometry is not None and fc.geometry.kind == "windows"
    assert fc.grid.xres == fc.grid.yres == pytest.approx(0.002)  # 0.01 / 4 -> 0.002
    assert fc.grid.xmin <= -112.02 and fc.grid.ymax >= 40.784
    assert Mesh.from_spec(fc.geometry).ids == ("landfill", "wwtp")


def test_a_negative_smooth_factor_is_refused():
    """A negative factor later failed with a bare ZeroDivisionError (#440); 0 turns smoothing off."""
    assert FootprintConfig(smooth_factor=0).smooth_factor == 0
    with pytest.raises(ValidationError, match="smooth_factor"):
        FootprintConfig(smooth_factor=-1)


def test_cells_per_target_and_an_explicit_grid_wins(tmp_path):
    spec = {"kind": "windows", "coords": [(0.5, 0.5)], "size": 0.1}
    fc = _resolved_footprint(tmp_path, geometry=spec, cells_per_target=10)
    assert fc.grid.xres == pytest.approx(0.01)
    explicit = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.05, yres=0.05)
    fc2 = _resolved_footprint(tmp_path, grid=explicit, geometry=spec)
    assert fc2.grid == explicit and fc2.geometry is not None


def test_footprint_needs_a_grid(point_release):
    import pandas as pd

    from stilt.footprint import calc_footprint

    assert FootprintConfig().grid is None
    with pytest.raises(ValueError, match="grid"):
        calc_footprint(pd.DataFrame(), point_release, None)


def test_footprint_h3_geometry(tmp_path):
    pytest.importorskip("h3")
    fc = _resolved_footprint(
        tmp_path,
        geometry={
            "kind": "h3",
            "resolution": 8,
            "bounds": {"xmin": -112.0, "xmax": -111.8, "ymin": 40.6, "ymax": 40.8},
        },
    )
    assert fc.grid.xres <= 0.0025  # res-8 hexagons are ~0.5 km across
    assert len(Mesh.from_spec(fc.geometry)) > 50


def test_model_config_yaml_roundtrip_with_geometry(tmp_path):
    gpd = pytest.importorskip("geopandas")
    import shapely

    gdf = gpd.GeoDataFrame(
        {"NAME": ["a", "b"]},
        geometry=[
            shapely.box(-112.0, 40.5, -111.9, 40.6),
            shapely.box(-111.9, 40.5, -111.8, 40.6),
        ],
        crs="EPSG:4326",
    )
    shp = tmp_path / "cells.geojson"
    gdf.to_file(shp, driver="GeoJSON")

    config = ProjectConfig(
        **blocks(
            mets={
                "hrrr": {
                    "directory": tmp_path / "met",
                }
            },
            n_hours=-6,
            numpar=100,
            geometry={"kind": "file", "path": str(shp), "ids": "NAME"},
            variants={"hrrr": {}},
        )
    )
    fc = next(iter(config.resolve().values())).stages.get("footprint")
    assert fc is not None
    assert fc.grid.xres == pytest.approx(0.02)  # 0.1 / 4 -> 0.025 -> 0.02
    assert fc.geometry is not None and fc.geometry.kind == "file"

    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    lfc = next(iter(loaded.resolve().values())).stages.get("footprint")
    assert lfc is not None
    assert lfc.grid == fc.grid
    assert lfc.geometry == fc.geometry
    assert Mesh.from_spec(lfc.geometry).ids == ("a", "b")


def test_file_geometry_spec_layer_and_where(tmp_path):
    gpd = pytest.importorskip("geopandas")
    import shapely

    from stilt.footprint.config import FileGeometrySpec

    gdf = gpd.GeoDataFrame(
        {"NAME": ["a", "b", "c"], "KEEP": [1, 1, 0]},
        geometry=[shapely.box(i, 0, i + 1, 1) for i in range(3)],
        crs="EPSG:4326",
    )
    gpkg = tmp_path / "cells.gpkg"
    gdf.to_file(gpkg, layer="cells", driver="GPKG")
    gdf.iloc[:1].to_file(gpkg, layer="other", driver="GPKG")

    spec = FileGeometrySpec(path=str(gpkg), ids="NAME", layer="cells", where="KEEP=1")
    assert Mesh.from_spec(spec).ids == ("a", "b")
    assert Mesh.from_spec(FileGeometrySpec(path=str(gpkg), layer="other")).ids == ("0",)


# -- variants --------------------------------------------------------------------


def _variant_config(tmp_path, **kwargs):
    return ProjectConfig(
        **blocks(mets=_met_config(tmp_path), **{"variants": {"hrrr": {}}, **kwargs})
    )


def test_a_footprint_without_a_grid_is_an_error(tmp_path, grid):
    """A footprint with nothing to grid on is refused, for the variant that has none."""
    stage = {"class": FOOTPRINT_STAGE, "smooth_factor": 2}
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={"hrrr": {}},
    )
    with pytest.raises(
        ValueError, match="Variant 'hrrr', stages: footprint: the footprint has no grid"
    ):
        config.resolve()
    # The block may leave the grid to each variant.
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={"hrrr": {"stages": {"footprint": {"grid": grid}}}},
    )
    footprint = config.resolve()["hrrr"].stages.get("footprint")
    assert footprint.grid == grid and footprint.smooth_factor == 2


def test_variant_overrides_merge_onto_the_defaults(tmp_path):
    cfg = _variant_config(
        tmp_path, numpar=50, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    variants = cfg.resolve()
    assert variants["zi08"].transport.numpar == 50
    assert variants["zi08"].transport.ziscale == 0.8
    assert variants["hrrr"].transport.ziscale == 1.0
    zi08 = variants["zi08"].transport.model_dump()
    hrrr = variants["hrrr"].transport.model_dump()
    assert sorted(k for k in zi08 if zi08[k] != hrrr[k]) == ["ziscale"]
    assert variants["zi08"].stages.get("footprint") == variants["hrrr"].stages.get(
        "footprint"
    )


def test_variant_grid_override_merges_field_by_field(tmp_path):
    grid = {
        "xmin": -114,
        "xmax": -111,
        "ymin": 39,
        "ymax": 42,
        "xres": 0.01,
        "yres": 0.01,
    }
    cfg = _variant_config(
        tmp_path,
        grid=grid,
        variants={
            "hrrr": {},
            "coarse": {"grid": {"xres": 0.1, "yres": 0.1}},
            "none": {"grid": None},
        },
    )
    variants = cfg.resolve()
    assert variants["coarse"].stages.get("footprint") is not None
    assert variants["coarse"].stages.get("footprint").grid.xres == 0.1
    assert variants["coarse"].stages.get("footprint").grid.xmin == -114
    assert variants["none"].stages.get("footprint") is None


def _all_keys(node) -> set[str]:
    """Every mapping key in a loaded YAML document, at any depth."""
    if isinstance(node, dict):
        return set(node).union(*(_all_keys(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_all_keys(v) for v in node))
    return set()


def test_to_yaml_writes_only_what_was_set(tmp_path, grid):
    from stilt.transforms import FirstOrderLifetime

    cfg = _variant_config(
        tmp_path,
        numpar=50,
        grid=grid,
        variants={
            "hrrr": {},
            "zi08": {"ziscale": 0.8},
            "decay": {
                "transforms": [FirstOrderLifetime(lifetime_hours=1.0)],
            },
        },
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    text = path.read_text()
    assert "numpar: 50" in text and "zi08" in text
    # Check the keys, not the text: the met directory is under tmp_path,
    # whose name can hold any of these words.
    keys = _all_keys(yaml.safe_load(text))
    assert "capemin" not in keys and "seed" not in keys  # defaults stay out
    assert "maxpar" not in keys and "n_min" not in keys  # at every level
    assert text.startswith("mets:")  # inputs first, then the settings
    assert "kind: first_order_lifetime" in text  # nested objects are written in full
    loaded = ProjectConfig.from_yaml(path).resolve()
    assert loaded["zi08"].transport.ziscale == 0.8
    assert loaded["decay"].stages.get("footprint").transforms[0].lifetime_hours == 1.0


def test_variants_that_differ_only_in_footprint_fields_keep_the_transport(
    tmp_path, grid
):
    cfg = _variant_config(
        tmp_path,
        grid=grid,
        variants={"hrrr": {"ziscale": 0.8}, "s2": {"ziscale": 0.8, "smooth_factor": 2}},
    )
    variants = cfg.resolve()
    s2 = variants["s2"]
    assert s2.met == "hrrr"
    assert s2.transport.ziscale == 0.8
    assert (
        s2.stages.get("footprint") is not None
        and s2.stages.get("footprint").smooth_factor == 2
    )
    assert s2.transport == variants["hrrr"].transport
    assert s2.particles_hash == variants["hrrr"].particles_hash


def test_variants_survive_a_yaml_roundtrip_as_written(tmp_path, grid):
    declared = {
        "hrrr": {},
        "zi08": {"ziscale": 0.8},
        "s2": {"stages": {"footprint": {"smooth_factor": 2}}},
    }
    cfg = _variant_config(tmp_path, grid=grid, variants=declared)
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.variants == declared
    assert list(loaded.resolve()) == ["hrrr", "zi08", "s2"]


_WINDOWS = {
    "kind": "windows",
    "coords": [[-111.97, 40.515], [-112.015, 40.779]],
    "size": 0.01,
    "ids": ["landfill", "wwtp"],
}


@pytest.mark.parametrize("defaults", ["grid", "geometry"])
def test_variant_geometry_derives_its_own_grid_and_hash(tmp_path, grid, defaults):
    """A variant's geometry is not rastered on the inherited grid (#42)."""
    other = {"kind": "windows", "coords": [[-111.5, 40.2]], "size": 0.05, "ids": ["c"]}
    base = {"grid": grid} if defaults == "grid" else {"geometry": other}
    cfg = _variant_config(
        tmp_path,
        **base,
        variants={
            "hrrr": {},
            "src": {"geometry": _WINDOWS},
            "src-run": {"geometry": _WINDOWS},
        },
    )
    variants = cfg.resolve()
    mesh = Mesh.from_spec(variants["src"].stages.get("footprint").geometry)
    expected = mesh.to_grid()
    for name in ("src", "src-run"):
        assert variants[name].stages.get("footprint").grid == expected
        assert variants[name].stages["footprint"].geometry_hash == mesh.hash
    assert variants["hrrr"].stages.get("footprint").grid != expected


def _count_builds(monkeypatch):
    """Count the geometries read (``Mesh.from_spec``)."""
    calls = []
    build = Mesh.from_spec.__func__

    def counting(cls, spec):
        calls.append(spec)
        return build(cls, spec)

    monkeypatch.setattr(Mesh, "from_spec", classmethod(counting))
    return calls


def test_each_geometry_is_built_once_and_not_on_load(tmp_path, monkeypatch):
    """Loading a config reads no geometry; resolving builds each one once (#64)."""
    calls = _count_builds(monkeypatch)
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(f"""
        mets:
          hrrr: {{directory: {tmp_path}/met}}
        models:
          hysplit: {{n_hours: -24}}
        stages:
          footprint:
            class: stilt.footprint.FootprintStage
            geometry: {{kind: windows, coords: [[-111.5, 40.2]], size: 0.05}}
        variants:
          hrrr: {{}}
          np50: {{numpar: 50}}
          err: {{realizations: 2, seed: 1, krand: 2}}
          src:
            stages:
              footprint:
                geometry: {{kind: windows, coords: [[-111.97, 40.515]], size: 0.01}}
        """)
    )
    cfg = ProjectConfig.from_yaml(tmp_path / "config.yaml")
    assert calls == []

    variants = cfg.resolve()
    assert len(calls) == 2  # the default geometry and the one of "src"
    inherited = [variants[n].stages.get("footprint") for n in ("hrrr", "np50", "err")]
    assert all(f is not None and f.grid == inherited[0].grid for f in inherited)
    assert len(calls) == 2


def test_config_loads_without_its_geometry_file(tmp_path):
    """A worker can load the config even where the geometry file is not readable."""
    cfg = _variant_config(
        tmp_path, geometry={"kind": "file", "path": str(tmp_path / "missing.shp")}
    )
    assert "grid" not in cfg.stages["footprint"]
    with pytest.raises(Exception, match="missing.shp"):
        cfg.resolve()


def test_stored_footprint_settings_never_read_the_geometry(monkeypatch):
    """A footprint folder's settings, with their grid and geometry hash, read back alone."""
    calls = _count_builds(monkeypatch)
    explicit = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.05, yres=0.05)
    config = FootprintConfig(
        grid=explicit,
        geometry={"kind": "windows", "coords": [(0.5, 0.5)], "size": 0.1},
    )
    config = config.model_copy(update={"geometry_hash": "deadbeef00"})
    stored = footprint_settings(config)
    assert read_footprint_settings(stored) == config
    assert calls == []


def test_to_yaml_does_not_write_the_derived_grid(tmp_path):
    cfg = _variant_config(tmp_path, geometry=_WINDOWS)
    foot = _default_footprint(cfg)
    assert foot is not None and foot.grid is not None
    text = cfg.to_yaml()
    assert "geometry:" in text
    assert "grid:" not in text and "geometry_hash" not in text


def test_partial_grid_over_a_derived_grid_is_an_error(tmp_path):
    with pytest.raises(
        ValueError,
        match="Variant 'coarse', stages: footprint: it changes part of a grid",
    ):
        _variant_config(
            tmp_path,
            geometry=_WINDOWS,
            variants={"hrrr": {}, "coarse": {"grid": {"xres": 0.1, "yres": 0.1}}},
        )


# ---------------------------------------------------------------------------
# Variants are declared; unknown keys name the nearest setting (#150)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"numpr": 10},
            "config: 'numpr' is not a setting. Did you mean 'numpar'? "
            "It is a hysplit setting. Write it under models: hysplit:.",
        ),
        ({"varients": {}}, "Did you mean 'variants', a config key?"),
        (
            {"models": {"hysplit": {"numpr": 10}}},
            "models: hysplit: 'numpr' is not a setting. Did you mean 'numpar', a hysplit setting?",
        ),
        (
            {"stages": {"footprint": {"class": FOOTPRINT_STAGE, "smooth_factr": 1}}},
            "stages: footprint: 'smooth_factr' is not a setting. Did you mean 'smooth_factor', a footprint setting?",
        ),
        (
            {
                "stages": {"footprint": {"class": FOOTPRINT_STAGE}},
                "variants": {"hrrr": {"stages": {"footprint": {"tress": "1h"}}}},
            },
            "Variant 'hrrr', stages: footprint: 'tress' is not a setting. Did you mean 'tres', a footprint setting?",
        ),
        (
            {"variants": {"hrrr": {"zisacle": 0.8}}},
            "Variant 'hrrr': 'zisacle' is not a setting. Did you mean 'ziscale', a hysplit setting?",
        ),
        (
            {"variants": {"hrrr": {"realisations": 2}}},
            "Did you mean 'realizations', a variant key?",
        ),
    ],
)
def test_an_unknown_key_names_the_nearest_setting(tmp_path, settings, message):
    settings = {"variants": {"hrrr": {}}, **settings}
    with pytest.raises(ValueError, match=re.escape(message)):
        ProjectConfig(mets=_met_config(tmp_path), **settings)


def test_a_relative_geometry_file_starts_from_the_project_directory(
    tmp_path, monkeypatch
):
    gpd = pytest.importorskip("geopandas")
    import shapely

    project = tmp_path / "proj"
    project.mkdir()
    gpd.GeoDataFrame(
        {"NAME": ["a"]},
        geometry=[shapely.box(-112.0, 40.5, -111.9, 40.6)],
        crs="EPSG:4326",
    ).to_file(project / "cells.geojson", driver="GeoJSON")
    config = ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            geometry={"kind": "file", "path": "cells.geojson", "ids": "NAME"},
            variants={"hrrr": {}},
        )
    )
    monkeypatch.chdir(tmp_path)  # not the project
    variant = config.resolve(project)["hrrr"]
    assert (
        variant.stages.get("footprint") is not None
        and variant.stages["footprint"].geometry_hash is not None
    )
    # The settings keep the path as written, so moving the project changes no hash.
    assert variant.stages.get("footprint").geometry.path == "cells.geojson"
