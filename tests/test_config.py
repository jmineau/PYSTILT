"""Tests for stilt.config - models and YAML roundtrip."""

import re
import textwrap

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from stilt.config import FOOTPRINT_STAGE, ProjectConfig
from stilt.footprint.config import FootprintConfig
from stilt.footprint.targets import Mesh
from stilt.identity import footprint_settings, read_footprint_settings
from stilt.spatial import Grid
from stilt.transforms import AveragingKernel, FirstOrderLifetime, PressureWeighting
from stilt.transport.hysplit.driver import winderrtf
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

    def apply(self, particles, receptor=None, directory=None):
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
    assert variants["hrrr"].footprint.grid == grid
    assert variants["hrrr"].footprint.smooth_factor == 0.5
    assert variants["hrrr-long"].footprint == variants["hrrr"].footprint


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (
            {"n_hours": -24},
            "config: 'n_hours' is a hysplit setting. Write it under models: hysplit:.",
        ),
        (
            {"smooth_factor": 0.5},
            "config: 'smooth_factor' is a footprint setting. Write it under stages:, "
            "in the footprint's block (class: stilt.footprint.FootprintStage).",
        ),
        (
            {"model": "hysplit"},
            "config: 'model' is no longer a key. Name each transport model as a "
            "block under models:, such as models: {hysplit: {}}.",
        ),
    ],
    ids=["model setting", "footprint setting", "model"],
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
            mets=mets, models={"hysplit": {"grid": grid}}, variants={"hrrr": {}}
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


def test_a_models_wrong_value_is_reported_for_its_block(tmp_path):
    """A block is checked when the file loads, as the variants that change none of it run it."""
    mets = _met_config(tmp_path)
    with pytest.raises(
        ValueError, match=r"models: hysplit \(hysplit\): [\s\S]*n_hours must not be 0"
    ):
        ProjectConfig(
            mets=mets, models={"hysplit": {"n_hours": 0}}, variants={"hrrr": {}}
        )
    with pytest.raises(
        ValueError, match=r"Variant 'hrrr' \(hysplit\): [\s\S]*n_hours must not be 0"
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            variants={"hrrr": {"n_hours": 0}},
        ).resolve()


def test_a_stage_block_names_the_footprint_stage_as_its_class(tmp_path, grid):
    mets = _met_config(tmp_path)
    with pytest.raises(
        ValueError,
        match="stages: footprint names no class. The footprint is "
        "class: stilt.footprint.FootprintStage.",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"footprint": {"grid": grid}},
            variants={"hrrr": {}},
        )
    with pytest.raises(
        ValueError, match="stages: wind: 'mypkg.WindRose' is not a stage"
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"wind": {"class": "mypkg.WindRose"}},
            variants={"hrrr": {}},
        )
    stage = {"class": FOOTPRINT_STAGE, "grid": grid}
    with pytest.raises(
        ValueError,
        match="stages: fine and coarse are both stilt.footprint.FootprintStage. "
        "A stage has one block",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=mets,
            stages={"fine": stage, "coarse": stage},
            variants={"hrrr": {}},
        )


def test_a_stage_blocks_name_is_what_a_variant_calls_it(tmp_path, grid):
    """The block's name is a handle: the class says what the stage is."""
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"foot": {"class": FOOTPRINT_STAGE, "grid": grid}},
        variants={
            "hrrr": {},
            "smooth": {"stages": {"foot": {"smooth_factor": 2}}},
            "particles": {"stages": {"foot": None}},
        },
    )
    variants = config.resolve()
    assert variants["hrrr"].footprint.smooth_factor == 1
    assert variants["smooth"].footprint.smooth_factor == 2
    assert variants["smooth"].footprint.grid == grid
    assert variants["particles"].footprint is None
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr' names the stage 'footprint', which stages: does "
        r"not have \(it has \['foot'\]\)",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            stages={"foot": {"class": FOOTPRINT_STAGE, "grid": grid}},
            variants={"hrrr": {"stages": {"footprint": None}}},
        )


@pytest.mark.parametrize(
    ("block", "what"),
    [("models", "Model"), ("stages", "Stage")],
    ids=["model", "stage"],
)
def test_a_blocks_name_is_plain_as_a_variants_is(tmp_path, grid, block, what):
    """A model's or a stage's name is what a variant writes, so it follows the variants' rule."""
    settings = {"class": FOOTPRINT_STAGE, "grid": grid} if block == "stages" else {}
    with pytest.raises(ValueError, match=f"{what} name 'Not_Plain' must match"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            variants={"hrrr": {}},
            **{block: {"Not_Plain": settings}},
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


def test_a_met_no_variant_uses_is_still_checked(tmp_path):
    """With one model, that model's met config checks every met, used or not."""
    mets = {**_met_config(tmp_path), "spare": {"subgrid_enable": "maybe"}}
    with pytest.raises(
        ValueError, match=r"Met 'spare' \(hysplit\): [\s\S]*subgrid_enable"
    ):
        ProjectConfig(
            mets=mets, models={"hysplit": {"n_hours": -24}}, variants={"hrrr": {}}
        )


def test_a_config_or_a_block_that_is_not_a_mapping_is_an_error(tmp_path):
    """The checks of keys leave anything that is not a mapping to pydantic's own message."""
    from stilt.execution.config import ExecutionConfig, SlurmConfig

    with pytest.raises(ValidationError, match="ProjectConfig"):
        ProjectConfig.model_validate(["mets", "variants"])
    with pytest.raises(ValidationError, match="mets"):
        ProjectConfig(mets=str(tmp_path / "met"), variants={"hrrr": {}})
    with pytest.raises(ValidationError, match="ExecutionConfig"):
        ExecutionConfig.model_validate("slurm")
    with pytest.raises(ValidationError, match="SlurmConfig"):
        SlurmConfig.model_validate("02:00:00")


def test_the_footprint_stage_is_at_the_path_a_config_names():
    import importlib

    module, _, name = FOOTPRINT_STAGE.rpartition(".")
    stage = getattr(importlib.import_module(module), name)
    assert stage.name == "footprint"
    assert stage.config_class is FootprintConfig


def test_model_config_requires_nonempty_mets():
    """ProjectConfig must have at least one met entry."""
    with pytest.raises(Exception, match="at least one"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}}, mets={}, variants={"hrrr": {}}
        )


def test_model_config_rejects_met_keys_that_cannot_name_a_variant(tmp_path):
    mc = make_met_config(directory=tmp_path / "met")
    with pytest.raises(Exception, match="must match"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets={"hrrr_v2": mc},
            variants={"hrrr_v2": {}},
        )


def test_a_config_loads_and_resolves_without_its_hysplit_build(tmp_path):
    """The build is read on first use, so a machine without hycs_std opens the project."""
    mc = make_met_config(directory=tmp_path / "met")
    elsewhere = tmp_path / "not-mounted"
    cfg = ProjectConfig(
        **blocks(mets={"hrrr": mc}, directory=elsewhere, variants={"hrrr": {}})
    )
    variant = cfg.resolve()["hrrr"]
    with pytest.raises(FileNotFoundError, match="hycs_std"):
        _ = variant.build
    with pytest.raises(ValueError, match="realizations must be >= 1"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets={"hrrr": mc},
            variants={"e": {"realizations": 0}},
        )


def _default_footprint(cfg):
    """The footprint settings of the first variant, which inherits the defaults."""
    return next(iter(cfg.resolve().values())).footprint


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


def test_model_config_yaml_roundtrip_basic(tmp_path):
    cfg = ProjectConfig(
        mets={"hrrr": make_met_config(directory=tmp_path / "met")},
        models={"hysplit": {"n_hours": -24, "numpar": 100}},
        variants={"hrrr": {}},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    text = path.read_text()

    assert "#" not in text
    assert loaded.models == {"hysplit": {"n_hours": -24, "numpar": 100}}
    assert "hrrr" in loaded.mets
    assert list(yaml.safe_load(text)) == ["mets", "models", "variants"]


def test_model_config_reports_invalid_yaml(tmp_path):
    """Malformed YAML is reported as an invalid config file, not a parser traceback."""
    path = tmp_path / "config.yaml"
    path.write_text("n_hours: [\n")

    with pytest.raises(
        ValueError, match=r"^config\.yaml is not valid YAML at line 2, column 1: "
    ):
        ProjectConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("text", "where"),
    [
        (
            "models:\n  hysplit:\n    n_hours: -24\n    numpar: 100\n    n_hours: 24\n",
            "line 5, column 5: 'n_hours'",
        ),
        (
            "variants:\n  a:\n    numpar: 10\n    numpar: 20\n",
            "line 4, column 5: 'numpar'",
        ),
    ],
    ids=["block", "variant"],
)
def test_model_config_refuses_a_repeated_key(tmp_path, text, where):
    """A key written twice is refused, not read as its last value (#440)."""
    path = tmp_path / "config.yaml"
    path.write_text(text)

    with pytest.raises(ValueError) as error:
        ProjectConfig.from_yaml(path)
    assert str(error.value) == (
        f"config.yaml is not valid YAML at {where} is written twice."
    )


def test_model_config_yaml_roundtrip_with_execution(tmp_path):
    cfg = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        execution={
            "backend": "slurm",
            "slurm": {
                "n_workers": 8,
                "time": "02:00:00",
                "partition": "compute",
                "account": "lin-group",
                "exclude": "node1",
            },
        },
        variants={"hrrr": {}},
    )
    path = tmp_path / "config.yaml"
    text = cfg.to_yaml(path)
    assert "cpus" not in text and "array_parallelism" not in text  # unset stays out
    assert "local:" not in text
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.execution == cfg.execution
    assert loaded.execution.backend == "slurm"
    assert loaded.execution.slurm.n_workers == 8
    assert loaded.execution.slurm.options == {"exclude": "node1"}


def test_execution_defaults_to_one_local_process(tmp_path):
    execution = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        variants={"hrrr": {}},
    ).execution
    assert (execution.backend, execution.cpus) == ("local", 1)
    assert (execution.slurm.n_workers, execution.slurm.cpus) == (1, 1)
    assert (
        "execution:"
        not in ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            variants={"hrrr": {}},
        ).to_yaml()
    )


def test_each_backends_settings_are_in_its_own_block():
    """``cpus`` is the block's that ``backend`` names; the other block is not read."""
    from stilt.execution.config import ExecutionConfig

    settings = {
        "local": {"cpus": 2},
        "slurm": {"cpus": 8, "n_workers": 40, "time": "04:00:00"},
        "timeout": 600,
    }
    local = ExecutionConfig.model_validate(settings)
    slurm = ExecutionConfig.model_validate({"backend": "slurm", **settings})
    assert (local.backend, local.cpus) == ("local", 2)
    assert (slurm.backend, slurm.cpus) == ("slurm", 8)
    assert local.timeout == slurm.timeout == 600
    # A config may hold a slurm block with no time and run locally.
    assert ExecutionConfig.model_validate({"slurm": {"n_workers": 4}}).cpus == 1
    with pytest.raises(ValueError, match="execution.slurm.time is required"):
        ExecutionConfig.model_validate({"backend": "slurm", "slurm": {"n_workers": 4}})


@pytest.mark.parametrize(
    ("setting", "blocks"),
    [
        (
            {"cpus": 4},
            "execution: local: or execution: slurm:, for the backend it is for.",
        ),
        ({"n_workers": 8}, "execution: slurm:."),
        ({"time": "02:00:00"}, "execution: slurm:."),
        ({"partition": "compute"}, "execution: slurm:."),
        (
            {"fetch": False},
            "execution: local: or execution: slurm:, for the backend it is for.",
        ),
    ],
)
def test_a_backends_setting_beside_backend_is_refused_naming_its_block(
    tmp_path, setting, blocks
):
    (key,) = setting
    message = f"'{key}' is a setting of a backend. Write it under {blocks}"
    with pytest.raises(ValueError, match=re.escape(message)):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution=setting,
            variants={"hrrr": {}},
        )


def test_unknown_execution_setting_is_an_error(tmp_path):
    """A typo, or an sbatch option outside `slurm:`, used to be ignored or passed on (#63)."""
    with pytest.raises(ValueError, match=r"execution\.partion\s+Extra inputs"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"partion": "compute"},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match=r"execution\.requeue\s+Extra inputs"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"requeue": True},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match=r"execution\.cpus_per_task\s+Extra inputs"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"cpus_per_task": 4},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="backend"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"backend": "kubernetes"},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="n_workers"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"slurm": {"n_workers": 0}},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match=r"execution\.local\.n_workers\s+Extra inputs"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            execution={"local": {"n_workers": 2}},
            variants={"hrrr": {}},
        )


def test_a_misspelled_slurm_setting_is_not_sent_to_sbatch(tmp_path):
    """Any other key of the slurm block is an sbatch option, so a near miss of a setting is refused."""
    from stilt.execution.config import SlurmConfig

    for typo, setting in (("partion", "partition"), ("n_worker", "n_workers")):
        with pytest.raises(
            ValueError,
            match=f"'{typo}' is not a setting of execution.slurm. Did you mean '{setting}'",
        ):
            SlurmConfig.model_validate({typo: 1})
    options = {"exclude": "node1", "constraint": "skl", "cpus-per-task": 8}
    assert SlurmConfig.model_validate(options).options == options


def test_execution_accepts_one_setup_line():
    from stilt.execution.config import SlurmConfig

    assert SlurmConfig.model_validate({"setup": "module load hysplit"}).setup == [
        "module load hysplit"
    ]


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


@pytest.mark.parametrize("where", ["top", "variant"])
def test_time_integrate_is_refused_saying_what_to_do_instead(tmp_path, grid, where):
    footprint = {"time_integrate": True}
    top = footprint if where == "top" else {}
    variant = {} if where == "top" else footprint
    with pytest.raises(ValueError, match=r"time_integrate was removed.*sum\('time'\)"):
        ProjectConfig(
            **blocks(
                mets=_met_config(tmp_path), grid=grid, variants={"hrrr": variant}, **top
            )
        )


@pytest.mark.parametrize("where", ["top", "variant"])
def test_veght_is_refused_naming_footprint_depth(tmp_path, where):
    top = {"veght": 0.3} if where == "top" else {}
    variant = {} if where == "top" else {"veght": 0.3}
    with pytest.raises(ValueError, match="veght was renamed footprint_depth"):
        ProjectConfig(
            **blocks(mets=_met_config(tmp_path), variants={"hrrr": variant}, **top)
        )


def test_footprint_depth_goes_to_hysplit_as_veght(tmp_path):
    from stilt.transport.hysplit.driver import setup_entries

    config = ProjectConfig(
        mets=_met_config(tmp_path),
        models={"hysplit": {"n_hours": -24, "footprint_depth": 0.3}},
        variants={"hrrr": {}},
    )
    entries = setup_entries(config.resolve()["hrrr"].transport)
    assert entries["veght"] == 0.3
    assert "footprint_depth" not in entries


def test_hnf_plume_needs_its_columns_in_varsiwant_only_with_a_footprint(tmp_path, grid):
    varsiwant = ["time", "indx", "long", "lati", "zagl", "foot"]
    with pytest.raises(ValueError, match=r"near-field correction \(hnf_plume\) reads"):
        ProjectConfig(
            **blocks(
                mets=_met_config(tmp_path),
                grid=grid,
                varsiwant=varsiwant,
                variants={"hrrr": {}},
            )
        )
    # Without a footprint, or with the correction off, nothing reads them.
    ProjectConfig(
        **blocks(mets=_met_config(tmp_path), varsiwant=varsiwant, variants={"hrrr": {}})
    )
    ProjectConfig(
        **blocks(
            mets=_met_config(tmp_path),
            grid=grid,
            varsiwant=varsiwant,
            hnf_plume=False,
            variants={"hrrr": {}},
        )
    )


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
    assert variants["hrrr"].footprint is not None
    assert variants["hrrr-err"].footprint is None
    # A footprint with its grid taken away has nothing to grid on: say how to turn it off.
    with pytest.raises(
        ValueError,
        match=r"Variant 'hrrr-err': its footprint has no grid. [\s\S]*"
        r"stages: \{footprint: null\} to make particles only",
    ):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            stages={"footprint": stage},
            variants={
                "hrrr": {},
                "hrrr-err": {"stages": {"footprint": {"grid": None}}},
            },
        )


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
    transforms = loaded.resolve()["hrrr"].footprint.transforms
    assert len(transforms) == 3
    assert transforms == given
    assert isinstance(transforms[0], AveragingKernel)
    assert isinstance(transforms[1], PressureWeighting)
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert loaded.resolve()["hrrr"].footprint.transforms == given


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
    transforms = loaded.resolve()["hrrr"].footprint.transforms
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
    assert variants["hrrr"].footprint.grid.xres == 0.01
    assert variants["coarse"].footprint.grid.xres == 0.05
    assert variants["coarse"].footprint.grid.xmin == -114.0


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
    footprint = ProjectConfig.from_yaml(path).resolve()["hrrr"].footprint
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
    assert variants["hrrr"].footprint is not None
    assert variants["traj"].footprint is None


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
    transforms = loaded.resolve()["hrrr"].footprint.transforms

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


def test_model_config_unknown_keys_raise(tmp_path):
    """from_yaml must fail fast on unrecognized keys."""
    yaml_text = textwrap.dedent(f"""\
        mets:
          hrrr:
            directory: {tmp_path / "met"}
        mystery_param: 99
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValidationError, match="mystery_param"):
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
    assert variant.footprint is not None
    return variant.footprint


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


def test_footprint_needs_a_grid(point_receptor):
    import pandas as pd

    from stilt.footprint import calc_footprint

    assert FootprintConfig().grid is None
    with pytest.raises(ValueError, match="grid"):
        calc_footprint(pd.DataFrame(), point_receptor, None)


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
    fc = next(iter(config.resolve().values())).footprint
    assert fc is not None
    assert fc.grid.xres == pytest.approx(0.02)  # 0.1 / 4 -> 0.025 -> 0.02
    assert fc.geometry is not None and fc.geometry.kind == "file"

    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    lfc = next(iter(loaded.resolve().values())).footprint
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
    with pytest.raises(ValueError, match="Variant 'hrrr': its footprint has no grid"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets=_met_config(tmp_path),
            stages={"footprint": stage},
            variants={"hrrr": {}},
        )
    # The block may leave the grid to each variant.
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets=_met_config(tmp_path),
        stages={"footprint": stage},
        variants={"hrrr": {"stages": {"footprint": {"grid": grid}}}},
    )
    footprint = config.resolve()["hrrr"].footprint
    assert footprint.grid == grid and footprint.smooth_factor == 2


def test_variants_default_to_one_per_met(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(
        **blocks(
            mets={"hrrr": mc, "gfs": mc}, ziscale=0.9, variants={"hrrr": {}, "gfs": {}}
        )
    )
    variants = cfg.resolve()
    assert list(variants) == ["hrrr", "gfs"]
    assert variants["gfs"].met == "gfs"
    assert variants["gfs"].transport.ziscale == 0.9
    assert variants["hrrr"].transport.ziscale == 0.9


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
    assert variants["zi08"].footprint == variants["hrrr"].footprint


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
    assert variants["coarse"].footprint is not None
    assert variants["coarse"].footprint.grid.xres == 0.1
    assert variants["coarse"].footprint.grid.xmin == -114
    assert variants["none"].footprint is None


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
    assert loaded["decay"].footprint.transforms[0].lifetime_hours == 1.0


def test_variant_must_name_its_met_when_there_are_several(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    with pytest.raises(ValueError, match="must name its met"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets={"hrrr": mc, "gfs": mc},
            variants={"a": {}},
        )
    with pytest.raises(ValueError, match="unknown met"):
        ProjectConfig(
            models={"hysplit": {"n_hours": -24}},
            mets={"hrrr": mc},
            variants={"a": {"met": "nam"}},
        )


@pytest.mark.parametrize("name", ["Hrrr", "hrrr_v2", "-x", "a b"])
def test_variant_names_are_plain(tmp_path, name):
    with pytest.raises(ValueError, match="must match"):
        _variant_config(tmp_path, variants={name: {}})


def test_an_ensemble_is_one_variant_whose_realizations_have_their_own_seed(tmp_path):
    cfg = _variant_config(
        tmp_path,
        krand=2,
        seed=42,
        variants={
            "hrrr": {},
            "err": {
                "siguverr": 1.0,
                "tluverr": 60.0,
                "zcoruverr": 500.0,
                "horcoruverr": 40.0,
                "realizations": 3,
            },
        },
    )
    variants = cfg.resolve()
    assert list(variants) == ["hrrr", "err"]
    err = variants["err"]
    assert err.realizations == 3 and err.realization_numbers == [0, 1, 2]
    assert err.transport.seed == 42  # the base seed
    assert [err.transport_for(k).seed for k in range(3)] == [42, 43, 44]
    assert winderrtf(err.transport_for(1)) == 1
    assert winderrtf(variants["hrrr"].transport) == 0
    assert variants["hrrr"].realization_numbers == [None]
    with pytest.raises(ValueError, match="realizations"):
        err.transport_for(3)


@pytest.mark.parametrize("krand", [0, 1, 2, 3, 12])
def test_several_realizations_require_krand_4_or_a_seed(tmp_path, krand):
    config = _variant_config(tmp_path, krand=krand, variants={"e": {"realizations": 3}})
    with pytest.raises(ValueError, match="requires krand=4 or krand=2 with a seed"):
        config.resolve()


def test_several_realizations_accept_krand_4_or_seeded_krand_2(tmp_path):
    a = _variant_config(tmp_path, krand=4, variants={"e": {"realizations": 3}})
    assert a.resolve()["e"].realizations == 3
    b = _variant_config(tmp_path, krand=2, seed=7, variants={"e": {"realizations": 2}})
    assert b.resolve()["e"].realizations == 2
    with pytest.raises(ValueError, match="realizations must be >= 1"):
        _variant_config(tmp_path, variants={"e": {"realizations": 0}})


def test_an_ensemble_keeps_its_folder_when_its_realizations_grow(tmp_path):
    """An ensemble of one is an ensemble, so raising the count later only adds runs."""
    cfg = _variant_config(tmp_path, variants={"e": {"realizations": 1}, "single": {}})
    variants = cfg.resolve()
    assert variants["e"].realization_numbers == [0]
    assert variants["single"].realization_numbers == [None]
    # An ensemble records that it is one, so it is not the single run.
    assert variants["e"].particles_hash != variants["single"].particles_hash
    more = _variant_config(
        tmp_path, krand=4, variants={"e": {"realizations": 3}, "single": {}}
    ).resolve()
    one = _variant_config(
        tmp_path, krand=4, variants={"e": {"realizations": 1}, "single": {}}
    ).resolve()
    assert more["e"].particles_hash == one["e"].particles_hash
    assert "realization" not in variants["single"].run_settings
    assert "ensemble" not in variants["single"].run_settings
    assert variants["e"].run_settings["ensemble"] is True


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
    assert s2.footprint is not None and s2.footprint.smooth_factor == 2
    assert s2.transport == variants["hrrr"].transport
    assert s2.particles_hash == variants["hrrr"].particles_hash


def test_from_is_rejected_with_advice(tmp_path):
    with pytest.raises(ValueError, match="no longer needed"):
        _variant_config(tmp_path, variants={"a": {}, "b": {"from": "a"}})


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


def test_to_yaml_always_writes_the_variants_that_run(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(
        **blocks(
            mets={"hrrr": mc, "gfs": mc}, numpar=50, variants={"hrrr": {}, "gfs": {}}
        )
    )
    text = cfg.to_yaml()
    assert "variants:\n  hrrr: {}\n  gfs: {}\n" in text
    loaded = ProjectConfig.model_validate(__import__("yaml").safe_load(text))
    assert list(loaded.resolve()) == ["hrrr", "gfs"]
    assert loaded.resolve()["gfs"].met == "gfs"


def test_variant_named_after_a_met_uses_it(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets={"hrrr": mc, "gfs": mc},
        variants={"gfs": {}, "hrrr": {}},
    )
    assert {n: v.met for n, v in cfg.resolve().items()} == {
        "gfs": "gfs",
        "hrrr": "hrrr",
    }


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
    mesh = Mesh.from_spec(variants["src"].footprint.geometry)
    expected = mesh.to_grid()
    for name in ("src", "src-run"):
        assert variants[name].footprint.grid == expected
        assert variants[name].geometry_hash == mesh.hash
    assert variants["hrrr"].footprint.grid != expected


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
    inherited = [variants[n].footprint for n in ("hrrr", "np50", "err")]
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
    stored = footprint_settings(config, "deadbeef00")
    assert read_footprint_settings(stored) == (config, "deadbeef00")
    assert calls == []


def test_to_yaml_does_not_write_the_derived_grid(tmp_path):
    cfg = _variant_config(tmp_path, geometry=_WINDOWS)
    foot = _default_footprint(cfg)
    assert foot is not None and foot.grid is not None
    text = cfg.to_yaml()
    assert "geometry:" in text
    assert "grid:" not in text and "geometry_hash" not in text


def test_partial_grid_over_a_derived_grid_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="'coarse' changes part of a grid"):
        _variant_config(
            tmp_path,
            geometry=_WINDOWS,
            variants={"hrrr": {}, "coarse": {"grid": {"xres": 0.1, "yres": 0.1}}},
        )


def test_maxpar_follows_each_variants_numpar(tmp_path):
    """A variant that raises numpar is not capped at the default's (the old validator did)."""
    cfg = _variant_config(
        tmp_path, numpar=1000, variants={"hrrr": {}, "np3k": {"numpar": 3000}}
    )
    variants = cfg.resolve()
    assert variants["hrrr"].transport.maxpar_for(1) == 1000
    assert variants["np3k"].transport.maxpar_for(1) == 3000
    assert variants["np3k"].transport.maxpar_for(40) == 40 * 76
    # An unset maxpar is recorded unset: what HYSPLIT gets depends on the
    # receptor's locations.
    assert variants["hrrr"].run_settings["maxpar"] is None


# ---------------------------------------------------------------------------
# Variants are declared; unknown keys name the nearest setting (#150)
# ---------------------------------------------------------------------------


def test_a_config_without_variants_says_what_to_write(tmp_path):
    with pytest.raises(
        ValueError, match=r"no variants. Add\n  variants:\n    hrrr: \{\}"
    ):
        ProjectConfig(models={"hysplit": {"n_hours": -24}}, mets=_met_config(tmp_path))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"smooth_factr": 0.5},
            "config: 'smooth_factr' is not a setting. Did you mean 'smooth_factor'? "
            "It is a footprint setting. Write it under stages:, in the footprint's block",
        ),
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
    assert variant.footprint is not None and variant.geometry_hash is not None
    # The settings keep the path as written, so moving the project changes no hash.
    assert variant.footprint.geometry.path == "cells.geojson"


def test_a_met_may_be_only_its_folder(tmp_path):
    """``hrrr: /path/to/hrrr`` is the whole entry, and stays one line in the file."""
    write_arl_file(tmp_path / "met" / "20230101_00-05_hrrr")
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets={"hrrr": str(tmp_path / "met")},
        variants={"hrrr": {}},
    )

    assert config.met_config("hrrr").directory == tmp_path / "met"
    written = config.to_yaml()
    assert yaml.safe_load(written)["mets"] == {"hrrr": str(tmp_path / "met")}


def test_a_met_may_be_a_path_in_python(tmp_path):
    write_arl_file(tmp_path / "met" / "20230101_00-05_hrrr")
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets={"hrrr": tmp_path / "met"},
        variants={"hrrr": {}},
    )

    assert config.mets == {"hrrr": str(tmp_path / "met")}


def test_a_relative_met_folder_starts_from_the_project(tmp_path):
    write_arl_file(tmp_path / "met" / "20230101_00-05_hrrr")
    config = ProjectConfig(
        models={"hysplit": {"n_hours": -24}},
        mets={"hrrr": "met"},
        variants={"hrrr": {}},
    )

    assert config.resolve(tmp_path)["hrrr"].met_config.directory == tmp_path / "met"
