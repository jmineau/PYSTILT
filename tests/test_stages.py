"""The stage interface, held with two toy stages that import nothing from the footprint."""

from __future__ import annotations

import pytest

from stilt.config import ProjectConfig
from stilt.exceptions import MissingClass, StiltError
from stilt.identity import settings_hash, stage_hash
from stilt.stages import check_stage, get_stage, ordered, reads

from .fixtures.factories import make_met_config
from .fixtures.stages import (
    COUNT,
    DOUBLE,
    CountConfig,
    CountStage,
    DoubleConfig,
    DoubleStage,
    Forgetful,
    chain,
)


def _config(tmp_path, **blocks) -> ProjectConfig:
    """Return a config with one met, HYSPLIT, and the blocks given."""
    blocks.setdefault("models", {"hysplit": {"n_hours": -24}})
    blocks.setdefault("variants", {"hrrr": {}})
    return ProjectConfig(mets={"hrrr": make_met_config(tmp_path / "met")}, **blocks)


# -- a stage class ------------------------------------------------------------


def test_a_stage_is_made_from_its_import_path():
    stage = get_stage(COUNT)
    assert isinstance(stage, CountStage)
    assert (stage.name, stage.suffix, reads(stage)) == ("count", ".txt", ())
    assert reads(get_stage(DOUBLE)) == ("count",)
    assert get_stage(COUNT) is stage  # made once


def test_a_stage_must_be_named_by_an_import_path():
    with pytest.raises(ValueError, match="import path of its class.*not 'footprint'"):
        get_stage("footprint")


def test_a_class_without_the_members_of_a_stage_is_refused():
    with pytest.raises(ValueError, match="is not a stage: it has no resolve, check"):
        get_stage("tests.fixtures.stages.Forgetful")
    with pytest.raises(ValueError, match="is not a stage: it has no name, config"):
        get_stage("tests.fixtures.stages.CountConfig")


@pytest.mark.parametrize("name", ["particles", "logs", "scratch", "state", "stage"])
def test_a_stage_cannot_take_a_name_the_output_or_the_status_table_uses(name):
    path = "tests.test_stages._Renamed"
    _Renamed.name = name
    get_stage.cache_clear()
    with pytest.raises(ValueError, match=f"cannot be named '{name}'"):
        get_stage(path)


def test_a_stage_name_is_plain():
    _Renamed.name = "Wind-Rose"
    get_stage.cache_clear()
    with pytest.raises(ValueError, match="a stage's name must match"):
        get_stage("tests.test_stages._Renamed")


class _Renamed(CountStage):
    """A count under a name set by the test."""


@pytest.mark.parametrize("suffix", ["", "txt", ".", None])
def test_a_stage_suffix_must_be_a_file_extension(suffix):
    """With no extension every release id would be read from the file names as empty."""
    _Renamed.name = "renamed"
    _Renamed.suffix = suffix
    get_stage.cache_clear()
    try:
        with pytest.raises(
            ValueError, match="a stage's suffix must be a file extension"
        ):
            get_stage("tests.test_stages._Renamed")
    finally:
        _Renamed.suffix = ".txt"
        get_stage.cache_clear()


def test_a_stage_config_class_must_be_a_pydantic_model():
    with pytest.raises(ValueError, match="config_class must be a pydantic model"):
        get_stage("tests.test_stages._PlainConfig")


class _PlainConfig(CountStage):
    """A count whose config class is a dict."""

    name = "plain_config"
    config_class = dict


def test_a_class_whose_package_is_not_installed_names_the_package():
    with pytest.raises(MissingClass) as error:
        get_stage("nopkg_for_this_test.stages.Wind")
    assert str(error.value) == (
        "config.yaml names nopkg_for_this_test.stages.Wind, but "
        "nopkg_for_this_test is not installed here. Install "
        "nopkg_for_this_test and run this again."
    )
    assert isinstance(error.value, ImportError)
    assert isinstance(error.value, StiltError)


def test_a_class_its_package_does_not_have_says_so():
    with pytest.raises(MissingClass, match="tests.fixtures.stages has no Gone"):
        get_stage("tests.fixtures.stages.Gone")
    with pytest.raises(MissingClass, match="could not be imported"):
        get_stage("tests.fixtures.nomodule.Gone")


# -- the order stages run in --------------------------------------------------


def test_stages_run_after_the_stages_they_read():
    count, double = CountStage(), DoubleStage()
    assert ordered({"double": double, "count": count}) == ["count", "double"]
    assert ordered({"count": count}) == ["count"]
    assert ordered({}) == []


def test_a_stage_that_reads_one_the_config_lacks_is_refused():
    with pytest.raises(ValueError, match="'double' reads the stage 'count', which"):
        ordered({"double": DoubleStage()})


def test_stages_that_read_each_other_are_refused():
    class Loop(CountStage):
        reads = ("double",)

    with pytest.raises(ValueError, match="read each other in a circle"):
        ordered({"count": Loop(), "double": DoubleStage()})


# -- the promises check_stage holds ------------------------------------------


@pytest.mark.parametrize(
    ("stage", "config"),
    [
        (CountStage(), CountConfig()),
        (CountStage(), CountConfig(scale=3)),
        (DoubleStage(), DoubleConfig(add=2)),
    ],
)
def test_the_toy_stages_keep_their_promises(stage, config):
    check_stage(stage, config)


def test_a_config_class_that_does_not_read_its_record_back_is_caught():
    with pytest.raises(AssertionError, match="does not read back what settings"):
        check_stage(Forgetful(), DoubleConfig(add=2))


def test_a_record_that_is_not_plain_data_is_caught():
    class Opaque(CountStage):
        def settings(self, config):
            return {"scale": object()}

    with pytest.raises(AssertionError, match="plain data"):
        check_stage(Opaque(), CountConfig())


def test_a_suffix_must_be_an_extension():
    class Bare(CountStage):
        suffix = "txt"

    with pytest.raises(AssertionError, match="suffix 'txt' must be a file extension"):
        check_stage(Bare(), CountConfig())


# -- stages in a config -------------------------------------------------------


def test_a_variant_uses_every_stage_in_the_order_they_run(tmp_path):
    config = _config(tmp_path, stages=chain(scale=2))
    variant = config.resolve()["hrrr"]
    assert list(variant.stages) == ["count", "double"]  # written double first
    assert variant.stages["count"] == CountConfig(scale=2)
    assert variant.stage_classes == {"count": COUNT, "double": DOUBLE}
    assert isinstance(variant.stage("double"), DoubleStage)
    assert variant.stage_settings == {
        "count": {"scale": 2, "fail": False, "silent": False, "stale": False},
        "double": {"add": 0},
    }


def test_a_variant_changes_a_stages_settings_or_turns_it_off(tmp_path):
    variants = {
        "hrrr": {},
        "tripled": {"stages": {"count": {"scale": 3}}},
        "plain": {"stages": {"count": None, "double": None}},
        "counted": {"stages": {"double": None}},
    }
    resolved = _config(tmp_path, stages=chain(), variants=variants).resolve()
    assert resolved["tripled"].stages["count"].scale == 3
    assert resolved["tripled"].stages["double"] == resolved["hrrr"].stages["double"]
    assert resolved["plain"].stages == {}
    assert list(resolved["counted"].stages) == ["count"]
    # Equal particles for all four; a stage's folder follows its own settings.
    assert len({v.particles_hash for v in resolved.values()}) == 1
    count = {name: v.stage_hashes.get("count") for name, v in resolved.items()}
    assert count["hrrr"] == count["counted"] != count["tripled"]
    assert count["plain"] is None


def test_a_variant_cannot_turn_off_a_stage_another_of_its_stages_reads(tmp_path):
    with pytest.raises(
        ValueError,
        match="Variant 'hrrr' sets the stage 'count' to null, and its stage "
        "'double' reads that stage's files. Set 'double' to null too",
    ):
        _config(
            tmp_path, stages=chain(), variants={"hrrr": {"stages": {"count": None}}}
        )


def test_a_stage_that_reads_one_the_config_does_not_have_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'double' reads the stage 'count', which"):
        _config(tmp_path, stages={"double": {"class": DOUBLE}})


def test_a_stages_hash_takes_the_particles_and_what_it_reads(tmp_path):
    variants = {"hrrr": {}, "tripled": {"stages": {"count": {"scale": 3}}}}
    resolved = _config(tmp_path, stages=chain(), variants=variants).resolve()
    hrrr, tripled = resolved["hrrr"], resolved["tripled"]
    assert hrrr.stage_hashes["count"] == settings_hash(
        {
            "particles": hrrr.particles_hash,
            "count": {"scale": 1, "fail": False, "silent": False, "stale": False},
        }
    )
    assert hrrr.stage_hashes["double"] == stage_hash(
        "double",
        {"add": 0},
        hrrr.particles_hash,
        {"count": hrrr.stage_hashes["count"]},
    )
    # The same settings of double, made from another count: another folder.
    assert hrrr.stage_settings["double"] == tripled.stage_settings["double"]
    assert hrrr.stage_hashes["double"] != tripled.stage_hashes["double"]


def test_a_stages_block_names_its_class(tmp_path):
    with pytest.raises(ValueError, match="stages: count names no class"):
        _config(tmp_path, stages={"count": {"scale": 2}})
    with pytest.raises(ValueError, match="stages: count: A stage is named by the"):
        _config(tmp_path, stages={"count": {"class": "count"}})
    with pytest.raises(MissingClass, match="but nopkg_for_this_test is not installed"):
        _config(tmp_path, stages={"wind": {"class": "nopkg_for_this_test.Wind"}})


def test_a_stages_block_is_named_for_its_stage(tmp_path):
    with pytest.raises(
        ValueError,
        match=f"stages: tally: {COUNT} is the count stage, and a stage's block "
        "is named for its stage. Write it as stages: count:.",
    ):
        _config(tmp_path, stages={"tally": {"class": COUNT}})


def test_a_stages_setting_is_checked_by_its_config_class(tmp_path):
    with pytest.raises(ValueError, match=r"stages: count: [\s\S]*scale[\s\S]*integer"):
        _config(tmp_path, stages={"count": {"class": COUNT, "scale": "many"}})
    with pytest.raises(ValueError, match=r"stages: count: [\s\S]*scael"):
        _config(tmp_path, stages={"count": {"class": COUNT, "scael": 2}})
    with pytest.raises(
        ValueError, match=r"Variant 'hrrr', stages: count: [\s\S]*scale"
    ):
        _config(
            tmp_path,
            stages={"count": {"class": COUNT}},
            variants={"hrrr": {"stages": {"count": {"scale": "many"}}}},
        )


def test_a_stage_setting_written_elsewhere_is_sent_to_its_block(tmp_path):
    stages = {"count": {"class": COUNT}}
    with pytest.raises(
        ValueError, match="'scale' is a count setting. Write it under stages: count:"
    ):
        _config(
            tmp_path, stages=stages, models={"hysplit": {"n_hours": -24, "scale": 2}}
        )
    with pytest.raises(
        ValueError,
        match="'scale' is a count setting. Write it under the variant's stages: count:",
    ):
        _config(tmp_path, stages=stages, variants={"hrrr": {"scale": 2}})
    with pytest.raises(
        ValueError,
        match="stages: lenient: 'numpar' is a hysplit setting. Write it under models: hysplit:",
    ):
        ProjectConfig(
            mets={"hrrr": make_met_config(tmp_path / "met")},
            models={"hysplit": {"n_hours": -24}},
            stages={"lenient": {"class": "tests.test_stages._Lenient", "numpar": 5}},
            variants={"hrrr": {}},
        )


class _LenientConfig(CountConfig):
    """A count's settings that ignore a key they do not know, as the footprint's do."""

    model_config = {"extra": "ignore"}


class _Lenient(CountStage):
    """A count whose config class ignores unknown keys."""

    name = "lenient"
    config_class = _LenientConfig


def test_a_stage_checks_that_it_fits_the_model_when_variants_are_resolved(tmp_path):
    models = {"hysplit": {"n_hours": -24}}
    variants = {"hrrr": {}, "few": {"numpar": 13}}
    config = _config(
        tmp_path, models=models, stages={"count": {"class": COUNT}}, variants=variants
    )
    with pytest.raises(
        ValueError,
        match="Variant 'few', stages: count: it cannot count the particles of "
        "hysplit with numpar 13",
    ):
        config.resolve()
    # The variant without the stage has nothing to fit.
    variants["few"]["stages"] = {"count": None}
    resolved = _config(
        tmp_path, models=models, stages={"count": {"class": COUNT}}, variants=variants
    ).resolve()
    assert resolved["few"].stages == {}


def test_to_yaml_writes_each_stage_block_as_given(tmp_path):
    config = _config(tmp_path, stages=chain(scale=2))
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.stages == {
        "double": {"class": DOUBLE},
        "count": {"class": COUNT, "scale": 2},
    }
    assert (
        loaded.resolve()["hrrr"].stage_hashes == config.resolve()["hrrr"].stage_hashes
    )
