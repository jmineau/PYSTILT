"""Tests for stilt.config - models and YAML roundtrip."""

import textwrap

import pytest
from pydantic import ValidationError

from stilt.config import (
    ErrorParams,
    FirstOrderLifetimeTransformSpec,
    FootprintConfig,
    MetConfig,
    ModelConfig,
    ModelParams,
    STILTParams,
    TransportParams,
    VerticalOperatorTransformSpec,
    build_control_entries,
    build_setup_entries,
    iter_documented_config_fields,
)
from stilt.config.model import _resolved_field_meta

# ---------------------------------------------------------------------------
# ErrorParams.winderrtf
# ---------------------------------------------------------------------------


def test_winderrtf_all_none():
    assert ErrorParams().winderrtf == 0


def test_winderrtf_xy_only():
    e = ErrorParams(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)
    assert e.winderrtf == 1


def test_winderrtf_zi_only():
    e = ErrorParams(sigzierr=0.6, tlzierr=60.0, horcorzierr=40.0)
    assert e.winderrtf == 2


def test_winderrtf_both():
    e = ErrorParams(
        siguverr=1.0,
        tluverr=60.0,
        zcoruverr=500.0,
        horcoruverr=40.0,
        sigzierr=0.6,
        tlzierr=60.0,
        horcorzierr=40.0,
    )
    assert e.winderrtf == 3


def test_winderrtf_zero_value_params():
    """0.0 error params are set (not None) - winderrtf must still be 1."""
    e = ErrorParams(siguverr=0.0, tluverr=0.0, zcoruverr=0.0, horcoruverr=0.0)
    assert e.winderrtf == 1


def test_winderrtf_partial_xy_raises():
    with pytest.raises(ValidationError):
        ErrorParams(siguverr=1.0)  # only one of four XY params set


def test_winderrtf_partial_zi_raises():
    with pytest.raises(ValidationError):
        ErrorParams(sigzierr=0.6, tlzierr=60.0)  # missing horcorzierr


# ---------------------------------------------------------------------------
# STILTParams - flat construction and maxpar default
# ---------------------------------------------------------------------------


def test_stilt_params_flat_construction():
    """STILTParams accepts all fields flat (no met fields)."""
    p = STILTParams(
        n_hours=-24,
        numpar=500,
    )
    assert p.n_hours == -24
    assert p.numpar == 500


def test_stilt_params_maxpar_defaults_to_numpar():
    """maxpar is None in TransportParams by default; STILTParams sets it from numpar."""
    p = STILTParams(numpar=500)
    assert p.maxpar == 500


def test_stilt_params_ziscale_defaults_to_scalar_one():
    p = STILTParams()
    assert p.ziscale == 1.0


def test_build_setup_entries_uses_metadata_targets():
    p = STILTParams()
    entries = build_setup_entries(p)

    assert entries["numpar"] == p.numpar
    assert entries["varsiwant"] == p.varsiwant
    assert entries["ichem"] == 8
    assert entries["idsp"] == 2
    assert "emisshrs" not in entries
    assert "w_option" not in entries
    assert "z_top" not in entries
    assert "ziscale" not in entries
    assert "seed" not in entries


def test_build_setup_entries_includes_seed_when_set():
    p = STILTParams(seed=17)
    entries = build_setup_entries(p)

    assert entries["seed"] == 17


def test_build_control_entries_uses_control_targets():
    p = STILTParams(emisshrs=0.5, w_option=1, z_top=12000.0)
    entries = build_control_entries(p)

    assert entries == {
        "emisshrs": 0.5,
        "w_option": 1,
        "z_top": 12000.0,
    }


def test_resolved_field_meta_applies_defaults_and_overrides():
    assert _resolved_field_meta(ModelParams, "numpar")["target"] == "setup"
    assert _resolved_field_meta(TransportParams, "seed")["target"] == "setup"
    assert _resolved_field_meta(TransportParams, "emisshrs")["target"] == "control"
    assert _resolved_field_meta(TransportParams, "w_option")["target"] == "control"
    assert _resolved_field_meta(TransportParams, "z_top")["target"] == "control"
    assert _resolved_field_meta(TransportParams, "ziscale")["target"] == "zicontrol"
    assert _resolved_field_meta(ErrorParams, "siguverr")["target"] == "winderr"
    assert _resolved_field_meta(ErrorParams, "sigzierr")["target"] == "zierr"
    assert _resolved_field_meta(TransportParams, "ichem")["visibility"] == "internal"


def test_iter_documented_config_fields_hides_internal_transport_fields_by_default():
    public_names = {
        name
        for model, name, _ in iter_documented_config_fields(TransportParams)
        if model is TransportParams
    }
    internal_names = {
        name
        for model, name, _ in iter_documented_config_fields(
            TransportParams, include_internal=True
        )
        if model is TransportParams
    }

    assert "kagl" not in public_names
    assert "pinpf" not in public_names
    assert "kagl" in internal_names
    assert "pinpf" in internal_names


# ---------------------------------------------------------------------------
# MetConfig - construction
# ---------------------------------------------------------------------------


def test_met_config_construction(tmp_path):
    mc = MetConfig(
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )
    assert mc.directory == tmp_path / "met"
    assert mc.file_format == "%Y%m%d_%H"
    assert mc.n_min == 1  # default


# ---------------------------------------------------------------------------
# ModelConfig - flat construction
# ---------------------------------------------------------------------------


def test_model_config_flat_construction(tmp_path):
    cfg = ModelConfig(
        n_hours=-24,
        numpar=100,
        seed=42,
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
    )
    assert cfg.n_hours == -24
    assert cfg.numpar == 100
    assert cfg.seed == 42


def test_model_config_requires_nonempty_mets():
    """ModelConfig must have at least one met entry."""
    with pytest.raises(Exception, match="at least one"):
        ModelConfig(mets={})


def test_model_config_rejects_non_alphanumeric_met_keys(tmp_path):
    mc = MetConfig(directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h")
    with pytest.raises(Exception, match="alphanumeric"):
        ModelConfig(mets={"hrrr_v2": mc})


def test_model_config_footprints_field(tmp_path, point_receptor, grid):
    """footprints dict accepted directly at construction."""
    fc = FootprintConfig(grid=grid)
    mc = MetConfig(directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h")
    cfg = ModelConfig(
        mets={"hrrr": mc},
        footprints={"slv": fc},
    )
    assert "slv" in cfg.footprints
    assert cfg.footprints["slv"].model_dump() == fc.model_dump()


# ---------------------------------------------------------------------------
# ModelConfig YAML roundtrip
# ---------------------------------------------------------------------------


def test_model_config_yaml_roundtrip_basic(tmp_path):
    cfg = ModelConfig(
        n_hours=-24,
        numpar=100,
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ModelConfig.from_yaml(path)
    text = path.read_text()

    assert "#" not in text
    assert loaded.n_hours == -24
    assert loaded.numpar == 100
    assert "hrrr" in loaded.mets
    assert loaded.mets["hrrr"].file_format == "%Y%m%d_%H"
    assert loaded.mets["hrrr"].file_tres == "1h"


def test_model_config_yaml_roundtrip_with_execution(tmp_path):
    cfg = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        execution={
            "backend": "slurm",
            "n_workers": 8,
            "timeout": 120,
            "partition": "compute",
            "account": "lin-group",
        },
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ModelConfig.from_yaml(path)
    assert loaded.execution is not None
    assert loaded.execution["backend"] == "slurm"
    assert loaded.execution["n_workers"] == 8
    assert loaded.execution["timeout"] == 120
    assert loaded.execution["partition"] == "compute"
    assert loaded.execution["account"] == "lin-group"


def test_model_config_accepts_execution_dict_with_extra_fields(tmp_path):
    cfg = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        execution={
            "backend": "kubernetes",
            "n_workers": 32,
            "namespace": "atmos",
            "image": "my/stilt:latest",
        },
    )
    assert cfg.execution["backend"] == "kubernetes"
    assert cfg.execution["namespace"] == "atmos"
    assert cfg.execution["image"] == "my/stilt:latest"


def test_model_config_yaml_roundtrip_with_footprint(tmp_path, point_receptor, grid):
    """FootprintConfig survives a to_yaml/from_yaml roundtrip."""
    fc = FootprintConfig(grid=grid)
    cfg = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        footprints={"slv_fine": fc},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ModelConfig.from_yaml(path)
    assert "slv_fine" in loaded.footprints
    assert loaded.footprints["slv_fine"].model_dump() == fc.model_dump()


def test_model_config_yaml_roundtrip_with_footprint_transforms(tmp_path, grid):
    fc = FootprintConfig(
        grid=grid,
        transforms=[
            VerticalOperatorTransformSpec(
                kind="vertical_operator",
                mode="ak_pwf",
                levels=[0.0, 1000.0],
                values=[0.2, 0.8],
                coordinate="xhgt",
            ),
            FirstOrderLifetimeTransformSpec(
                kind="first_order_lifetime",
                lifetime_hours=4.0,
                time_column="time",
                time_unit="min",
            ),
        ],
    )
    cfg = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        footprints={"slv_fine": fc},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ModelConfig.from_yaml(path)
    transforms = loaded.footprints["slv_fine"].transforms
    assert len(transforms) == 2
    assert transforms[0].model_dump() == fc.transforms[0].model_dump()
    assert transforms[1].model_dump() == fc.transforms[1].model_dump()


def test_model_config_domain_ref_in_yaml(tmp_path):
    """Footprint config loaded from YAML with named grid reference."""
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        grids:
          slv:
            xmin: -114.0
            xmax: -111.0
            ymin: 39.0
            ymax: 42.0
            xres: 0.01
            yres: 0.01
        footprints:
          slv_fine:
            grid: slv
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    loaded = ModelConfig.from_yaml(path)
    assert "slv_fine" in loaded.footprints
    fc = loaded.footprints["slv_fine"]
    assert fc.grid.xmin == -114.0
    assert fc.grid.xres == 0.01


def test_model_config_inline_bounds_in_yaml(tmp_path):
    """Footprint config with inline grid definition."""
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        footprints:
          slv_coarse:
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    loaded = ModelConfig.from_yaml(path)
    assert "slv_coarse" in loaded.footprints
    fc = loaded.footprints["slv_coarse"]
    assert fc.grid.xres == 0.05
    assert fc.grid.xmin == -114.0


def test_model_config_footprint_grid_shorthand_in_yaml(tmp_path):
    """Footprint config accepts grid bounds directly under the footprint name."""
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        footprints:
          slv_coarse:
            xmin: -114.0
            xmax: -111.0
            ymin: 39.0
            ymax: 42.0
            xres: 0.05
            yres: 0.05
            smooth_factor: 0.75
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    loaded = ModelConfig.from_yaml(path)
    fc = loaded.footprints["slv_coarse"]

    assert fc.grid.xmin == -114.0
    assert fc.grid.xres == 0.05
    assert fc.smooth_factor == 0.75


def test_model_config_loads_footprint_transforms_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        footprints:
          weighted:
            grid:
              xmin: -114.0
              xmax: -111.0
              ymin: 39.0
              ymax: 42.0
              xres: 0.05
              yres: 0.05
            transforms:
              - kind: vertical_operator
                mode: ak
                levels: [0.0, 1000.0]
                values: [0.3, 0.7]
                coordinate: xhgt
              - kind: first_order_lifetime
                lifetime_hours: 3.0
                time_column: time
                time_unit: min
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    loaded = ModelConfig.from_yaml(path)
    transforms = loaded.footprints["weighted"].transforms

    assert isinstance(transforms[0], VerticalOperatorTransformSpec)
    assert transforms[0].mode == "ak"
    assert isinstance(transforms[1], FirstOrderLifetimeTransformSpec)
    assert transforms[1].lifetime_hours == pytest.approx(3.0)


def test_model_config_unknown_keys_raise(tmp_path):
    """from_yaml must fail fast on unrecognized keys."""

    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        mystery_param: 99
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValidationError, match="mystery_param"):
        ModelConfig.from_yaml(path)
