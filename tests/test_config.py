"""Tests for stilt.config - models and YAML roundtrip."""

import textwrap

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from stilt.config import (
    ErrorParams,
    FootprintConfig,
    Grid,
    MetConfig,
    ModelParams,
    ProjectConfig,
    STILTParams,
    TransportParams,
)
from stilt.transforms import AveragingKernel, FirstOrderLifetime, PressureWeighting


class ScaleFoot(BaseModel):
    """A user transform, addressed in YAML by its dotted import path."""

    model_config = ConfigDict(frozen=True)

    factor: float = 1.0

    def apply(self, particles, context=None):
        out = particles.copy()
        out["foot"] = out["foot"] * self.factor
        return out


SCALE_FOOT_KIND = f"{__name__}.ScaleFoot"

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


def test_error_enabled_false_by_default():
    assert ErrorParams().error_enabled is False


def test_error_enabled_true_when_xy_params_set():
    e = ErrorParams(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)
    assert e.error_enabled is True


def test_error_enabled_true_when_zi_params_set():
    e = ErrorParams(sigzierr=0.6, tlzierr=60.0, horcorzierr=40.0)
    assert e.error_enabled is True


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
    """maxpar stays unset in the config; SETUP.CFG gets numpar in its place."""
    p = STILTParams(numpar=500)
    assert p.maxpar is None
    assert p.setup_entries()["maxpar"] == 500
    assert STILTParams(numpar=500, maxpar=800).setup_entries()["maxpar"] == 800


def test_stilt_params_ziscale_defaults_to_scalar_one():
    p = STILTParams()
    assert p.ziscale == 1.0


def test_zicontroltf_is_derived_from_ziscale():
    assert STILTParams().zicontroltf == 0
    assert STILTParams().ziscale_factors is None
    assert STILTParams(ziscale=[1.0, 1.0]).zicontroltf == 0
    assert STILTParams(ziscale=0.8).zicontroltf == 1
    assert STILTParams(ziscale=[1.0, 0.9]).zicontroltf == 1


def test_ziscale_scalar_repeats_for_every_hour_and_list_is_used_as_given():
    assert STILTParams(n_hours=-3, ziscale=0.8).ziscale_factors == [0.8, 0.8, 0.8]
    assert STILTParams(n_hours=-24, ziscale=[0.8]).ziscale_factors == [0.8]
    assert STILTParams(ziscale=[[0.8, 0.9]]).ziscale_factors == [0.8, 0.9]


def test_setup_entries_write_the_derived_zicontroltf():
    assert STILTParams().setup_entries()["zicontroltf"] == 0
    assert STILTParams(ziscale=1.2).setup_entries()["zicontroltf"] == 1


def test_saved_config_no_longer_carries_zicontroltf():
    assert "zicontroltf" not in STILTParams(ziscale=0.8).model_dump()


def test_zicontroltf_is_not_a_setting():
    with pytest.raises(ValidationError, match="zicontroltf"):
        STILTParams(zicontroltf=1)


@pytest.mark.parametrize("ziscale", [0.0, [1.0, 0.0]])
def test_ziscale_zero_is_rejected(ziscale):
    with pytest.raises(ValueError, match="ziscale of 0"):
        STILTParams(ziscale=ziscale)


def test_ziscale_empty_list_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        STILTParams(ziscale=[])


def test_ziscale_rejects_multiple_per_simulation_lists():
    with pytest.raises(ValueError, match="Per-simulation ziscale lists"):
        STILTParams(ziscale=[[0.8, 0.8], [0.9, 0.9]])


def test_ziscale_at_most_150_hourly_factors():
    assert len(STILTParams(n_hours=-150, ziscale=0.8).ziscale_factors) == 150
    assert STILTParams(n_hours=-240, ziscale=[0.8] * 150).zicontroltf == 1
    with pytest.raises(ValueError, match="at most 150"):
        STILTParams(n_hours=-151, ziscale=0.8)
    with pytest.raises(ValueError, match="at most 150"):
        STILTParams(ziscale=[0.8] * 151)


def test_long_run_without_scaling_is_fine():
    assert STILTParams(n_hours=-240).zicontroltf == 0


def test_setup_entries_route_transport_params_to_setup_cfg():
    p = STILTParams()
    entries = p.setup_entries()

    assert entries["numpar"] == p.numpar
    assert entries["varsiwant"] == p.varsiwant
    assert entries["ichem"] == 8
    assert entries["idsp"] == 2
    # CONTROL / ZICONTROL / WINDERR fields never appear in SETUP.CFG
    for name in ("n_hours", "emisshrs", "w_option", "z_top", "ziscale", "siguverr"):
        assert name not in entries
    # None-valued fields are omitted
    assert "seed" not in entries
    assert "maxpar" in entries  # defaulted from numpar


# Fortran type of every SETUP.CFG entry PYSTILT writes, from the SETUP
# namelist declarations in HYSPLIT's hysetup.f. The bundled v5.1.0 build
# reads the same types.
HYSPLIT_SETUP_TYPES = {
    **dict.fromkeys(
        [
            "capemin", "delt", "dxf", "dyf", "dzf", "frhmax", "frhs", "frme",
            "frmr", "frts", "frvs", "hscale", "p10f", "qcycle", "splitf",
            "tkerd", "tkern", "tlfrac", "tratio", "tvmix", "veght", "vscale",
            "vscales", "vscaleu", "wbbh", "wbwf", "wbwr",
        ],
        "REAL",
    ),
    **dict.fromkeys(
        [
            "cmass", "conage", "cpack", "ichem", "idsp", "initd", "k10m",
            "kagl", "kbls", "kblt", "kdef", "khinp", "khmax", "kmix0", "kmixd",
            "kmsl", "kpuff", "krand", "krnd", "kspl", "kwet", "kzmix", "maxdim",
            "maxpar", "mgmin", "mhrs", "nbptyp", "ncycl", "ndump", "ninit",
            "nstr", "numpar", "nturb", "nver", "outdt", "rhb", "rht", "seed",
            "tout", "zicontroltf",
        ],
        "INTEGER",
    ),
    **dict.fromkeys(["efile", "pinbc", "pinpf", "poutf", "varsiwant"], "CHARACTER"),
    "wvert": "LOGICAL",
}  # fmt: skip


def test_hysplit_setup_types_cover_every_setup_entry():
    entries = STILTParams(seed=1, krand=2, kmsl=0).setup_entries()
    assert set(entries) == set(HYSPLIT_SETUP_TYPES)


@pytest.mark.parametrize(
    "name", [n for n, t in HYSPLIT_SETUP_TYPES.items() if t == "REAL"]
)
def test_real_setup_fields_accept_fractions(name):
    # HYSPLIT reads these as REAL, so a fractional value is valid.
    assert STILTParams(**{name: 0.5}).setup_entries()[name] == 0.5


@pytest.mark.parametrize(
    "name",
    [
        n
        for n, t in HYSPLIT_SETUP_TYPES.items()
        if t == "INTEGER" and n in STILTParams.model_fields
    ],
)
def test_integer_setup_fields_reject_fractions(name):
    # HYSPLIT stops with a namelist read error on a fractional INTEGER.
    with pytest.raises(ValidationError):
        STILTParams(**{name: 0.5})


def test_setup_entries_map_seed_to_negative_namelist_value():
    # HYSPLIT's ran1 re-initializes only from a negative value; -(|seed|+1)
    # keeps every seed distinct and off the unseeded default (state 1).
    assert STILTParams(seed=17, krand=2).setup_entries()["seed"] == -18
    assert STILTParams(seed=-17, krand=2).setup_entries()["seed"] == -18
    assert STILTParams(seed=0, krand=2).setup_entries()["seed"] == -1
    assert STILTParams.setup_seed(42) == -43


def test_realization_seeds_are_distinct_and_start_at_the_main_seed():
    p = STILTParams(seed=42, krand=2)
    assert [p.realization_seed(k) for k in range(3)] == [42, 43, 44]
    assert STILTParams().realization_seed(0) is None


@pytest.mark.parametrize("krand", [0, 1, 3, 4, 10, 13])
def test_seed_requires_krand_2(krand):
    with pytest.raises(ValueError, match="requires krand=2"):
        STILTParams(seed=1, krand=krand)


@pytest.mark.parametrize("krand", [0, 1, 2, 3, 4, 10, 11, 12, 13])
def test_krand_accepts_hysplit_modes(krand):
    assert STILTParams(krand=krand).krand == krand


@pytest.mark.parametrize("krand", [-1, 5, 9, 14, 20])
def test_krand_rejects_undocumented_values(krand):
    with pytest.raises(ValueError, match="krand"):
        STILTParams(krand=krand)


def test_control_and_zicontrol_fields_are_transport_params():
    for name in STILTParams.CONTROL_FIELDS - {"n_hours"}:
        assert name in TransportParams.model_fields
    assert "n_hours" in ModelParams.model_fields
    for name in STILTParams.ZICONTROL_FIELDS:
        assert name in TransportParams.model_fields


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
# ProjectConfig - flat construction
# ---------------------------------------------------------------------------


def test_model_config_flat_construction(tmp_path):
    cfg = ProjectConfig(
        n_hours=-24,
        numpar=100,
        seed=42,
        krand=2,
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
    """ProjectConfig must have at least one met entry."""
    with pytest.raises(Exception, match="at least one"):
        ProjectConfig(mets={})


def test_model_config_rejects_met_keys_that_cannot_name_a_variant(tmp_path):
    mc = MetConfig(directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h")
    with pytest.raises(Exception, match="must match"):
        ProjectConfig(mets={"hrrr_v2": mc})


def test_model_config_footprint_fields_are_flat(tmp_path, grid):
    """The footprint settings sit beside the transport ones and form the footprint."""
    mc = MetConfig(directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h")
    cfg = ProjectConfig(mets={"hrrr": mc}, grid=grid, smooth_factor=0.5)
    foot = cfg.footprint
    assert foot is not None
    assert foot.grid == grid
    assert foot.smooth_factor == 0.5
    assert ProjectConfig(mets={"hrrr": mc}).footprint is None


# ---------------------------------------------------------------------------
# ProjectConfig YAML roundtrip
# ---------------------------------------------------------------------------


def test_model_config_yaml_roundtrip_basic(tmp_path):
    cfg = ProjectConfig(
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
    loaded = ProjectConfig.from_yaml(path)
    text = path.read_text()

    assert "#" not in text
    assert loaded.n_hours == -24
    assert loaded.numpar == 100
    assert "hrrr" in loaded.mets
    assert loaded.mets["hrrr"].file_format == "%Y%m%d_%H"
    assert loaded.mets["hrrr"].file_tres == "1h"


def test_model_config_yaml_roundtrip_with_execution(tmp_path):
    cfg = ProjectConfig(
        mets=_met_config(tmp_path),
        execution={
            "backend": "slurm",
            "n_workers": 8,
            "time": "02:00:00",
            "partition": "compute",
            "account": "lin-group",
            "slurm": {"exclude": "node1"},
        },
    )
    path = tmp_path / "config.yaml"
    text = cfg.to_yaml(path)
    assert "cpus" not in text and "array_parallelism" not in text  # unset stays out
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.execution == cfg.execution
    assert loaded.execution.backend == "slurm"
    assert loaded.execution.n_workers == 8
    assert loaded.execution.slurm == {"exclude": "node1"}


def test_execution_defaults_to_one_local_process(tmp_path):
    execution = ProjectConfig(mets=_met_config(tmp_path)).execution
    assert (execution.backend, execution.n_workers, execution.cpus) == ("local", 1, 1)
    assert "execution:" not in ProjectConfig(mets=_met_config(tmp_path)).to_yaml()


def test_unknown_execution_setting_is_an_error(tmp_path):
    """A typo, or an sbatch option outside `slurm:`, used to be ignored or passed on (#63)."""
    with pytest.raises(ValueError, match=r"execution\.partion\s+Extra inputs"):
        ProjectConfig(mets=_met_config(tmp_path), execution={"partion": "compute"})
    with pytest.raises(ValueError, match=r"execution\.requeue\s+Extra inputs"):
        ProjectConfig(mets=_met_config(tmp_path), execution={"requeue": True})
    with pytest.raises(ValueError, match=r"execution\.cpus_per_task\s+Extra inputs"):
        ProjectConfig(mets=_met_config(tmp_path), execution={"cpus_per_task": 4})
    with pytest.raises(ValueError, match="backend"):
        ProjectConfig(mets=_met_config(tmp_path), execution={"backend": "kubernetes"})
    with pytest.raises(ValueError, match="n_workers"):
        ProjectConfig(mets=_met_config(tmp_path), execution={"n_workers": 0})


def test_execution_accepts_one_setup_line():
    from stilt.config import ExecutionConfig

    execution = ExecutionConfig.model_validate({"setup": "module load hysplit"})
    assert execution.setup == ["module load hysplit"]


def test_model_config_yaml_roundtrip_with_footprint(tmp_path, grid):
    """The footprint settings survive a to_yaml/from_yaml roundtrip."""
    cfg = ProjectConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        grid=grid,
        time_integrate=True,
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.footprint is not None
    assert loaded.footprint.model_dump() == cfg.footprint.model_dump()


def _met_config(tmp_path):
    return {
        "hrrr": MetConfig(
            directory=tmp_path / "met",
            file_format="%Y%m%d_%H",
            file_tres="1h",
        )
    }


def test_model_config_yaml_roundtrip_with_footprint_transforms(tmp_path, grid):
    given = [
        AveragingKernel(levels=[0.0, 1000.0], values=[0.2, 0.8], coordinate="xhgt"),
        PressureWeighting(),
        FirstOrderLifetime(lifetime_hours=4.0),
    ]
    cfg = ProjectConfig(mets=_met_config(tmp_path), grid=grid, transforms=given)
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.transforms
    assert len(transforms) == 3
    assert transforms == given
    assert isinstance(transforms[0], AveragingKernel)
    assert isinstance(transforms[1], PressureWeighting)
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert loaded.resolve_variants()["hrrr"].footprint.transforms == given


def test_model_config_yaml_roundtrip_with_user_transform(tmp_path, grid):
    cfg = ProjectConfig(
        mets=_met_config(tmp_path), grid=grid, transforms=[ScaleFoot(factor=2.5)]
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)

    text = path.read_text()
    assert f"kind: {SCALE_FOOT_KIND}" in text
    assert "factor: 2.5" in text

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.transforms
    assert len(transforms) == 1
    assert isinstance(transforms[0], ScaleFoot)
    assert transforms[0].factor == pytest.approx(2.5)
    assert transforms[0] == cfg.transforms[0]


def test_model_config_variant_grid_in_yaml(tmp_path):
    """A variant may carry its own grid; the others inherit the default."""
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
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
    variants = ProjectConfig.from_yaml(path).resolve_variants()
    assert variants["hrrr"].footprint.grid.xres == 0.01
    assert variants["coarse"].footprint.grid.xres == 0.05
    assert variants["coarse"].footprint.grid.xmin == -114.0


def test_model_config_inline_grid_in_yaml(tmp_path):
    """The default grid is a mapping under ``grid``."""
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
        grid:
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
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.grid is not None
    assert loaded.grid.xres == 0.05
    assert loaded.grid.xmin == -114.0
    assert loaded.smooth_factor == 0.75


def test_model_config_null_grid_means_trajectory_only(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
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
            grid: null
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    variants = ProjectConfig.from_yaml(path).resolve_variants()
    assert variants["hrrr"].footprint is not None
    assert variants["traj"].footprint is None


def test_model_config_loads_footprint_transforms_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
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
            coordinate: xhgt
          - kind: pressure_weighting
          - kind: first_order_lifetime
            lifetime_hours: 3.0
          - kind: {SCALE_FOOT_KIND}
            factor: 0.5
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.transforms

    assert len(transforms) == 4
    assert isinstance(transforms[0], AveragingKernel)
    assert transforms[0].levels == [0.0, 1000.0]
    assert transforms[0].values == [0.3, 0.7]
    assert transforms[0].coordinate == "xhgt"
    assert isinstance(transforms[1], PressureWeighting)
    assert transforms[1].surface_pressure is None
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert transforms[2].lifetime_hours == pytest.approx(3.0)
    assert isinstance(transforms[3], ScaleFoot)
    assert transforms[3].factor == pytest.approx(0.5)


def test_model_config_rejects_unimportable_transform_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(f"""\
        n_hours: -24
        numpar: 100
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
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
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ImportError, match="could not be imported"):
        ProjectConfig.from_yaml(path)


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
        ProjectConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# FootprintConfig.geometry
# ---------------------------------------------------------------------------


def test_footprint_config_derives_grid_from_windows_geometry():
    fc = FootprintConfig(
        geometry={
            "kind": "windows",
            "coords": [(-111.97, 40.515), (-112.015, 40.779)],
            "size": 0.01,
            "ids": ["landfill", "wwtp"],
        }
    )
    assert fc.grid is None  # derived when resolved, not when loaded
    fc = fc.resolve()
    assert fc.geometry is not None and fc.geometry.kind == "windows"
    assert fc.grid.xres == fc.grid.yres == pytest.approx(0.002)  # 0.01 / 4 -> 0.002
    assert fc.grid.xmin <= -112.02 and fc.grid.ymax >= 40.784
    mesh = fc.geometry.build()
    assert mesh.ids == ("landfill", "wwtp")


def test_footprint_config_cells_per_target_and_explicit_grid_wins():
    spec = {"kind": "windows", "coords": [(0.5, 0.5)], "size": 0.1}
    fc = FootprintConfig(geometry=spec, cells_per_target=10).resolve()
    assert fc.grid.xres == pytest.approx(0.01)
    explicit = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.05, yres=0.05)
    fc2 = FootprintConfig(grid=explicit, geometry=spec).resolve()
    assert fc2.grid == explicit and fc2.geometry is not None


def test_footprint_needs_a_grid(point_receptor):
    import pandas as pd

    from stilt.footprint import calculate

    assert FootprintConfig().grid is None
    with pytest.raises(ValueError, match="grid"):
        calculate(pd.DataFrame(), point_receptor, FootprintConfig())


def test_footprint_config_h3_geometry():
    pytest.importorskip("h3")
    fc = FootprintConfig(
        geometry={
            "kind": "h3",
            "resolution": 8,
            "bounds": {"xmin": -112.0, "xmax": -111.8, "ymin": 40.6, "ymax": 40.8},
        }
    ).resolve()
    assert fc.grid.xres <= 0.0025  # res-8 hexagons are ~0.5 km across
    assert len(fc.geometry.build()) > 50


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
        mets={
            "hrrr": {
                "directory": tmp_path,
                "file_format": "%Y%m%d_%H",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        geometry={"kind": "file", "path": str(shp), "ids": "NAME"},
    )
    fc = config.footprint
    assert fc is not None
    assert fc.grid.xres == pytest.approx(0.02)  # 0.1 / 4 -> 0.025 -> 0.02
    assert fc.geometry is not None and fc.geometry.kind == "file"

    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    lfc = loaded.footprint
    assert lfc is not None
    assert lfc.grid == fc.grid
    assert lfc.geometry == fc.geometry
    assert lfc.geometry.build().ids == ("a", "b")


def test_file_geometry_spec_layer_and_where(tmp_path):
    gpd = pytest.importorskip("geopandas")
    import shapely

    from stilt.config import FileGeometrySpec

    gdf = gpd.GeoDataFrame(
        {"NAME": ["a", "b", "c"], "KEEP": [1, 1, 0]},
        geometry=[shapely.box(i, 0, i + 1, 1) for i in range(3)],
        crs="EPSG:4326",
    )
    gpkg = tmp_path / "cells.gpkg"
    gdf.to_file(gpkg, layer="cells", driver="GPKG")
    gdf.iloc[:1].to_file(gpkg, layer="other", driver="GPKG")

    spec = FileGeometrySpec(path=str(gpkg), ids="NAME", layer="cells", where="KEEP=1")
    mesh = spec.build()
    assert mesh.ids == ("a", "b")
    assert FileGeometrySpec(path=str(gpkg), layer="other").build().ids == ("0",)


# -- variants --------------------------------------------------------------------


def _variant_config(tmp_path, **kwargs):
    return ProjectConfig(mets=_met_config(tmp_path), **kwargs)


def test_footprint_settings_without_a_grid_are_an_error(tmp_path):
    """Settings that would be dropped are refused instead."""
    from stilt.transforms import FirstOrderLifetime

    with pytest.raises(
        ValueError, match="'decay' sets footprint settings .transforms. but no grid"
    ):
        _variant_config(
            tmp_path,
            variants={
                "decay": {"transforms": [FirstOrderLifetime(lifetime_hours=1.0)]}
            },
        )
    with pytest.raises(ValueError, match="smooth_factor"):
        _variant_config(tmp_path, smooth_factor=2)


def test_variants_default_to_one_per_met(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(mets={"hrrr": mc, "gfs": mc}, ziscale=0.9)
    variants = cfg.resolve_variants()
    assert list(variants) == ["hrrr", "gfs"]
    assert variants["gfs"].met == "gfs"
    assert variants["gfs"].transport.ziscale == 0.9
    assert variants["hrrr"].transport.ziscale == 0.9


def test_variant_overrides_merge_onto_the_defaults(tmp_path):
    cfg = _variant_config(
        tmp_path, numpar=50, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    variants = cfg.resolve_variants()
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
    variants = cfg.resolve_variants()
    assert variants["coarse"].footprint is not None
    assert variants["coarse"].footprint.grid.xres == 0.1
    assert variants["coarse"].footprint.grid.xmin == -114
    assert variants["none"].footprint is None


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
    assert "capemin" not in text and "seed" not in text  # defaults stay out
    assert "maxpar" not in text and "n_min" not in text  # at every level
    assert text.startswith("mets:")  # inputs first, then the settings
    assert "kind: first_order_lifetime" in text  # nested objects are written in full
    loaded = ProjectConfig.from_yaml(path).resolve_variants()
    assert loaded["zi08"].transport.ziscale == 0.8
    assert loaded["decay"].footprint.transforms[0].lifetime_hours == 1.0


def test_variant_must_name_its_met_when_there_are_several(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    with pytest.raises(ValueError, match="must name its met"):
        ProjectConfig(mets={"hrrr": mc, "gfs": mc}, variants={"a": {}})
    with pytest.raises(ValueError, match="unknown met"):
        ProjectConfig(mets={"hrrr": mc}, variants={"a": {"met": "nam"}})


@pytest.mark.parametrize("name", ["Hrrr", "hrrr_v2", "-x", "a b"])
def test_variant_names_are_plain(tmp_path, name):
    with pytest.raises(ValueError, match="must match"):
        _variant_config(tmp_path, variants={name: {}})


def test_realizations_expand_into_numbered_variants_with_their_own_seed(tmp_path):
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
    variants = cfg.resolve_variants()
    assert list(variants) == ["hrrr", "err-0", "err-1", "err-2"]
    assert [variants[f"err-{k}"].transport.seed for k in range(3)] == [42, 43, 44]
    assert all(variants[f"err-{k}"].group == "err" for k in range(3))
    assert [variants[f"err-{k}"].realization for k in range(3)] == [0, 1, 2]
    assert variants["err-1"].transport.winderrtf == 1
    assert variants["hrrr"].transport.winderrtf == 0


@pytest.mark.parametrize("krand", [0, 1, 2, 3, 12])
def test_several_realizations_require_krand_4_or_a_seed(tmp_path, krand):
    with pytest.raises(ValueError, match="requires krand=4 or krand=2 with a seed"):
        _variant_config(tmp_path, krand=krand, variants={"e": {"realizations": 3}})


def test_several_realizations_accept_krand_4_or_seeded_krand_2(tmp_path):
    a = _variant_config(tmp_path, krand=4, variants={"e": {"realizations": 3}})
    assert len(a.resolve_variants()) == 3
    b = _variant_config(tmp_path, krand=2, seed=7, variants={"e": {"realizations": 2}})
    assert len(b.resolve_variants()) == 2
    with pytest.raises(ValueError, match="realizations must be >= 1"):
        _variant_config(tmp_path, variants={"e": {"realizations": 0}})


def test_declaring_realizations_always_makes_a_group(tmp_path):
    """A group of one is ``e-0``, so raising the count later only adds runs."""
    cfg = _variant_config(tmp_path, variants={"e": {"realizations": 1}, "single": {}})
    assert list(cfg.resolve_variants()) == ["e-0", "single"]
    assert cfg.resolve_variants()["e-0"].group == "e"
    assert cfg.resolve_variants()["single"].realization is None


def test_realization_names_may_not_collide_with_declared_variants(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        _variant_config(
            tmp_path, krand=4, variants={"e": {"realizations": 2}, "e-1": {}}
        )


def test_variants_that_differ_only_in_footprint_fields_keep_the_transport(
    tmp_path, grid
):
    cfg = _variant_config(
        tmp_path,
        grid=grid,
        variants={"hrrr": {"ziscale": 0.8}, "s2": {"ziscale": 0.8, "smooth_factor": 2}},
    )
    variants = cfg.resolve_variants()
    s2 = variants["s2"]
    assert s2.met == "hrrr"
    assert s2.transport.ziscale == 0.8
    assert s2.footprint is not None and s2.footprint.smooth_factor == 2
    assert s2.transport == variants["hrrr"].transport
    assert s2.transport.hash == variants["hrrr"].transport.hash


def test_from_is_rejected_with_advice(tmp_path):
    with pytest.raises(ValueError, match="no longer needed"):
        _variant_config(tmp_path, variants={"a": {}, "b": {"from": "a"}})


def test_variants_survive_a_yaml_roundtrip_as_written(tmp_path, grid):
    declared = {
        "hrrr": {},
        "zi08": {"ziscale": 0.8},
        "s2": {"smooth_factor": 2},
    }
    cfg = _variant_config(tmp_path, grid=grid, variants=declared)
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.variants == declared
    assert list(loaded.resolve_variants()) == ["hrrr", "zi08", "s2"]


def test_to_yaml_always_writes_the_variants_that_run(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(mets={"hrrr": mc, "gfs": mc}, numpar=50)
    text = cfg.to_yaml()
    assert "variants:\n  hrrr: {}\n  gfs: {}\n" in text
    loaded = ProjectConfig.model_validate(__import__("yaml").safe_load(text))
    assert list(loaded.resolve_variants()) == ["hrrr", "gfs"]
    assert loaded.resolve_variants()["gfs"].met == "gfs"


def test_variant_named_after_a_met_uses_it(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(mets={"hrrr": mc, "gfs": mc}, variants={"gfs": {}, "hrrr": {}})
    assert {n: v.met for n, v in cfg.resolve_variants().items()} == {
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
    variants = cfg.resolve_variants()
    mesh = variants["src"].footprint.geometry.build()
    expected = Grid.from_geometry(mesh)
    for name in ("src", "src-run"):
        assert variants[name].footprint.grid == expected
        assert variants[name].footprint.geometry_hash == mesh.hash
    assert variants["hrrr"].footprint.grid != expected


def _count_builds(monkeypatch):
    """Count calls to ``WindowsGeometrySpec.build``."""
    from stilt.config import WindowsGeometrySpec

    calls = []
    build = WindowsGeometrySpec.build

    def counting(self):
        calls.append(self)
        return build(self)

    monkeypatch.setattr(WindowsGeometrySpec, "build", counting)
    return calls


def test_each_geometry_is_built_once_and_not_on_load(tmp_path, monkeypatch):
    """Loading a config reads no geometry; resolving builds each one once (#64)."""
    calls = _count_builds(monkeypatch)
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(f"""
        mets:
          hrrr: {{directory: {tmp_path}/met, file_format: '%Y%m%d_%H', file_tres: 1h}}
        geometry: {{kind: windows, coords: [[-111.5, 40.2]], size: 0.05}}
        variants:
          hrrr: {{}}
          np50: {{numpar: 50}}
          err: {{realizations: 2, seed: 1, krand: 2}}
          src:
            geometry: {{kind: windows, coords: [[-111.97, 40.515]], size: 0.01}}
        """)
    )
    cfg = ProjectConfig.from_yaml(tmp_path / "config.yaml")
    assert calls == []

    variants = cfg.resolve_variants()
    assert len(calls) == 2  # the default geometry and the one of "src"
    inherited = [variants[n].footprint for n in ("hrrr", "np50", "err-0", "err-1")]
    assert all(f is not None and f.grid == inherited[0].grid for f in inherited)
    assert cfg.footprint == inherited[0]
    assert len(calls) == 2


def test_config_loads_without_its_geometry_file(tmp_path):
    """A worker can load the config even where the geometry file is not readable."""
    cfg = _variant_config(
        tmp_path, geometry={"kind": "file", "path": str(tmp_path / "missing.shp")}
    )
    assert cfg.geometry is not None and cfg.grid is None
    with pytest.raises(Exception, match="missing.shp"):
        cfg.resolve_variants()


def test_resolved_settings_do_not_build_again(monkeypatch):
    """Settings read back with a grid and geometry hash never read the geometry."""
    calls = _count_builds(monkeypatch)
    explicit = Grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xres=0.05, yres=0.05)
    stored = FootprintConfig(
        grid=explicit,
        geometry={"kind": "windows", "coords": [(0.5, 0.5)], "size": 0.1},
        geometry_hash="deadbeef00",
    )
    assert stored.resolve() is stored
    assert calls == []


def test_to_yaml_does_not_write_the_derived_grid(tmp_path):
    cfg = _variant_config(tmp_path, geometry=_WINDOWS)
    assert cfg.footprint is not None and cfg.footprint.grid is not None
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
    variants = cfg.resolve_variants()
    assert variants["hrrr"].transport.setup_entries()["maxpar"] == 1000
    assert variants["np3k"].transport.setup_entries()["maxpar"] == 3000
    # A run's identity holds the value HYSPLIT got, so an unset maxpar equals
    # numpar and a variant that raises numpar is a different run.
    assert variants["hrrr"].transport.identity()["maxpar"] == 1000
    assert variants["np3k"].transport.identity()["maxpar"] == 3000
