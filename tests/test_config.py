"""Tests for stilt.config - models and YAML roundtrip."""

import re
import textwrap

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from stilt.config import ProjectConfig
from stilt.footprint.config import FootprintConfig
from stilt.footprint.targets import Mesh
from stilt.identity import footprint_settings, read_footprint_settings
from stilt.spatial import Grid
from stilt.transforms import AveragingKernel, FirstOrderLifetime, PressureWeighting
from stilt.transport.hysplit.driver import (
    setup_entries,
    winderrtf,
)

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
# ProjectConfig - flat construction
# ---------------------------------------------------------------------------


def test_model_config_flat_construction(tmp_path):
    cfg = ProjectConfig(
        n_hours=-24,
        numpar=100,
        seed=42,
        krand=2,
        mets={
            "hrrr": make_met_config(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        variants={"hrrr": {}},
    )
    assert cfg.n_hours == -24
    assert cfg.numpar == 100
    assert cfg.seed == 42


def test_model_config_requires_nonempty_mets():
    """ProjectConfig must have at least one met entry."""
    with pytest.raises(Exception, match="at least one"):
        ProjectConfig(mets={}, variants={"hrrr": {}})


def test_model_config_rejects_met_keys_that_cannot_name_a_variant(tmp_path):
    mc = make_met_config(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )
    with pytest.raises(Exception, match="must match"):
        ProjectConfig(mets={"hrrr_v2": mc}, variants={"hrrr_v2": {}})


def test_a_config_loads_without_its_hysplit_build(tmp_path):
    """Loading checks the variants but builds none, so exe_dir is not needed yet."""
    mc = make_met_config(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )
    elsewhere = tmp_path / "not-mounted"
    cfg = ProjectConfig(mets={"hrrr": mc}, exe_dir=elsewhere, variants={"hrrr": {}})
    with pytest.raises(FileNotFoundError, match="version"):
        cfg.resolve()
    with pytest.raises(ValueError, match="realizations must be >= 1"):
        ProjectConfig(mets={"hrrr": mc}, variants={"e": {"realizations": 0}})


def _default_footprint(cfg):
    """The footprint settings of the first variant, which inherits the defaults."""
    return next(iter(cfg.resolve().values())).footprint


def test_model_config_footprint_fields_are_flat(tmp_path, grid):
    """The footprint settings sit beside the transport ones and form the footprint."""
    mc = make_met_config(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )
    cfg = ProjectConfig(
        mets={"hrrr": mc}, grid=grid, smooth_factor=0.5, variants={"hrrr": {}}
    )
    foot = _default_footprint(cfg)
    assert foot is not None
    assert foot.grid == grid
    assert foot.smooth_factor == 0.5
    assert (
        _default_footprint(ProjectConfig(mets={"hrrr": mc}, variants={"hrrr": {}}))
        is None
    )


# ---------------------------------------------------------------------------
# ProjectConfig YAML roundtrip
# ---------------------------------------------------------------------------


def test_model_config_yaml_roundtrip_basic(tmp_path):
    cfg = ProjectConfig(
        n_hours=-24,
        numpar=100,
        mets={
            "hrrr": make_met_config(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        variants={"hrrr": {}},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    text = path.read_text()

    assert "#" not in text
    assert loaded.n_hours == -24
    assert loaded.numpar == 100
    assert "hrrr" in loaded.mets
    assert loaded.mets["hrrr"]["file_format"] == "%Y%m%d_%H"
    assert loaded.mets["hrrr"]["file_tres"] == "1h"


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
        variants={"hrrr": {}},
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
    execution = ProjectConfig(
        mets=_met_config(tmp_path), variants={"hrrr": {}}
    ).execution
    assert (execution.backend, execution.n_workers, execution.cpus) == ("local", 1, 1)
    assert (
        "execution:"
        not in ProjectConfig(
            mets=_met_config(tmp_path), variants={"hrrr": {}}
        ).to_yaml()
    )


def test_unknown_execution_setting_is_an_error(tmp_path):
    """A typo, or an sbatch option outside `slurm:`, used to be ignored or passed on (#63)."""
    with pytest.raises(ValueError, match=r"execution\.partion\s+Extra inputs"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            execution={"partion": "compute"},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match=r"execution\.requeue\s+Extra inputs"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            execution={"requeue": True},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match=r"execution\.cpus_per_task\s+Extra inputs"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            execution={"cpus_per_task": 4},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="backend"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            execution={"backend": "kubernetes"},
            variants={"hrrr": {}},
        )
    with pytest.raises(ValueError, match="n_workers"):
        ProjectConfig(
            mets=_met_config(tmp_path),
            execution={"n_workers": 0},
            variants={"hrrr": {}},
        )


def test_execution_accepts_one_setup_line():
    from stilt.execution.config import ExecutionConfig

    execution = ExecutionConfig.model_validate({"setup": "module load hysplit"})
    assert execution.setup == ["module load hysplit"]


def test_model_config_yaml_roundtrip_with_footprint(tmp_path, grid):
    """The footprint settings survive a to_yaml/from_yaml roundtrip."""
    cfg = ProjectConfig(
        mets={
            "hrrr": make_met_config(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        grid=grid,
        time_integrate=True,
        variants={"hrrr": {}},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert _default_footprint(loaded) is not None
    assert (
        _default_footprint(loaded).model_dump() == _default_footprint(cfg).model_dump()
    )


def _met_config(tmp_path):
    return {
        "hrrr": make_met_config(
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
    cfg = ProjectConfig(
        mets=_met_config(tmp_path), grid=grid, transforms=given, variants={"hrrr": {}}
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.footprint.transforms
    assert len(transforms) == 3
    assert transforms == given
    assert isinstance(transforms[0], AveragingKernel)
    assert isinstance(transforms[1], PressureWeighting)
    assert isinstance(transforms[2], FirstOrderLifetime)
    assert loaded.resolve()["hrrr"].footprint.transforms == given


def test_model_config_yaml_roundtrip_with_user_transform(tmp_path, grid):
    cfg = ProjectConfig(
        mets=_met_config(tmp_path),
        grid=grid,
        transforms=[ScaleFoot(factor=2.5)],
        variants={"hrrr": {}},
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)

    text = path.read_text()
    assert f"kind: {SCALE_FOOT_KIND}" in text
    assert "factor: 2.5" in text

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.footprint.transforms
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
    variants = ProjectConfig.from_yaml(path).resolve()
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
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.footprint.grid is not None
    assert loaded.footprint.grid.xres == 0.05
    assert loaded.footprint.grid.xmin == -114.0
    assert loaded.footprint.smooth_factor == 0.75


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
    variants = ProjectConfig.from_yaml(path).resolve()
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
        variants:
          hrrr: {{}}
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    loaded = ProjectConfig.from_yaml(path)
    transforms = loaded.footprint.transforms

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
        n_hours: -24
        mets:
          hrrr:
            directory: {tmp_path / "met"}
            file_format: "%Y%m%d_%H"
            file_tres: 1h
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
            mets=_met_config(tmp_path), **{"variants": {"hrrr": {}}, **footprint}
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
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        geometry={"kind": "file", "path": str(shp), "ids": "NAME"},
        variants={"hrrr": {}},
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
        mets=_met_config(tmp_path), **{"variants": {"hrrr": {}}, **kwargs}
    )


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
    cfg = ProjectConfig(
        mets={"hrrr": mc, "gfs": mc}, ziscale=0.9, variants={"hrrr": {}, "gfs": {}}
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
    loaded = ProjectConfig.from_yaml(path).resolve()
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
        "s2": {"smooth_factor": 2},
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
        mets={"hrrr": mc, "gfs": mc}, numpar=50, variants={"hrrr": {}, "gfs": {}}
    )
    text = cfg.to_yaml()
    assert "variants:\n  hrrr: {}\n  gfs: {}\n" in text
    loaded = ProjectConfig.model_validate(__import__("yaml").safe_load(text))
    assert list(loaded.resolve()) == ["hrrr", "gfs"]
    assert loaded.resolve()["gfs"].met == "gfs"


def test_variant_named_after_a_met_uses_it(tmp_path):
    mc = _met_config(tmp_path)["hrrr"]
    cfg = ProjectConfig(mets={"hrrr": mc, "gfs": mc}, variants={"gfs": {}, "hrrr": {}})
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
    assert cfg.footprint.geometry is not None and cfg.footprint.grid is None
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
    assert setup_entries(variants["hrrr"].transport)["maxpar"] == 1000
    assert setup_entries(variants["np3k"].transport)["maxpar"] == 3000
    # A run's identity holds the value HYSPLIT got, so an unset maxpar equals
    # numpar and a variant that raises numpar is a different run.
    assert variants["hrrr"].run_settings["maxpar"] == 1000
    assert variants["np3k"].run_settings["maxpar"] == 3000


# ---------------------------------------------------------------------------
# Variants are declared; unknown keys name the nearest setting (#150)
# ---------------------------------------------------------------------------


def test_a_config_without_variants_says_what_to_write(tmp_path):
    with pytest.raises(
        ValueError, match=r"no variants. Add\n  variants:\n    hrrr: \{\}"
    ):
        ProjectConfig(mets=_met_config(tmp_path))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"smooth_factr": 0.5},
            "config: 'smooth_factr' is not a setting. Did you mean 'smooth_factor', a footprint setting?",
        ),
        (
            {"numpr": 10},
            "config: 'numpr' is not a setting. Did you mean 'numpar', a hysplit setting?",
        ),
        ({"varients": {}}, "Did you mean 'variants', a config key?"),
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
        mets=_met_config(tmp_path),
        geometry={"kind": "file", "path": "cells.geojson", "ids": "NAME"},
        variants={"hrrr": {}},
    )
    monkeypatch.chdir(tmp_path)  # not the project
    variant = config.resolve(project)["hrrr"]
    assert variant.footprint is not None and variant.geometry_hash is not None
    # The settings keep the path as written, so moving the project changes no hash.
    assert variant.footprint.geometry.path == "cells.geojson"
