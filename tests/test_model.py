"""Tests for stilt.model: project inputs, variants, the output directory, collections, run()."""

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.collections import (
    FOOTPRINT,
    TRAJECTORY,
    OutputCollection,
    SimulationCollection,
)
from stilt.config import Grid, MetConfig, ModelConfig
from stilt.errors import ConfigValidationError
from stilt.execution import LocalHandle, register, resolve_compute_root
from stilt.footprint import Footprint
from stilt.model import Model
from stilt.output import Output
from stilt.project import CONFIG_KEY, RECEPTORS_KEY
from stilt.receptors import PointReceptor
from stilt.simulation import SimID, Simulation
from stilt.trajectory import Trajectories

matplotlib.use("Agg")

_XYERR = {
    "siguverr": 1.0,
    "tluverr": 60.0,
    "zcoruverr": 100.0,
    "horcoruverr": 10.0,
}

_GRID = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _met(tmp_path) -> MetConfig:
    return MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )


def _config(tmp_path, include_footprint=True, **overrides) -> ModelConfig:
    """Minimal ModelConfig with one met and (optionally) a footprint grid."""
    overrides.setdefault("grid", _GRID if include_footprint else None)
    return ModelConfig(mets={"hrrr": _met(tmp_path)}, **overrides)


def _receptor(hour: int, longitude: float = -111.85) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, hour),
        longitude=longitude,
        latitude=40.77,
        altitude=5.0,
    )


def _rid(receptor) -> str:
    return str(receptor.id)


def _sid(receptor, variant="hrrr") -> SimID:
    return SimID(receptor.id, variant)


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60.0, -120.0],
            "indx": [1.0, 1.0],
            "long": [-113.9, -113.5],
            "lati": [39.7, 39.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def _write_trajectory(model: Model, sid) -> Path:
    """Write a small particle file for *sid* into the output directory."""
    sim = model.simulation(sid)
    traj = Trajectories.from_particles(
        _particles(), receptor=sim.receptor, params=sim.params, met_files=[]
    )
    run = sim.output.run(sim.variant.name, sim.variant.transport)
    return run.write_particles(traj)


def _write_footprint(model: Model, sid, *, empty=False) -> Path:
    """Write a footprint (or an empty one) for *sid* into the output directory."""
    sim = model.simulation(sid)
    assert sim.footprint_config is not None
    run = sim.output.run(sim.variant.name, sim.variant.transport)
    feet = run.footprints(sim.footprint_config, name=sim.variant.name)
    if empty:
        return feet.write_empty(sim.receptor, "outside_domain", name=sim.variant.name)
    grid = sim.footprint_config.grid
    assert grid is not None
    x_axis, y_axis = grid.axes
    data = xr.DataArray(
        np.random.rand(1, len(y_axis), len(x_axis)),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.receptor.time], "lat": y_axis, "lon": x_axis},
    )
    foot = Footprint(
        receptor=sim.receptor,
        config=sim.footprint_config,
        data=data,
        name=sim.variant.name,
    )
    return feet.write(foot)


class _CapturingExecutor:
    """Stand-in for the runner's dispatch: records each call and runs nothing."""

    def __init__(self, handle=None):
        self.handle = handle if handle is not None else LocalHandle()
        self.start_calls: list[dict] = []

    def dispatch(self, model, pending, execution, *, compute_root, skip_existing):
        self.start_calls.append(
            {
                "pending": list(pending),
                "project": model.project.root,
                "execution": execution,
                "compute_root": compute_root,
                "skip_existing": skip_existing,
            }
        )
        return self.handle

    def run(self, model, **kwargs):
        """Run *model* with its receptors handed to this recorder."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("stilt.execution.runner._dispatch", self.dispatch)
            return model.run(**kwargs)

    @property
    def was_started(self) -> bool:
        return len(self.start_calls) > 0


class _CountingHandle:
    job_id = "fake"
    detached = False
    done = True

    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------


def test_receptors_from_csv(tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(project=tmp_path, receptors=csv)

    assert len(model.receptors) == 1
    assert model.receptors[0].latitude == pytest.approx(40.77)


def test_receptors_csv_path_is_relative_to_the_project(tmp_path):
    (tmp_path / "in.csv").write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(project=tmp_path, receptors="in.csv")

    assert len(model.receptors) == 1


def test_model_accepts_empty_receptor_list(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert len(model.receptors) == 0
    assert list(model.receptors) == []
    assert register(model) == []


def test_receptors_support_lookup_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    sid = _sid(point_receptor)

    assert model.receptors[sid.receptor] == point_receptor
    assert sid.receptor in model.receptors
    with pytest.raises(KeyError):
        model.receptors["nope"]


def test_receptors_are_normalized_once(tmp_path):
    model = Model(project=tmp_path, receptors=[_receptor(12)])

    assert model.receptors is model.receptors


def test_receptors_must_be_receptor_objects_or_a_path(tmp_path):
    model = Model(
        project=tmp_path, receptors=[("2023-01-01 12:00", -111.85, 40.77, 5.0)]
    )
    with pytest.raises(TypeError, match="Receptor"):
        len(model.receptors)


def test_receptors_raise_when_nothing_available(tmp_path):
    model = Model(project=tmp_path, receptors=None)

    with pytest.raises(FileNotFoundError, match="No receptors available"):
        len(model.receptors)


def test_receptors_default_to_project_receptors_csv(tmp_path, point_receptor):
    register(
        Model(project=tmp_path, config=_config(tmp_path), receptors=[point_receptor])
    )

    model = Model(project=tmp_path)

    assert len(model.receptors) == 1
    assert model.receptors[0].id == point_receptor.id


# ---------------------------------------------------------------------------
# Config, output, compute_root, queue
# ---------------------------------------------------------------------------


def test_config_raises_when_no_yaml(tmp_path):
    model = Model(project=tmp_path)

    with pytest.raises(FileNotFoundError):
        _ = model.config


def test_config_loads_from_project_yaml(tmp_path):
    _config(tmp_path, numpar=77).to_yaml(tmp_path / CONFIG_KEY)

    model = Model(project=tmp_path)

    assert model.config.numpar == 77
    assert model.config.mets["hrrr"].directory == tmp_path / "met"


def test_model_kwargs_build_config(tmp_path):
    model = Model(project=tmp_path, mets={"hrrr": _met(tmp_path)}, numpar=33)

    assert isinstance(model.config, ModelConfig)
    assert model.config.numpar == 33


def test_model_config_and_kwargs_raises(tmp_path):
    with pytest.raises(TypeError, match="both"):
        Model(project=tmp_path, config=_config(tmp_path), numpar=3)


def test_model_repr(tmp_path):
    model = Model(project=tmp_path / "my_project")

    assert model.project.name == "my_project"
    assert repr(model) == f"Model(project={str(tmp_path / 'my_project')!r})"
    assert model.project.root == str(tmp_path / "my_project")


def test_output_defaults_to_the_projects_output_directory(tmp_path):
    model = Model(project=tmp_path / "proj", config=_config(tmp_path))

    assert isinstance(model.output, Output)
    assert model.output.path == tmp_path / "proj" / "output"
    assert not model.output.path.exists()  # nothing is made by looking


def test_output_can_be_shared_between_projects(tmp_path, point_receptor):
    shared = tmp_path / "shared"
    config = _config(tmp_path, output=str(shared))
    a = Model(project=tmp_path / "a", config=config, receptors=[point_receptor])
    b = Model(project=tmp_path / "b", config=config, receptors=[point_receptor])

    _write_trajectory(a, _sid(point_receptor))

    assert a.output.path == b.output.path == shared
    assert b.simulations[_sid(point_receptor)].has_trajectory
    assert (
        a.simulations[_sid(point_receptor)].run
        == b.simulations[_sid(point_receptor)].run
    )


def test_compute_root_defaults_under_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("PYSTILT_COMPUTE_ROOT", raising=False)

    model = Model(project=tmp_path / "proj", config=_config(tmp_path))

    assert (
        resolve_compute_root(model.project)
        == (tmp_path / "tmp" / "pystilt" / "proj").resolve()
    )


def test_default_compute_root_is_resolved_like_an_explicit_one(tmp_path, monkeypatch):
    """A TMPDIR behind a symlink (as on macOS) gives the path a pool worker gets."""
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "link"))
    monkeypatch.delenv("PYSTILT_COMPUTE_ROOT", raising=False)

    project = Model(project=tmp_path / "proj", config=_config(tmp_path)).project

    default = resolve_compute_root(project)
    assert default == real.resolve() / "pystilt" / "proj"
    assert resolve_compute_root(project, str(default)) == default


def test_compute_root_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))

    project = Model(project=tmp_path / "project", config=_config(tmp_path)).project

    assert resolve_compute_root(project) == (tmp_path / "scratch").resolve()


def test_compute_root_explicit_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "env-scratch"))

    project = Model(project=tmp_path / "project", config=_config(tmp_path)).project

    assert (
        resolve_compute_root(project, tmp_path / "explicit")
        == (tmp_path / "explicit").resolve()
    )


# ---------------------------------------------------------------------------
# Variants and their output
# ---------------------------------------------------------------------------


def test_variants_default_to_one_per_met(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert list(model.variants) == ["hrrr"]
    assert model.variants["hrrr"].met == "hrrr"


def test_simulation_handles_carry_the_variant_settings(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        krand=4,
        variants={
            "hrrr": {},
            "err": {**_XYERR, "realizations": 2, "grid": None},
            "zi08": {"ziscale": 0.8},
            "s2": {"smooth_factor": 2},
        },
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    base = model.simulation(_sid(point_receptor))
    err = model.simulation(_sid(point_receptor, "err-1"))
    zi = model.simulation(_sid(point_receptor, "zi08"))
    s2 = model.simulation(_sid(point_receptor, "s2"))

    assert base.params.winderrtf == 0
    assert err.params.winderrtf == 1
    assert err.footprint_config is None
    assert zi.params.ziscale == 0.8
    assert zi.footprint_config == base.footprint_config
    assert s2.footprint_config is not None and s2.footprint_config.smooth_factor == 2
    assert s2.variant.transport == base.variant.transport  # shares hrrr's particles
    assert model.simulation(str(_sid(point_receptor))) == base  # a value, not a handle


def test_variants_with_equal_transport_settings_share_a_run(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        variants={"hrrr": {}, "s2": {"smooth_factor": 2}, "zi08": {"ziscale": 0.8}},
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    hrrr, s2, zi = (model.variants[v].transport for v in ("hrrr", "s2", "zi08"))
    assert hrrr.hash == s2.hash != zi.hash

    _write_trajectory(model, _sid(point_receptor))
    assert model.simulations[_sid(point_receptor, "s2")].has_trajectory
    assert not model.simulations[_sid(point_receptor, "zi08")].has_trajectory
    assert len(model.output.runs()) == 1


def test_unreferenced_lists_output_folders_the_config_no_longer_uses(
    tmp_path, point_receptor
):
    first = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}),
        receptors=[point_receptor],
    )
    for variant in ("hrrr", "zi08"):
        _write_trajectory(first, _sid(point_receptor, variant))
        _write_footprint(first, _sid(point_receptor, variant))
    assert first.unreferenced() == {"particles": [], "footprints": []}

    # zi08 dropped, and hrrr's smoothing changed: its old footprint folder is stale.
    later = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {"smooth_factor": 0.5}}),
        receptors=[point_receptor],
    )
    stale = later.unreferenced()
    assert [k.split("-")[0] for k in stale["particles"]] == ["zi08"]
    assert sorted(k.rsplit("-", 1)[0] for k in stale["footprints"]) == ["hrrr", "zi08"]
    # Nothing was deleted.
    assert first.simulations[_sid(point_receptor, "zi08")].is_complete()


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_writes_config_and_receptors_to_project(tmp_path, point_receptor):
    project_dir = tmp_path / "project"
    model = Model(
        project=project_dir, config=_config(tmp_path), receptors=[point_receptor]
    )

    ids = register(model)

    assert ids == [_rid(point_receptor)]
    assert (project_dir / CONFIG_KEY).exists()
    assert (project_dir / RECEPTORS_KEY).exists()
    assert not (project_dir / "simulations").exists()

    clone = Model(project=project_dir)
    assert len(clone.receptors) == 1
    assert clone.receptors[0].id == point_receptor.id
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_register_writes_the_receptors_of_a_csv(tmp_path):
    """A receptors CSV is read, then written out with its extra columns kept."""
    csv = tmp_path / "inputs" / "my_receptors.csv"
    csv.parent.mkdir()
    csv.write_text(
        "time,lati,long,zagl,site\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,wbb\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0,wbb\n"
    )
    project_dir = tmp_path / "project"

    model = Model(project=project_dir, config=_config(tmp_path), receptors=csv)

    assert len(register(model)) == 2
    clone = Model(project=project_dir)
    assert [r.id for r in clone.receptors] == [r.id for r in model.receptors]
    assert [r.attrs["site"] for r in clone.receptors] == ["wbb", "wbb"]


def test_register_preserves_project_receptors_csv(tmp_path):
    """Registering the project's own receptors.csv must not rewrite the file."""
    _config(tmp_path).to_yaml(tmp_path / CONFIG_KEY)
    original = (
        "time,lati,long,zagl\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0\n"
    )
    csv = tmp_path / RECEPTORS_KEY
    csv.write_text(original)

    assert len(register(Model(project=tmp_path))) == 2
    assert csv.read_text() == original


def test_register_appends_in_memory_receptors_to_existing_csv(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    register(Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a]))

    ids = register(Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_b]))

    assert ids == [_rid(rec_b)]
    assert [r.id for r in Model(project=tmp_path).receptors] == [rec_a.id, rec_b.id]


def test_register_explicit_batch_merges_with_existing_receptors(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a])

    assert register(model) == [_rid(rec_a)]
    assert register(model, receptors=[rec_b]) == [_rid(rec_b)]

    # The model is a view of what it was given; the project holds both.
    assert model.simulations.keys() == [_sid(rec_a)]
    project = Model(project=tmp_path)
    assert project.simulations.keys() == [_sid(rec_a), _sid(rec_b)]
    assert project.receptors[_rid(rec_b)] == rec_b


def test_register_explicit_batch_dedupes_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    register(model)

    assert register(model, receptors=[point_receptor]) == [_rid(point_receptor)]
    assert len(model.receptors) == 1
    assert len(Model(project=tmp_path).receptors) == 1


def test_model_register_rereads_the_project_file_it_views(tmp_path):
    """A model that reads receptors.csv sees receptors registered through it."""
    rec_a, rec_b = _receptor(12), _receptor(13)
    register(Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a]))
    model = Model(project=tmp_path)
    assert model.simulations.keys() == [_sid(rec_a)]  # read and cached

    assert model.register(receptors=[rec_b]) == [_rid(rec_b)]

    assert [r.id for r in model.receptors] == [rec_a.id, rec_b.id]
    assert model.simulations.keys() == [_sid(rec_a), _sid(rec_b)]


def test_register_leaves_the_model_as_it_was(tmp_path, point_receptor):
    """Registering writes to the project; the model it was called on does not change."""
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    [rid] = register(model, receptors=[point_receptor])

    assert len(model.receptors) == 0
    reopened = Model(project=tmp_path)
    assert reopened.receptors[rid] == point_receptor
    assert reopened.simulations[(rid, "hrrr")].receptor == point_receptor


def test_register_never_rewrites_an_existing_config_yaml(tmp_path, point_receptor):
    path = tmp_path / CONFIG_KEY
    _config(tmp_path).to_yaml(path)
    text = path.read_text() + "# a comment the user added\n"
    path.write_text(text)

    register(Model(project=tmp_path, receptors=[point_receptor]))

    assert path.read_text() == text


def test_registering_a_python_config_again_leaves_config_yaml(tmp_path, point_receptor):
    """Writing fills in the variants, so the file never equals the config it came from."""
    path = tmp_path / CONFIG_KEY
    register(
        Model(project=tmp_path, config=_config(tmp_path), receptors=[point_receptor])
    )
    text = path.read_text()

    register(
        Model(project=tmp_path, config=_config(tmp_path), receptors=[point_receptor])
    )

    assert path.read_text() == text


def test_a_config_that_differs_from_config_yaml_is_refused(tmp_path, point_receptor):
    path = tmp_path / CONFIG_KEY
    register(
        Model(project=tmp_path, config=_config(tmp_path), receptors=[point_receptor])
    )
    text = path.read_text()
    changed = Model(
        project=tmp_path,
        config=_config(tmp_path, ziscale=0.8),
        receptors=[point_receptor],
    )

    with pytest.raises(ConfigValidationError, match="other settings"):
        register(changed)
    assert path.read_text() == text


def test_an_unreadable_config_yaml_is_not_replaced(tmp_path, point_receptor):
    import yaml

    path = tmp_path / CONFIG_KEY
    path.write_text("mets: [unclosed\n")
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    with pytest.raises(yaml.YAMLError):
        register(model)
    assert path.read_text() == "mets: [unclosed\n"


def test_changed_settings_make_a_new_run_instead_of_an_error(tmp_path, point_receptor):
    """Editing a setting is not refused; the next run goes to a new folder (#67)."""
    first = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(first, _sid(point_receptor))

    changed = Model(
        project=tmp_path,
        config=_config(tmp_path, ziscale=0.8),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    exc.run(changed)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]
    assert not changed.simulations[_sid(point_receptor)].has_trajectory
    assert first.simulations[_sid(point_receptor)].has_trajectory  # untouched
    assert [k.split("-")[0] for k in changed.unreferenced()["particles"]] == ["hrrr"]


# ---------------------------------------------------------------------------
# simulations: collection protocol and selection
# ---------------------------------------------------------------------------


def test_simulations_mapping_protocol(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    sims = model.simulations

    assert isinstance(sims, SimulationCollection)
    assert sims.keys() == [sid_a, sid_b]
    assert len(sims) == 2
    assert sid_a in sims
    assert str(sid_a) in sims
    assert (str(rec_a.id), "hrrr") in sims
    assert _sid(_receptor(14)) not in sims
    assert _sid(rec_a, "nam") not in sims
    assert "not a sim id" not in sims
    assert 42 not in sims

    sim = sims[sid_a]
    assert isinstance(sim, Simulation)
    assert sims[str(sid_a)] == sim  # equal inputs, equal simulation
    assert not model.output.path.exists()  # building one has no side effects
    assert [s.id for s in sims] == sims.keys()
    with pytest.raises(KeyError):
        sims[_sid(rec_a, "nam")]


def test_simulations_span_receptors_times_variants(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    config = _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}})
    model = Model(project=tmp_path, config=config, receptors=[rec_a, rec_b])

    assert model.simulations.keys() == [
        _sid(rec_a),
        _sid(rec_a, "zi08"),
        _sid(rec_b),
        _sid(rec_b, "zi08"),
    ]
    assert model.simulations.receptors == [_rid(rec_a), _rid(rec_b)]
    assert model.simulations.variants == ["hrrr", "zi08"]


def test_simulations_empty_when_no_receptors(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert model.simulations.keys() == []
    assert len(model.simulations.incomplete()) == 0
    assert model.simulations.status().empty


def test_sel_by_variant_and_realization_group(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 3}},
    )
    sims = Model(
        project=tmp_path, config=config, receptors=[point_receptor]
    ).simulations

    assert sims.sel(variant="err").variants == ["err-0", "err-1", "err-2"]
    assert sims.sel(variant="err-1").variants == ["err-1"]
    assert sims.sel(variant=["hrrr", "err-2"]).variants == ["hrrr", "err-2"]
    with pytest.raises(KeyError):
        sims.sel(variant="nope")


def test_sel_by_receptor_time_location_and_predicate(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    sims = model.simulations

    assert sims.sel(receptor=_rid(rec_b)).keys() == [_sid(rec_b)]
    assert sims.sel(time=slice("2023-01-01 12:30", None)).keys() == [_sid(rec_b)]
    assert sims.sel(time=("2023-01-01 11:00", "2023-01-01 12:00")).keys() == [
        _sid(rec_a)
    ]
    assert sims.sel(time="2023-01-01 13:00").keys() == [_sid(rec_b)]
    assert sims.sel(location=rec_b.location_id).keys() == [_sid(rec_b)]
    assert sims.sel(where=lambda r: r.longitude < -111.87).keys() == [_sid(rec_b)]
    # Selections compose.
    assert sims.sel(location=rec_a.location_id).sel(receptor=_rid(rec_b)).keys() == []


def test_receptor_collection_sel(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])

    assert [
        r.id for r in model.receptors.sel(time=slice(None, "2023-01-01 12:00"))
    ] == [rec_a.id]
    assert [r.id for r in model.receptors.sel(location=[rec_b.location_id])] == [
        rec_b.id
    ]


def test_sel_where_uses_receptor_attrs(tmp_path):
    (tmp_path / RECEPTORS_KEY).write_text(
        "time,longitude,latitude,altitude,scene\n"
        "2023-01-01 12:00:00,-111.85,40.77,5.0,A\n"
        "2023-01-01 13:00:00,-111.85,40.77,5.0,B\n"
        "2023-01-01 14:00:00,-111.85,40.77,5.0,A\n"
    )
    model = Model(project=tmp_path, config=_config(tmp_path))

    scene_a = model.simulations.sel(where=lambda r: r.attrs["scene"] == "A")
    assert len(scene_a) == 2
    assert [str(s.id.receptor)[:12] for s in scene_a] == [
        "202301011200",
        "202301011400",
    ]


def test_sel_by_label_and_the_receptor_table(tmp_path):
    (tmp_path / RECEPTORS_KEY).write_text(
        "time,longitude,latitude,altitude,scene\n"
        "2023-01-01 12:00:00,-111.85,40.77,5.0,A\n"
        "2023-01-01 13:00:00,-111.85,40.77,5.0,B\n"
        "2023-01-01 14:00:00,-111.85,40.77,5.0,C\n"
    )
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert [r.attrs["scene"] for r in model.receptors.sel(scene="B")] == ["B"]
    assert len(model.receptors.sel(scene=["A", "C"])) == 2
    assert len(model.receptors.sel(scene="Z")) == 0
    assert len(model.simulations.sel(scene="A")) == 1
    assert len(model.simulations.sel(scene=["A", "C"], time="2023-01-01 14:00")) == 1

    frame = model.receptors.to_frame()
    assert list(frame.columns)[-1] == "scene"
    assert frame["scene"].tolist() == ["A", "B", "C"]
    assert len(frame) == 3


def test_receptors_that_would_share_result_files_are_refused(tmp_path):
    """Two receptors with one id would overwrite each other's results."""
    from stilt.receptors import MultiPointReceptor

    def slant(alt):
        return MultiPointReceptor(
            time="2023-01-01 12:00",
            longitudes=[-111.85, -111.86],
            latitudes=[40.77, 40.78],
            altitudes=[alt, 500.0],
        )

    a, b = slant(10.001), slant(10.004)
    assert a.id == b.id and a != b
    with pytest.raises(ValueError, match="share the id"):
        _ = Model(
            project=tmp_path, config=_config(tmp_path), receptors=[a, b]
        ).receptors

    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[a])
    register(model)
    with pytest.raises(ValueError, match="share the id"):
        register(model, [b])
    assert register(model, [slant(10.001)]) == [a.id]  # the same receptor again is fine


def test_sel_raises_for_unknown_receptor_or_variant(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "err": {"realizations": 2}}),
        receptors=[point_receptor],
    )
    assert model.simulations.sel(variant="err").variants == ["err-0", "err-1"]
    assert (
        model.simulations.sel(time="2020-01-01").keys() == []
    )  # a filter may be empty
    with pytest.raises(KeyError, match="hrr"):
        model.simulations.sel(variant="hrr")
    with pytest.raises(KeyError, match="nope"):
        model.simulations.sel(receptor="nope")


# ---------------------------------------------------------------------------
# completion, status, outputs over a selection
# ---------------------------------------------------------------------------


def test_incomplete_follows_each_variant_outputs(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        variants={"hrrr": {}, "traj": {"grid": None}, "s2": {"smooth_factor": 2}},
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    sims = model.simulations

    assert sims.incomplete().variants == ["hrrr", "traj", "s2"]

    _write_trajectory(model, _sid(point_receptor))  # shared by hrrr, traj, and s2
    assert sims.incomplete().variants == ["hrrr", "s2"]

    _write_footprint(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor, "s2"), empty=True)
    assert sims.incomplete().keys() == []
    assert isinstance(sims.incomplete(), SimulationCollection)


def test_status_frame_marks_outputs_a_variant_does_not_produce(
    tmp_path, point_receptor
):
    config = _config(tmp_path, variants={"hrrr": {}, "traj": {"grid": None}})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    _write_trajectory(model, _sid(point_receptor))

    status = model.simulations.status().set_index("variant")

    assert list(status.columns) == [
        "receptor",
        TRAJECTORY,
        FOOTPRINT,
        "empty",
        "complete",
    ]
    assert status.loc["hrrr", TRAJECTORY] == True  # noqa: E712
    assert status.loc["hrrr", FOOTPRINT] == False  # noqa: E712
    assert status.loc["hrrr", "empty"] == False  # noqa: E712
    assert pd.isna(status.loc["traj", FOOTPRINT])
    assert pd.isna(status.loc["traj", "empty"])
    assert status.loc["traj", "complete"] == True  # noqa: E712
    assert status.loc["hrrr", "complete"] == False  # noqa: E712


def test_status_is_the_simulation_table(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    assert list(model.status().columns) == [
        "receptor",
        "variant",
        "trajectory",
        "footprint",
        "empty",
        "complete",
    ]
    assert model.status().empty

    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    assert model.status()["complete"].tolist() == [False, False]

    _write_trajectory(model, _sid(rec_a))
    _write_footprint(model, _sid(rec_a))
    _write_trajectory(model, _sid(rec_b))
    _write_footprint(model, _sid(rec_b), empty=True)
    assert model.status()["complete"].tolist() == [True, True]
    assert model.status()["empty"].tolist() == [False, True]


def test_trajectories_paths_load_and_missing(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    trajectories = model.simulations.trajectories

    assert isinstance(trajectories, OutputCollection)
    assert len(trajectories) == 2
    assert trajectories.paths() == {}

    path = _write_trajectory(model, _sid(rec_a))

    assert path.parent == model.output.runs()[0].particles_dir / "date=2023-01-01"
    assert trajectories.paths() == {_sid(rec_a): path}
    [loaded] = trajectories.load().values()
    assert isinstance(loaded, Trajectories)
    assert loaded.receptor.id == rec_a.id
    assert model.simulations.sel(time="2023-01-01 13:00").trajectories.paths() == {}


def test_footprint_paths_and_load_skip_empty_footprints(tmp_path):
    rec_done, rec_empty, rec_missing = _receptor(12), _receptor(13), _receptor(14)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_done, rec_empty, rec_missing],
    )
    foot_path = _write_footprint(model, _sid(rec_done))
    empty_path = _write_footprint(model, _sid(rec_empty), empty=True)

    assert model.simulations.footprint.paths() == {
        _sid(rec_done): foot_path,
        _sid(rec_empty): empty_path,
    }
    loaded = model.simulations.footprint.load()
    assert list(loaded) == [_sid(rec_done)]
    assert isinstance(loaded[_sid(rec_done)], Footprint)


def test_footprint_excludes_trajectory_only_variants(tmp_path, point_receptor):
    config = _config(tmp_path, variants={"hrrr": {}, "traj": {"grid": None}})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    assert len(model.simulations.footprint) == 1
    assert len(model.simulations.sel(variant="traj").footprint) == 0
    assert len(model.simulations.trajectories) == 2


def test_footprint_load_by_variant(tmp_path, point_receptor):
    config = _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    for variant in ("hrrr", "zi08"):
        _write_footprint(model, _sid(point_receptor, variant))

    assert len(model.simulations.footprint.load()) == 2
    loaded = model.simulations.sel(variant="zi08").footprint.load()
    assert list(loaded) == [_sid(point_receptor, "zi08")]
    [foot] = loaded.values()
    assert isinstance(foot, Footprint)
    assert foot.name == "zi08"


def test_jacobian_over_a_selection(tmp_path):
    rec_a, rec_b, rec_c = _receptor(12), _receptor(13), _receptor(14)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}),
        receptors=[rec_a, rec_b, rec_c],
    )
    _write_footprint(model, _sid(rec_a))
    _write_footprint(model, _sid(rec_b), empty=True)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="12h"), closed="left"
    )

    H = model.simulations.sel(variant="hrrr").jacobian(target, bins)

    assert list(H.receptors) == [_rid(rec_a)]
    assert H.empty == [_rid(rec_b)]
    assert H.missing == [_rid(rec_c)]
    assert H.data.shape == (1, len(bins) * len(target.index))
    expected = model.simulations[_sid(rec_a)].footprint.aggregate(target, bins)
    np.testing.assert_allclose(
        H.to_frame().iloc[0].to_numpy().reshape(len(bins), -1).T,
        expected.to_numpy(),
        rtol=1e-6,
    )
    right_closed = pd.IntervalIndex.from_breaks(
        bins.left.append(bins.right[-1:]), closed="right"
    )
    with pytest.raises(ValueError, match="closed on the left"):
        model.simulations.sel(variant="hrrr").jacobian(target, right_closed)
    with pytest.raises(ValueError, match="one variant"):
        model.simulations.jacobian(target, bins)
    with pytest.raises(ValueError, match="no footprints yet"):
        model.simulations.sel(variant="zi08").jacobian(target, bins)


# ---------------------------------------------------------------------------
# accessors and plotting
# ---------------------------------------------------------------------------


def test_accessors_are_cached(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert model.receptors is model.receptors
    assert model.plot is model.plot
    assert model.variants is model.variants
    assert model.simulations is model.simulations
    assert model.output is model.output


def test_plot_availability_returns_axes(tmp_path, point_receptor):
    from matplotlib.axes import Axes

    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    ax = model.plot.availability()

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Simulation Availability"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def _run_model(tmp_path, point_receptor, executor, skip_existing=True, wait=True):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    return model, executor.run(model, skip_existing=skip_existing, wait=wait)


def test_run_dispatches_receptors_with_project_and_compute_root(
    tmp_path, point_receptor
):
    exc = _CapturingExecutor()

    model, handle = _run_model(tmp_path, point_receptor, exc)

    assert handle is exc.handle
    call = exc.start_calls[0]
    assert call["pending"] == [_rid(point_receptor)]
    assert call["project"] == str(tmp_path) == model.project.root
    assert call["compute_root"] is None  # left to the worker unless given
    assert call["skip_existing"] is True
    assert call["execution"] == model.config.execution


def test_run_forwards_explicit_compute_root(tmp_path, point_receptor):
    project_dir = tmp_path / "project"
    compute_root = tmp_path / "scratch"
    model = Model(
        project=project_dir, config=_config(tmp_path), receptors=[point_receptor]
    )
    exc = _CapturingExecutor()

    exc.run(model, skip_existing=False, compute_root=compute_root)

    assert exc.start_calls[0]["project"] == str(project_dir)
    assert exc.start_calls[0]["compute_root"] == compute_root.resolve()


def test_run_propagates_skip_existing_false(tmp_path, point_receptor):
    exc = _CapturingExecutor()

    _run_model(tmp_path, point_receptor, exc, skip_existing=False)

    assert exc.start_calls[0]["skip_existing"] is False


def test_run_uses_the_execution_settings_of_the_config(tmp_path, point_receptor):
    config = _config(tmp_path, execution={"backend": "local", "n_workers": 3})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    exc = _CapturingExecutor()

    handle = exc.run(model, skip_existing=False, wait=False)

    assert handle is exc.handle
    assert exc.start_calls[0]["execution"].n_workers == 3


def test_run_takes_execution_settings_in_place_of_the_configs(tmp_path, point_receptor):
    from stilt.config import ExecutionConfig

    config = _config(tmp_path, execution={"n_workers": 3})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    exc = _CapturingExecutor()

    exc.run(model, execution=ExecutionConfig(n_workers=8))

    assert exc.start_calls[0]["execution"].n_workers == 8


def test_run_registers_inputs_before_start(tmp_path, point_receptor):
    class _CheckingExecutor(_CapturingExecutor):
        def dispatch(self, model, pending, execution, **kwargs):
            root = model.project.directory
            self.seen = ((root / CONFIG_KEY).exists(), (root / RECEPTORS_KEY).exists())
            return super().dispatch(model, pending, execution, **kwargs)

    exc = _CheckingExecutor()

    _run_model(tmp_path, point_receptor, exc)

    assert exc.seen == (True, True)
    assert len(Model(project=tmp_path).receptors) == 1


def test_run_skip_existing_omits_complete_receptors(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor), empty=True)
    exc = _CapturingExecutor()

    handle = exc.run(model, skip_existing=True)

    assert not exc.was_started
    assert isinstance(handle, LocalHandle)


def test_run_skip_existing_redispatches_missing_footprint(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    exc.run(model, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]


def test_run_after_adding_a_variant_redispatches_the_receptor(tmp_path, point_receptor):
    """A finished project grows a variant: only that variant is incomplete."""
    first = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    register(first)
    _write_trajectory(first, _sid(point_receptor))
    _write_footprint(first, _sid(point_receptor))

    # The user adds a variant to config.yaml.
    _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}).to_yaml(
        tmp_path / CONFIG_KEY
    )
    grown = Model(project=tmp_path)
    assert grown.simulations.incomplete().keys() == [_sid(point_receptor, "zi08")]
    exc = _CapturingExecutor()

    exc.run(grown, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]


def test_run_skip_existing_redispatches_a_missing_realization(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        grid=None,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    _write_trajectory(model, _sid(point_receptor))
    _write_trajectory(model, _sid(point_receptor, "err-0"))
    exc = _CapturingExecutor()

    exc.run(model, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]
    assert model.simulations.incomplete().keys() == [_sid(point_receptor, "err-1")]

    _write_trajectory(model, _sid(point_receptor, "err-1"))
    again = _CapturingExecutor()
    again.run(model, skip_existing=True)
    assert not again.was_started


def test_run_skip_existing_dispatches_only_incomplete_receptors(tmp_path):
    rec_done, rec_todo = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_done, rec_todo],
    )
    _write_trajectory(model, _sid(rec_done))
    exc = _CapturingExecutor()

    exc.run(model, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(rec_todo)]


def test_run_no_skip_dispatches_everything(tmp_path):
    rec_done, rec_todo = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_done, rec_todo],
    )
    _write_trajectory(model, _sid(rec_done))
    exc = _CapturingExecutor()

    exc.run(model, skip_existing=False)

    assert exc.start_calls[0]["pending"] == [_rid(rec_done), _rid(rec_todo)]
    assert exc.start_calls[0]["skip_existing"] is False


def test_run_returns_completed_handle_without_receptors(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    exc = _CapturingExecutor()

    handle = exc.run(model, skip_existing=False)

    assert not exc.was_started
    assert isinstance(handle, LocalHandle)


def test_run_wait_calls_handle_wait(tmp_path, point_receptor):
    handle = _CountingHandle()
    exc = _CapturingExecutor(handle=handle)

    _, returned = _run_model(tmp_path, point_receptor, exc, wait=True)

    assert returned is handle
    assert handle.wait_calls == 1


def test_run_no_wait_returns_handle_without_waiting(tmp_path, point_receptor):
    handle = _CountingHandle()
    exc = _CapturingExecutor(handle=handle)

    _, returned = _run_model(tmp_path, point_receptor, exc, wait=False)

    assert returned is handle
    assert handle.wait_calls == 0


# ---------------------------------------------------------------------------
# Completion by listing agrees with completion by file
# ---------------------------------------------------------------------------


def _mixed_state_model(tmp_path):
    """Six receptors under three variants, in every state a simulation can be in."""
    receptors = [_receptor(h) for h in range(10, 16)]
    config = _config(
        tmp_path,
        variants={
            "hrrr": {},
            "smooth": {"smooth_factor": 2},  # shares hrrr's particles
            "zi08": {"ziscale": 0.8, "grid": None},  # particles only
        },
    )
    model = Model(project=tmp_path, config=config, receptors=receptors)
    a, b, c, d, e, _ = receptors
    _write_trajectory(model, _sid(a))  # particles, no footprint
    _write_trajectory(model, _sid(b))
    _write_footprint(model, _sid(b))  # complete for hrrr, not for smooth
    _write_trajectory(model, _sid(c))
    _write_footprint(model, _sid(c), empty=True)  # an empty footprint counts
    _write_footprint(model, _sid(c, "smooth"))
    _write_trajectory(model, _sid(d, "zi08"))  # complete: zi08 makes no footprint
    _write_trajectory(model, _sid(e))
    _write_footprint(model, _sid(e))
    _write_footprint(model, _sid(e, "smooth"))
    _write_trajectory(model, _sid(e, "zi08"))  # complete under every variant
    return model


@pytest.mark.parametrize("list_from", [0, 10_000], ids=["listing", "file-by-file"])
def test_incomplete_and_status_agree_with_is_complete(tmp_path, monkeypatch, list_from):
    """Both ways of checking must give `Simulation.is_complete()`'s answer."""
    import stilt.collections as collections

    monkeypatch.setattr(collections, "_LIST_FROM", list_from)
    model = _mixed_state_model(tmp_path)
    sims = model.simulations

    expected = [sim.id for sim in sims if not sim.is_complete()]
    assert 0 < len(expected) < len(sims)
    assert sims.incomplete().keys() == expected

    status = sims.status()
    assert status["complete"].tolist() == [sim.is_complete() for sim in sims]
    assert status["trajectory"].tolist() == [sim.has_trajectory for sim in sims]
    for row, sim in zip(status.itertuples(), sims, strict=True):
        if sim.makes_footprint:
            assert row.footprint == sim.has_footprint
            assert row.empty == (sim.empty_reason is not None)
        else:
            assert pd.isna(row.footprint) and pd.isna(row.empty)

    one_variant = sims.sel(variant="smooth")
    assert one_variant.incomplete().keys() == [
        sim.id for sim in one_variant if not sim.is_complete()
    ]


def test_incomplete_of_a_project_with_no_results_is_everything(tmp_path, monkeypatch):
    import stilt.collections as collections

    monkeypatch.setattr(collections, "_LIST_FROM", 0)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[_receptor(12)])
    assert model.simulations.incomplete().keys() == model.simulations.keys()
    assert not model.output.path.exists()  # looking creates nothing
