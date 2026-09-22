"""Integration-style tests for the observation-layer bridge into Model."""

from stilt.config import ModelConfig
from stilt.model import Model
from stilt.observations import Observation, build_point_receptor, group_by_overpass


def _minimal_config(tmp_path):
    return ModelConfig(
        n_hours=-24,
        numpar=100,
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
    )


def test_scene_receptors_register_into_model(tmp_path):
    observations = [
        Observation(
            sensor="tower",
            species="co2",
            time="2023-01-01 12:00:00",
            latitude=40.77,
            longitude=-111.85,
            altitude=30.0,
            observation_id="tower-001",
        ),
        Observation(
            sensor="tower",
            species="co2",
            time="2023-01-01 12:05:00",
            latitude=40.78,
            longitude=-111.84,
            altitude=30.0,
            observation_id="tower-002",
        ),
    ]

    [scene] = group_by_overpass(observations)
    receptors = scene.receptors(build_point_receptor)

    model = Model(project=tmp_path, config=_minimal_config(tmp_path))
    sim_ids = model.register(receptors=receptors)

    assert scene.id == "tower-202301011200"
    assert len(sim_ids) == 2
    assert set(model.simulations.ids()) == set(sim_ids)
    assert set(model.simulations.ids(time_range=scene.time_range)) == set(sim_ids)
