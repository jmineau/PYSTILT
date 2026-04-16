"""Integration-style tests for the observation-layer bridge into Model."""

from stilt.config import ModelConfig
from stilt.model import Model
from stilt.observations import PointSensor


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


def test_point_sensor_scene_batch_id_bridges_into_model_submit(tmp_path):
    sensor = PointSensor(name="tower", supported_species=("co2",))
    observations = [
        sensor.make_observation(
            time="2023-01-01 12:00:00",
            latitude=40.77,
            longitude=-111.85,
            altitude=30.0,
            observation_id="tower-001",
        ),
        sensor.make_observation(
            time="2023-01-01 12:05:00",
            latitude=40.78,
            longitude=-111.84,
            altitude=30.0,
            observation_id="tower-002",
        ),
    ]

    [scene] = sensor.group_scenes(observations)
    receptors = [sensor.build_receptor(obs) for obs in scene.observations]

    model = Model(project=tmp_path, config=_minimal_config(tmp_path))
    sim_ids = model.submit(receptors=receptors, batch_id=scene.batch_id)

    assert scene.batch_id == "tower-20230101120000"
    assert len(sim_ids) == 2
    assert set(model.get_simulation_ids()) == set(sim_ids)
    assert model.repository.batch_progress(scene.batch_id) == (0, 2)
