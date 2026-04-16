"""Tests for the thin stilt.service facade."""

import datetime as dt

from stilt.config import ModelConfig
from stilt.model import Model
from stilt.receptor import Receptor
from stilt.service import BatchStatus, QueueStatus, Service
from stilt.simulation import SimID


def _write_minimal_config(tmp_path):
    cfg = ModelConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
    )
    cfg.to_yaml(tmp_path / "config.yaml")
    return cfg


def _make_model(tmp_path) -> tuple[Model, str]:
    cfg = _write_minimal_config(tmp_path)
    receptor = Receptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    model = Model(project=tmp_path, config=cfg, receptors=[receptor])
    sim_id = str(SimID.from_parts("hrrr", receptor))
    return model, sim_id


def test_service_status_reflects_queue_state(tmp_path):
    model, sim_id = _make_model(tmp_path)
    service = Service(model=model)
    service.submit(batch_id="batch-1")

    pending = service.status()
    assert pending == QueueStatus(
        project=model.project,
        total=1,
        completed=0,
        running=0,
        pending=1,
        failed=0,
    )

    claim = model.repository.claim_pending_claims(
        worker_id="svc-test",
        lease_ttl=600.0,
    )[0]
    running = service.status()
    assert running.running == 1
    assert running.pending == 0

    model.repository.release_claims([claim])
    model.repository.mark_trajectory_complete(sim_id)
    complete = service.status()
    assert complete.completed == 1
    assert complete.failed == 0


def test_service_batches_and_attempts(tmp_path):
    model, sim_id = _make_model(tmp_path)
    service = Service(model=model)
    service.submit(batch_id="batch-1")

    batch = service.batch_status("batch-1")
    assert batch == BatchStatus(batch_id="batch-1", completed=0, total=1)
    assert batch.percent_complete == 0.0
    assert batch.is_complete is False

    claim = model.repository.claim_pending_claims(worker_id="svc-test")[0]
    now = dt.datetime.now(dt.timezone.utc)
    assert service.active_claims(now=now) == [claim]
    assert service.active_claims(include_expired=True) == [claim]

    model.repository.release_claims([claim])
    assert service.active_claims() == []
    assert service.attempts(sim_id) == []

    model.repository.mark_trajectory_complete(sim_id)
    batch = service.batch_status("batch-1")
    assert batch.completed == 1
    assert batch.total == 1
    assert batch.is_complete is True


def test_service_drain_delegates_to_worker_loop(tmp_path, monkeypatch):
    model, _ = _make_model(tmp_path)
    service = Service(model=model)
    calls: list[dict[str, object]] = []

    def fake_worker_loop(
        model_arg,
        n_cores=1,
        follow=False,
        poll_interval=10.0,
        lease_ttl=1800.0,
    ):
        calls.append(
            {
                "model": model_arg,
                "n_cores": n_cores,
                "follow": follow,
                "poll_interval": poll_interval,
                "lease_ttl": lease_ttl,
            }
        )

    monkeypatch.setattr("stilt.service.project.worker_loop", fake_worker_loop)

    service.drain(cpus=3, poll_interval=2.5, lease_ttl=60.0)
    service.serve(cpus=2, poll_interval=1.0, lease_ttl=30.0)

    assert calls == [
        {
            "model": model,
            "n_cores": 3,
            "follow": False,
            "poll_interval": 2.5,
            "lease_ttl": 60.0,
        },
        {
            "model": model,
            "n_cores": 2,
            "follow": True,
            "poll_interval": 1.0,
            "lease_ttl": 30.0,
        },
    ]
