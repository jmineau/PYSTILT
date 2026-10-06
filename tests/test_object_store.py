"""
An output directory on an object store, against a local S3 server (moto).

The same Output code reads and writes a URL such as ``s3://bucket/output``:
listings go through fsspec, files are put directly, and pyarrow reads
through the store's filesystem.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("s3fs")
moto_server = pytest.importorskip("moto.server")

from stilt.footprint import open_footprints, read_footprint  # noqa: E402
from stilt.output import Output  # noqa: E402
from stilt.particles import read_particles  # noqa: E402

from .conftest import integration  # noqa: E402
from .test_output import (  # noqa: E402
    FEET,
    VARIANT,
    _footprint,
    _receptor,
    _trajectories,
    _write,
)

BUCKET = "pystilt-test"


@pytest.fixture(scope="module")
def s3_server():
    """A local S3 server for the module, with the credentials and endpoint boto reads."""
    import boto3
    import s3fs

    server = moto_server.ThreadedMotoServer(port=0, verbose=False)
    server.start()
    _, port = server.get_host_and_port()
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")
    saved = {k: os.environ.get(k) for k in (*names, "AWS_ENDPOINT_URL")}
    os.environ.update(dict.fromkeys(names[:2], "testing"))
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_ENDPOINT_URL"] = f"http://127.0.0.1:{port}"
    s3fs.S3FileSystem.clear_instance_cache()
    boto3.client("s3").create_bucket(Bucket=BUCKET)
    yield
    server.stop()
    s3fs.S3FileSystem.clear_instance_cache()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def store(s3_server) -> str:
    """A fresh folder on the S3 server."""
    return f"s3://{BUCKET}/{uuid.uuid4().hex}"


def test_output_on_an_object_store_writes_lists_and_reads(store):
    out = Output(f"{store}/output/")
    assert str(out.directory) == f"{store}/output"
    receptor = _receptor()
    rid = str(receptor.id)

    path = _write(out, VARIANT, receptor)
    assert str(path).startswith("s3://")
    assert out.present("particles", VARIANT) == {rid}
    assert out.hashes("particles") == {
        out.folder("particles", VARIANT).name[
            len("settings=") :
        ]: VARIANT.particles_hash  # type: ignore[union-attr]
    }
    particles = read_particles(path)
    assert len(particles) == len(_trajectories(receptor))
    assert out.table("particles", VARIANT).num_rows == len(particles)

    foot = _footprint(receptor)
    out.write_footprint(FEET, foot)
    assert out.present("footprints", FEET) == {rid}
    back = read_footprint(out.path("footprints", FEET, rid))  # type: ignore[arg-type]
    assert back is not None
    assert float(abs(back - foot).max()) < 1e-6
    ds = open_footprints([out.path("footprints", FEET, rid)])  # type: ignore[list-item]
    assert float(ds.foot.sum()) == pytest.approx(float(foot.sum()), rel=1e-6)

    later = _receptor(hour=18)
    out.write_empty_footprint(FEET, later, "outside_domain")
    assert out.present("footprints", FEET) == {rid, str(later.id)}
    assert read_footprint(out.path("footprints", FEET, str(later.id))) is None  # type: ignore[arg-type]


def test_logs_and_failure_records_on_an_object_store(store):
    out = Output(f"{store}/output")
    rid = str(_receptor().id)
    log = out.write_log(VARIANT, rid, "hycs_std ran\n")
    assert log.read_text() == "hycs_std ran\n"
    record = {"step": "particles", "reason": "MET_COVERAGE", "message": "boom"}
    out.record_failure("particles", VARIANT, rid, record)
    assert out.failure("particles", VARIANT, rid) == record
    out.clear_failure("particles", VARIANT, rid)
    assert out.failure("particles", VARIANT, rid) is None


def test_a_kept_workdir_on_an_object_store_leaves_out_links(store, tmp_path):
    out = Output(f"{store}/output")
    rid = str(_receptor().id)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "CONTROL").write_text("x")
    (workdir / "met").symlink_to(tmp_path)  # HYSPLIT links its met files

    kept = out.keep_workdir(VARIANT, rid, workdir)

    assert kept == out.kept_workdir(VARIANT, rid)
    assert (kept / "CONTROL").read_text() == "x"
    assert not (kept / "met").exists()


@integration
def test_a_project_runs_with_its_output_on_an_object_store(
    tmp_path, store, wbb_config, wbb_receptor
):
    """A run writes to S3, and status, the footprints, and the Jacobian read it back."""
    import pandas as pd

    from stilt.project import Project

    config = wbb_config.model_copy(update={"output": f"{store}/output"})
    project = Project.init(
        tmp_path / "s3_project", config=config, receptors=[wbb_receptor]
    )

    status = project.run()

    assert list(status["state"]) == ["complete"]
    assert set(project.status()["state"]) == {"complete"}
    sim = project.simulation(str(wbb_receptor.id), "hrrr")
    assert sim.log  # the run's log, read back from the store
    ds = project.footprints()
    assert ds.sizes["receptor"] == 1
    grid = config.grid
    assert grid is not None
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range(
            wbb_receptor.time - pd.Timedelta("1D"),
            wbb_receptor.time + pd.Timedelta("1h"),
            freq="1D",
        ),
        closed="left",
    )
    H = project.jacobian(project.simulations, grid, bins)
    assert H.data.sum() == pytest.approx(float(ds.foot.sum()), rel=1e-4)
