"""Tests for stilt.store (byte stores addressed by output key)."""

import shutil
from pathlib import Path

import pytest

from stilt.store import FsspecStore, LocalStore, Store, is_uri, make_store, uri_join

# ---------------------------------------------------------------------------
# Protocol / factory
# ---------------------------------------------------------------------------


def test_local_store_is_runtime_checkable(tmp_path):
    store = LocalStore(tmp_path / "output")
    assert isinstance(store, Store)


def test_remote_store_is_runtime_checkable(tmp_path):
    store = FsspecStore(f"memory://proto-check-{tmp_path.name}")
    assert isinstance(store, Store)


def test_make_store_returns_local_store_for_local_path(tmp_path):
    store = make_store(tmp_path / "output")
    assert isinstance(store, LocalStore)
    assert store.root == (tmp_path / "output").resolve()


def test_make_store_returns_fsspec_store_for_remote_uri(tmp_path):
    store = make_store(f"memory://make-store-{tmp_path.name}")
    assert isinstance(store, FsspecStore)


# ---------------------------------------------------------------------------
# is_uri / uri_join
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "root",
    ["s3://bucket/project", "gs://bucket", "memory://proj", "file://x"],
)
def test_is_uri_true_for_scheme_roots(root):
    assert is_uri(root)


@pytest.mark.parametrize("root", ["/abs/path", "relative/path", "project"])
def test_is_uri_false_for_local_paths(root):
    assert not is_uri(root)
    assert not is_uri(Path(root))


def test_uri_join_on_uri_root():
    assert uri_join("s3://bucket/project", "simulations", "by-id") == (
        "s3://bucket/project/simulations/by-id"
    )


def test_uri_join_strips_slashes_and_skips_empty_parts():
    assert uri_join("s3://bucket/project/", "/simulations/", "", "by-id/") == (
        "s3://bucket/project/simulations/by-id"
    )


def test_uri_join_with_no_parts_returns_root_without_trailing_slash():
    assert uri_join("s3://bucket/project/") == "s3://bucket/project"


def test_uri_join_on_local_root(tmp_path):
    joined = uri_join(str(tmp_path), "simulations", "by-id")
    assert joined == str(tmp_path / "simulations" / "by-id")


# ---------------------------------------------------------------------------
# Contract tests — run against both LocalStore and FsspecStore(memory://)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["local", "memory"])
def any_store(request, tmp_path):
    if request.param == "local":
        return make_store(tmp_path / "output")
    return make_store(
        f"memory://contract-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )


def test_store_contract_read_write_round_trip(any_store):
    any_store.write_bytes("nested/file.txt", b"hello")
    assert any_store.read_bytes("nested/file.txt") == b"hello"
    assert any_store.exists("nested/file.txt")
    local = any_store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello"


def test_store_contract_nonexistent_key(any_store):
    assert not any_store.exists("does/not/exist.txt")


def test_store_contract_publish_missing_source_is_noop(any_store, tmp_path):
    any_store.publish_file(tmp_path / "nope.bin", "data/out.bin")
    assert not any_store.exists("data/out.bin")


# ---------------------------------------------------------------------------
# LocalStore-specific tests
# ---------------------------------------------------------------------------


def test_local_store_file_round_trip(tmp_path):
    store = LocalStore(tmp_path / "output")
    store.write_bytes("data/test.bin", b"hello")

    assert store.read_bytes("data/test.bin") == b"hello"
    assert store.exists("data/test.bin")
    assert store.path("data/test.bin") == tmp_path / "output" / "data" / "test.bin"
    assert store.local_path("data/test.bin") == store.path("data/test.bin")


def test_local_store_path_strips_leading_slash(tmp_path):
    store = LocalStore(tmp_path / "output")
    assert store.path("/data/test.bin") == tmp_path / "output" / "data" / "test.bin"


def test_publish_file_is_atomic_via_tmp_then_replace(tmp_path, monkeypatch):
    """A failure mid-copy must leave the canonical key untouched."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"final-payload")
    store = LocalStore(tmp_path / "output")
    target = tmp_path / "output" / "data" / "out.bin"

    # Pre-populate the canonical key so we can verify it isn't clobbered on failure.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old-payload")

    real_copy = shutil.copy2

    def _failing_copy(s, dst):
        # Simulate a writer that crashes after creating the .tmp.
        real_copy(s, dst)
        raise RuntimeError("boom")

    monkeypatch.setattr("stilt.store.shutil.copy2", _failing_copy)

    with pytest.raises(RuntimeError, match="boom"):
        store.publish_file(src, "data/out.bin")

    # Canonical key is unchanged; tmp sibling cleaned up.
    assert target.read_bytes() == b"old-payload"
    assert not (target.parent / "out.bin.tmp").exists()


def test_publish_file_replaces_existing_local_target(tmp_path):
    src_v1 = tmp_path / "src1.bin"
    src_v2 = tmp_path / "src2.bin"
    src_v1.write_bytes(b"v1")
    src_v2.write_bytes(b"v2")
    store = LocalStore(tmp_path / "output")

    store.publish_file(src_v1, "data/out.bin")
    store.publish_file(src_v2, "data/out.bin")

    target = tmp_path / "output" / "data" / "out.bin"
    assert target.read_bytes() == b"v2"
    assert not (target.parent / "out.bin.tmp").exists()


def test_publish_file_noop_when_source_is_target(tmp_path):
    """Publishing a file onto its own key must not copy, truncate, or leave a tmp."""
    store = LocalStore(tmp_path / "output")
    store.write_bytes("data/out.bin", b"same")
    target = store.path("data/out.bin")
    mtime = target.stat().st_mtime_ns

    store.publish_file(target, "data/out.bin")

    assert target.read_bytes() == b"same"
    assert target.stat().st_mtime_ns == mtime
    assert not (target.parent / "out.bin.tmp").exists()


# ---------------------------------------------------------------------------
# FsspecStore (remote / memory://) tests
# ---------------------------------------------------------------------------


def test_remote_store_memory_round_trip(tmp_path):
    store = FsspecStore(
        f"memory://artifact-store-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )
    store.write_bytes("nested/file.txt", b"hello memory")

    assert store.read_bytes("nested/file.txt") == b"hello memory"
    assert store.exists("nested/file.txt")

    local = store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello memory"
    assert local.is_file()


def test_remote_store_strips_trailing_slash_from_root(tmp_path):
    store = FsspecStore(f"memory://trailing-{tmp_path.name}/")
    assert not store.root.endswith("/")


def test_remote_store_publish_file_and_local_path(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"published")
    store = FsspecStore(
        f"memory://publish-tests-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )

    store.publish_file(src, "simulations/by-id/sim-1/out.bin")

    key = "simulations/by-id/sim-1/out.bin"
    assert store.exists(key)
    assert store.read_bytes(key) == b"published"
    local = store.local_path(key)
    assert local.is_file()
    assert local.read_bytes() == b"published"
    assert tmp_path / "cache" in local.parents
