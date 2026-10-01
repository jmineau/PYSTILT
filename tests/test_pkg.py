"""Test basic functionality of stilt."""

import stilt


def test_version():
    """Test that version is defined."""
    assert hasattr(stilt, "__version__")
    assert isinstance(stilt.__version__, str)


def test_documented_top_level_symbols_are_importable():
    """The curated top-level API matches the core reference surface."""
    expected = [
        "Receptor",
        "Bounds",
        "ColumnReceptor",
        "FootprintConfig",
        "Grid",
        "MetConfig",
        "Met",
        "Project",
        "Output",
        "ProjectConfig",
        "MultiPointReceptor",
        "PointReceptor",
        "RuntimeSettings",
        "SimID",
        "Simulation",
        "read_receptors",
        "read_particles",
        "particles_metadata",
        "write_particles",
        "read_footprint",
    ]
    for name in expected:
        assert hasattr(stilt, name), name


def test_the_old_names_are_gone():
    """Model and its collections were replaced by Project (#99); no aliases in an alpha."""
    for name in ("Model", "ModelConfig"):
        assert not hasattr(stilt, name), name


def test_results_are_plain_data():
    """Particles and footprints are a DataFrame and a DataArray, not wrapper classes (#107)."""
    for name in ("Trajectories", "Footprint"):
        assert not hasattr(stilt, name), name
