"""Test basic functionality of stilt."""

import stilt


def test_version():
    """Test that version is defined."""
    assert hasattr(stilt, "__version__")
    assert isinstance(stilt.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(stilt, "__author__")
    assert isinstance(stilt.__author__, str)


def test_email():
    """Test that email is defined."""
    assert hasattr(stilt, "__email__")
    assert isinstance(stilt.__email__, str)


def test_documented_top_level_symbols_are_importable():
    """The curated top-level API matches the core reference surface."""
    expected = [
        "Bounds",
        "Footprint",
        "FootprintConfig",
        "Grid",
        "LocationID",
        "MetArchive",
        "MetConfig",
        "MetStream",
        "Model",
        "ModelConfig",
        "Receptor",
        "ReceptorID",
        "RuntimeSettings",
        "SimID",
        "Simulation",
        "Trajectories",
        "read_receptors",
    ]
    for name in expected:
        assert hasattr(stilt, name), name
