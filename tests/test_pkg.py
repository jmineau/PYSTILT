"""Test basic functionality of stilt."""

import stilt


def test_version():
    """Test that version is defined."""
    assert hasattr(stilt, "__version__")
    assert isinstance(stilt.__version__, str)


def test_the_top_level_is_what_a_user_calls():
    """Every name in stilt.__all__ imports, and the plumbing stays in its module (#150)."""
    for name in stilt.__all__:
        assert hasattr(stilt, name), name
    for name in ("Output", "Met", "Variant", "SimID", "Geometry", "write_particles"):
        assert not hasattr(stilt, name), name


def test_the_old_names_are_gone():
    """Model and its collections were replaced by Project (#99); no aliases in an alpha."""
    for name in ("Model", "ModelConfig"):
        assert not hasattr(stilt, name), name


def test_results_are_plain_data():
    """Particles and footprints are a DataFrame and a DataArray, not wrapper classes (#107)."""
    for name in ("Trajectories", "Footprint"):
        assert not hasattr(stilt, name), name
