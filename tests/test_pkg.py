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
