"""Tests for the jsf package initialization."""

import jsf


def test_version_exists() -> None:
    """Test that version attribute exists."""
    assert hasattr(jsf, "__version__")
    assert isinstance(jsf.__version__, str)


def test_version_format() -> None:
    """Test that version follows semantic versioning."""
    version = jsf.__version__
    # Should be in format X.Y.Z or X.Y.Z-suffix
    parts = version.split("-")[0].split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts[:3])


def test_author_exists() -> None:
    """Test that author attribute exists."""
    assert hasattr(jsf, "__author__")
    assert jsf.__author__ == "JBAC EdTech"


def test_license_exists() -> None:
    """Test that license attribute exists."""
    assert hasattr(jsf, "__license__")
    assert jsf.__license__ == "MIT"
