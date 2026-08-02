"""Tests for desktop UI helper behavior that does not require a window."""

from pathlib import Path

from flat_analyzer.ui.main_window import MainWindow


def test_parse_drop_paths_preserves_spaces_in_braced_paths():
    dropped = "{/Users/test data/flat frame.fits} {/Users/other.fit}"

    assert MainWindow._parse_drop_paths(dropped) == Path(
        "/Users/test data/flat frame.fits"
    )


def test_parse_drop_paths_accepts_unbraced_paths():
    assert MainWindow._parse_drop_paths("/tmp/flat.fits") == Path("/tmp/flat.fits")
