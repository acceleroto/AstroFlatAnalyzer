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


def test_parse_slider_value_rounds_and_clamps():
    assert MainWindow._parse_slider_value("101.6", 25, 200) == 102
    assert MainWindow._parse_slider_value("10", 25, 200) == 25
    assert MainWindow._parse_slider_value("250", 25, 200) == 200


def test_parse_slider_value_rejects_non_finite_or_invalid_values():
    assert MainWindow._parse_slider_value("not a number", 1, 10) is None
    assert MainWindow._parse_slider_value("nan", 1, 10) is None
