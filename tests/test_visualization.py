"""Tests for visualization rendering."""

from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from flat_analyzer.illumination import (
    ReferenceMode,
    compute_contour_levels,
    effective_contour_levels,
    normalize_illumination,
)
from flat_analyzer.visualization import (
    RenderSettings,
    build_export_figure,
    build_preview_figure,
    save_figure,
)


def test_build_preview_figure(tmp_path):
    image = np.ones((100, 150)) * 1000
    raw = normalize_illumination(image, ReferenceMode.CENTER)
    settings = RenderSettings(
        vmin=60,
        vmax=100,
        show_contours=True,
        contour_levels=np.array([100, 90, 80]),
        reference_label="center",
    )
    fig = build_preview_figure(raw, settings)
    assert fig is not None
    fig.clf()


def test_save_export_figure(tmp_path):
    image = np.linspace(800, 1200, 100 * 100).reshape(100, 100)
    raw = normalize_illumination(image, ReferenceMode.CENTER)
    settings = RenderSettings(
        vmin=60,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
        reference_label="center",
    )
    fig = build_export_figure(raw, settings, 150, 100)
    out = tmp_path / "out.png"
    save_figure(fig, out, fmt="png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_contours_are_wider_and_labels_are_larger():
    from flat_analyzer.visualization import _contour_linewidth, _label_fontsize

    assert _contour_linewidth(1800, 1000, True) == pytest.approx(5.0)
    assert _contour_linewidth(1800, 1000, True, outline=True) == pytest.approx(
        11.0
    )
    assert _label_fontsize(1800, 1000, True) == pytest.approx(32.0 * 4.0 / 3.0)

    # Preview styling remains unchanged.
    assert _contour_linewidth(1800, 1000, False) == pytest.approx(1.0)
    assert _label_fontsize(1800, 1000, False) == pytest.approx(9.0)


def test_explicit_agg_export_matches_legacy_png_pixels(tmp_path):
    image = np.linspace(80, 100, 20 * 30).reshape(20, 30)
    settings = _export_settings(False)
    legacy_path = tmp_path / "legacy.png"
    agg_path = tmp_path / "agg.png"

    legacy_figure = build_export_figure(image, settings, 60, 40)
    legacy_figure.savefig(
        legacy_path,
        format="png",
        dpi=100,
        pad_inches=0,
    )
    save_figure(
        build_export_figure(image, settings, 60, 40),
        agg_path,
        fmt="png",
    )

    legacy_dimensions, legacy_pixels = _decoded_pixels(legacy_path)
    agg_dimensions, agg_pixels = _decoded_pixels(agg_path)
    assert agg_dimensions == legacy_dimensions == (60, 40)
    assert np.array_equal(agg_pixels, legacy_pixels)


def _export_settings(show_contours: bool) -> RenderSettings:
    return RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=show_contours,
        contour_levels=np.array([90, 95]),
        contour_step=5,
        reference_label="center",
    )


def _decoded_pixels(path):
    with Image.open(path) as image:
        dimensions = image.size
        pixels = np.asarray(image.convert("RGBA")).copy()
    return dimensions, pixels


def test_png_compression_preserves_decoded_pixels_and_dimensions(tmp_path):
    image = np.linspace(80, 100, 20 * 30).reshape(20, 30)
    default_path = tmp_path / "default.png"
    fast_path = tmp_path / "fast.png"

    default_figure = build_export_figure(
        image,
        _export_settings(False),
        60,
        40,
    )
    fast_figure = build_export_figure(
        image,
        _export_settings(False),
        60,
        40,
    )
    save_figure(default_figure, default_path, fmt="png")
    save_figure(
        fast_figure,
        fast_path,
        fmt="png",
        png_compress_level=1,
    )

    default_dimensions, default_pixels = _decoded_pixels(default_path)
    fast_dimensions, fast_pixels = _decoded_pixels(fast_path)
    assert default_dimensions == (60, 40)
    assert fast_dimensions == default_dimensions
    assert np.array_equal(fast_pixels, default_pixels)


def test_contours_change_raster_but_preserve_exact_dimensions(tmp_path):
    image = np.linspace(80, 100, 20 * 30).reshape(20, 30)
    no_contours = tmp_path / "no-contours.png"
    with_contours = tmp_path / "with-contours.png"

    save_figure(
        build_export_figure(image, _export_settings(False), 60, 40),
        no_contours,
    )
    save_figure(
        build_export_figure(image, _export_settings(True), 60, 40),
        with_contours,
    )

    no_dimensions, no_pixels = _decoded_pixels(no_contours)
    contours_dimensions, contours_pixels = _decoded_pixels(with_contours)
    assert no_dimensions == contours_dimensions == (60, 40)
    assert not np.array_equal(no_pixels, contours_pixels)


def test_jpeg_quality_is_passed_unchanged_to_matplotlib(tmp_path):
    figure = build_export_figure(
        np.ones((20, 30)) * 90,
        _export_settings(False),
        60,
        40,
    )
    figure.savefig = Mock()

    save_figure(figure, tmp_path / "quality.jpg", fmt="jpeg", jpg_quality=73)

    figure.savefig.assert_called_once()
    kwargs = figure.savefig.call_args.kwargs
    assert kwargs["format"] == "jpeg"
    assert kwargs["pil_kwargs"] == {"quality": 73}
