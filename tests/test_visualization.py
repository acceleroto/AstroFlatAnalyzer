"""Tests for visualization rendering."""

import numpy as np

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
