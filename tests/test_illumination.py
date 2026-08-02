"""Tests for illumination profile computation."""

import numpy as np
import pytest

from flat_analyzer.illumination import (
    ReferenceMode,
    compute_contour_levels,
    compute_corner_falloff,
    compute_export_dimensions,
    compute_illumination_map,
    compute_stats,
    effective_contour_levels,
    normalize_illumination,
    resample_for_export,
    smooth_illumination,
)


def _make_vignetting(shape: tuple[int, int], sigma_frac: float = 0.35) -> np.ndarray:
    """Synthetic flat with Gaussian vignetting, peak at center."""
    ny, nx = shape
    cy, cx = ny / 2, nx / 2
    y, x = np.mgrid[0:ny, 0:nx]
    r2 = ((x - cx) / nx) ** 2 + ((y - cy) / ny) ** 2
    return 1000.0 * np.exp(-r2 / (2 * sigma_frac**2))


def test_center_normalization_region_median_is_100():
    image = _make_vignetting((200, 300))
    result = normalize_illumination(image, ReferenceMode.CENTER)
    center = result[98:103, 147:154]
    assert np.median(center) == pytest.approx(100.0, rel=1e-5)


def test_corners_lower_than_center():
    image = _make_vignetting((200, 300))
    result = normalize_illumination(image, ReferenceMode.CENTER)
    assert result[0, 0] < result[100, 150]
    assert result[-1, -1] < result[100, 150]


def test_smoothing_increases_corner_values():
    image = _make_vignetting((100, 100))
    raw = compute_illumination_map(image, ReferenceMode.CENTER, sigma=0)
    smoothed = compute_illumination_map(image, ReferenceMode.CENTER, sigma=10)
    assert smoothed[0, 0] > raw[0, 0]


def test_smooth_before_normalize_keeps_center_region_at_100():
    image = _make_vignetting((200, 200))
    result = compute_illumination_map(image, ReferenceMode.CENTER, sigma=20)
    assert np.median(result[98:103, 98:103]) == pytest.approx(100.0, rel=1e-5)


def test_smoothing_sigma_zero_unchanged():
    image = _make_vignetting((50, 50))
    raw = normalize_illumination(image, ReferenceMode.CENTER)
    mapped = compute_illumination_map(image, ReferenceMode.CENTER, sigma=0)
    assert np.allclose(mapped, raw)


def test_median_reference():
    image = _make_vignetting((100, 100))
    result = normalize_illumination(image, ReferenceMode.MEDIAN)
    assert np.median(result) == pytest.approx(100.0, rel=0.01)


def test_compute_contour_levels_descending():
    levels = compute_contour_levels(100, 70, 5)
    assert list(levels) == [100, 95, 90, 85, 80, 75, 70]


def test_export_dimensions_lock_aspect():
    dims = compute_export_dimensions(6000, 4000, 3000)
    assert dims.width == 3000
    assert dims.height == 2000


def test_export_dimensions_clamp_to_native():
    dims = compute_export_dimensions(6000, 4000, 9000)
    assert dims.width == 6000
    assert dims.height == 4000


def test_export_dimensions_minimum_width():
    dims = compute_export_dimensions(6000, 4000, 10)
    assert dims.width == 64


def test_resample_for_export_shape():
    image = _make_vignetting((400, 600))
    raw = normalize_illumination(image, ReferenceMode.CENTER)
    out = resample_for_export(raw, 300, 200)
    assert out.shape == (200, 300)


def test_compute_stats():
    image = _make_vignetting((200, 200))
    raw = normalize_illumination(image, ReferenceMode.CENTER)
    stats = compute_stats(raw)
    assert stats.center_pct == pytest.approx(100.0, rel=1e-4)
    assert stats.corner_avg_pct < stats.center_pct


def test_center_reference_resists_single_bad_center_pixel():
    image = np.full((200, 200), 1000.0)
    image[100, 100] = 100.0
    result = normalize_illumination(image, ReferenceMode.CENTER)

    assert result[90, 90] == pytest.approx(100.0)
    assert result[100, 100] == pytest.approx(10.0)


def test_compute_25x25_corner_falloff_relative_to_center():
    image = np.full((100, 100), 1000.0)
    image[:25, :25] = 800.0
    image[:25, -25:] = 900.0
    image[-25:, :25] = 700.0
    image[-25:, -25:] = 1100.0

    result = compute_corner_falloff(image)

    assert result.region_size == 25
    assert result.center_average == pytest.approx(1000.0)
    assert result.top_left_pct == pytest.approx(80.0)
    assert result.top_right_pct == pytest.approx(90.0)
    assert result.bottom_left_pct == pytest.approx(70.0)
    assert result.bottom_right_pct == pytest.approx(110.0)
    assert result.top_left_falloff_pct == pytest.approx(20.0)
    assert result.bottom_right_falloff_pct == pytest.approx(-10.0)


def test_effective_contour_levels_filters_to_data_range():
    illumination = np.linspace(88, 100, 100).reshape(10, 10)
    requested = compute_contour_levels(100, 70, 5)
    levels = effective_contour_levels(illumination, requested, 5)
    assert levels.size > 0
    assert levels.min() >= 88
    assert levels.max() <= 100


def test_effective_contour_levels_adapts_when_requested_miss_data():
    illumination = np.linspace(96, 100, 100).reshape(10, 10)
    requested = compute_contour_levels(100, 70, 5)  # 70-85 never hit data
    levels = effective_contour_levels(illumination, requested, 5)
    assert levels.size > 0
    assert levels.min() >= 96
    assert np.all(np.diff(levels) > 0)


def test_zero_center_raises():
    image = np.zeros((10, 10))
    with pytest.raises(ValueError):
        normalize_illumination(image, ReferenceMode.CENTER)
