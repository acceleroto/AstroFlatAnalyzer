"""Tests for illumination profile computation."""

import numpy as np
import pytest
from scipy.ndimage import zoom

from flat_analyzer.illumination import (
    ReferenceMode,
    compute_contour_levels,
    compute_corner_falloff,
    compute_export_dimensions,
    compute_illumination_map,
    compute_stats,
    effective_contour_levels,
    normalize_illumination,
    reduce_for_preview,
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


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_parallel_gaussian_matches_serial_result(workers):
    image = _make_vignetting((500, 800))
    image[0, 0] = np.nan

    serial = compute_illumination_map(
        image,
        ReferenceMode.CENTER,
        sigma=9,
        workers=1,
    )
    parallel = compute_illumination_map(
        image,
        ReferenceMode.CENTER,
        sigma=9,
        workers=workers,
    )

    assert np.allclose(parallel, serial, equal_nan=True, rtol=0, atol=1e-12)


def test_parallel_gaussian_can_use_a_reusable_executor():
    from concurrent.futures import ThreadPoolExecutor

    image = _make_vignetting((500, 800))
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = compute_illumination_map(
            image,
            ReferenceMode.CENTER,
            sigma=9,
            workers=4,
            executor=executor,
        )
    expected = compute_illumination_map(image, ReferenceMode.CENTER, sigma=9)

    assert np.allclose(result, expected, equal_nan=True, rtol=0, atol=1e-12)


def test_reduce_for_preview_uses_integer_area_averaging():
    image = np.arange(16, dtype=float).reshape(4, 4)
    reduced, factor = reduce_for_preview(image, max_dimension=2)

    assert factor == 2
    assert reduced.shape == (2, 2)
    assert np.allclose(reduced, [[2.5, 4.5], [10.5, 12.5]])
    assert np.array_equal(image, np.arange(16, dtype=float).reshape(4, 4))


def test_reduce_for_preview_handles_odd_dimensions_and_non_finite_values():
    image = np.arange(15, dtype=float).reshape(3, 5)
    image[0, 0] = np.nan

    reduced, factor = reduce_for_preview(image, max_dimension=2)

    assert factor == 3
    assert reduced.shape == (1, 2)
    assert np.all(np.isfinite(reduced))
    assert max(reduced.shape) <= 2


def test_reduce_for_preview_keeps_small_images_at_full_size():
    image = _make_vignetting((40, 60))

    reduced, factor = reduce_for_preview(image, max_dimension=900)

    assert factor == 1
    assert reduced.shape == image.shape
    assert reduced is not image
    assert np.array_equal(reduced, image)


@pytest.mark.parametrize("max_dimension", [1200, 1500, 1800])
def test_reduced_preview_is_close_to_full_resolution_reference(max_dimension):
    ny, nx = 300, 2500
    y, x = np.mgrid[0:ny, 0:nx]
    image = _make_vignetting((ny, nx))
    image *= 1.0 + 0.05 * np.sin(x / 12.0) * np.sin(y / 17.0)
    sigma = 25.0

    full_illumination = compute_illumination_map(
        image, ReferenceMode.CENTER, sigma=sigma
    )
    reference, _ = reduce_for_preview(full_illumination, max_dimension=max_dimension)
    preview_source, factor = reduce_for_preview(image, max_dimension=max_dimension)
    preview = compute_illumination_map(
        preview_source,
        ReferenceMode.CENTER,
        sigma=sigma / factor,
    )

    assert preview.shape == reference.shape
    assert np.mean(np.abs(preview - reference)) < 0.1

    levels = compute_contour_levels(100, 70, 5)
    assert np.array_equal(
        effective_contour_levels(reference, levels, 5),
        effective_contour_levels(preview, levels, 5),
    )


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


def test_resample_for_export_matches_zoom_for_odd_nonfinite_input():
    image = np.arange(17 * 23, dtype=float).reshape(17, 23)
    image[2, 3] = np.nan
    image[8, 11] = np.inf
    expected = zoom(
        np.nan_to_num(image, nan=0.0),
        (29 / 17, 31 / 23),
        order=1,
    )

    actual = resample_for_export(image, 31, 29)

    assert actual.shape == (29, 31)
    assert np.array_equal(actual, expected)


def test_resample_for_export_native_dimensions_return_an_independent_copy():
    image = _make_vignetting((17, 23))

    actual = resample_for_export(image, 23, 17)
    actual[0, 0] = -1

    assert actual.shape == image.shape
    assert image[0, 0] != -1


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
