"""Tests for worker-safe numerical processing helpers."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from flat_analyzer.illumination import (
    ReferenceMode,
    compute_illumination_map,
    resample_for_export,
)
from flat_analyzer.processing import (
    compute_export_result,
    compute_preview_result,
    preview_worker_capacity,
    preview_worker_count,
)


def _make_image() -> np.ndarray:
    y, x = np.mgrid[0:40, 0:60]
    return 1000.0 + 100.0 * np.sin(x / 12.0) * np.cos(y / 9.0)


def test_compute_preview_result_returns_map_and_stats():
    result = compute_preview_result(_make_image(), sigma=3)

    assert result.illumination.shape == (40, 60)
    assert result.stats.center_pct == pytest.approx(100.0, rel=0.01)


def test_preview_worker_count_falls_back_for_small_images():
    image = np.ones((40, 60), dtype=float)

    assert preview_worker_count(image, requested=8) == 1


def test_preview_worker_capacity_is_serial_on_single_core(monkeypatch):
    monkeypatch.setattr("flat_analyzer.processing.os.cpu_count", lambda: 1)

    assert preview_worker_capacity() == 1


def test_compute_preview_result_threaded_path_matches_direct_path():
    image = np.ones((500, 800), dtype=float) * 1000
    image += np.sin(np.arange(image.shape[1], dtype=float))[None, :]

    direct = compute_preview_result(image, sigma=9, worker_count=1)
    threaded = compute_preview_result(image, sigma=9, worker_count=4)

    assert np.allclose(
        threaded.illumination,
        direct.illumination,
        equal_nan=True,
        rtol=0,
        atol=1e-12,
    )
    assert threaded.stats == direct.stats


def test_compute_export_result_matches_full_resolution_pipeline():
    image = _make_image()
    expected_illumination = compute_illumination_map(
        image,
        mode=ReferenceMode.CENTER,
        sigma=3,
    )
    expected_export = resample_for_export(expected_illumination, 30, 20)

    result = compute_export_result(image, sigma=3, export_width=30, export_height=20)

    assert np.allclose(result.illumination, expected_illumination)
    assert np.allclose(result.export_data, expected_export)


def test_threaded_export_result_matches_serial_pipeline_with_reusable_executor():
    image = np.ones((500, 800), dtype=float) * 1000
    image += np.sin(np.arange(image.shape[1], dtype=float))[None, :]

    serial = compute_export_result(
        image,
        sigma=9,
        export_width=640,
        export_height=400,
        worker_count=1,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = compute_export_result(
            image,
            sigma=9,
            export_width=640,
            export_height=400,
            worker_count=4,
            executor=executor,
        )

    assert np.allclose(
        threaded.illumination,
        serial.illumination,
        equal_nan=True,
        rtol=0,
        atol=1e-12,
    )
    assert np.array_equal(threaded.export_data, serial.export_data)
