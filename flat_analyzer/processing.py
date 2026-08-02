"""Worker-safe numerical processing for previews and exports."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
import os

import numpy as np

from flat_analyzer.illumination import (
    IlluminationStats,
    ReferenceMode,
    compute_illumination_map,
    compute_stats,
    resample_for_export,
)

PREVIEW_MAX_WORKERS = 8
PREVIEW_MIN_PIXELS_PER_WORKER = 100_000


@dataclass(frozen=True)
class PreviewResult:
    """Numerical preview data ready for main-thread rendering."""

    illumination: np.ndarray
    stats: IlluminationStats


@dataclass(frozen=True)
class ExportResult:
    """Full-resolution illumination and requested export data."""

    illumination: np.ndarray
    export_data: np.ndarray


def preview_worker_capacity() -> int:
    """Return the bounded number of filter workers for this machine."""
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return min(PREVIEW_MAX_WORKERS, cpu_count - 1)


def preview_worker_count(
    image: np.ndarray,
    requested: int | None = None,
) -> int:
    """Choose preview workers without oversubscribing small images."""
    if image.ndim != 2 or image.size < PREVIEW_MIN_PIXELS_PER_WORKER:
        return 1

    capacity = preview_worker_capacity() if requested is None else int(requested)
    if capacity <= 1:
        return 1

    by_pixels = max(1, image.size // PREVIEW_MIN_PIXELS_PER_WORKER)
    return max(1, min(capacity, image.shape[0], by_pixels))


def compute_preview_result(
    image: np.ndarray,
    sigma: float,
    *,
    worker_count: int | None = None,
    executor: Executor | None = None,
) -> PreviewResult:
    """Compute a preview illumination map and its display statistics."""
    effective_workers = preview_worker_count(image, worker_count)
    illumination = compute_illumination_map(
        image,
        mode=ReferenceMode.CENTER,
        sigma=sigma,
        workers=effective_workers,
        executor=executor,
    )
    return PreviewResult(
        illumination=illumination,
        stats=compute_stats(illumination),
    )


def compute_export_result(
    image: np.ndarray,
    sigma: float,
    export_width: int,
    export_height: int,
    *,
    worker_count: int = 1,
    executor: Executor | None = None,
) -> ExportResult:
    """Compute a full-resolution map and resample it for export."""
    illumination = compute_illumination_map(
        image,
        mode=ReferenceMode.CENTER,
        sigma=sigma,
        workers=worker_count,
        executor=executor,
    )
    return ExportResult(
        illumination=illumination,
        export_data=resample_for_export(
            illumination,
            export_width,
            export_height,
        ),
    )
