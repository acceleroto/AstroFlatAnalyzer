"""Illumination profile computation and statistics."""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

CORNER_REGION_SIZE = 50
CENTER_REGION_FRACTION = 0.02
FALLOFF_REGION_SIZE = 25
_GAUSSIAN_TRUNCATE = 4.0
_MIN_FILTER_PIXELS_PER_WORKER = 100_000


class ReferenceMode(str, Enum):
    CENTER = "center"
    MEDIAN = "median"


@dataclass(frozen=True)
class IlluminationStats:
    """Summary statistics for an illumination map."""

    min_pct: float
    max_pct: float
    median_pct: float
    corner_avg_pct: float
    center_pct: float


@dataclass(frozen=True)
class CornerFalloffStats:
    """Raw 25x25-region illumination comparison."""

    region_size: int
    center_average: float
    top_left_average: float
    top_right_average: float
    bottom_left_average: float
    bottom_right_average: float
    top_left_pct: float
    top_right_pct: float
    bottom_left_pct: float
    bottom_right_pct: float
    top_left_falloff_pct: float
    top_right_falloff_pct: float
    bottom_left_falloff_pct: float
    bottom_right_falloff_pct: float


@dataclass(frozen=True)
class ExportDimensions:
    """Export pixel dimensions with locked aspect ratio."""

    width: int
    height: int


def _center_region_median(image: np.ndarray) -> float:
    """Return a robust median from the central 2% of the image."""
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    half_height = max(1, round(ny * CENTER_REGION_FRACTION / 2))
    half_width = max(1, round(nx * CENTER_REGION_FRACTION / 2))
    region = image[
        max(0, cy - half_height) : min(ny, cy + half_height + 1),
        max(0, cx - half_width) : min(nx, cx + half_width + 1),
    ]
    return float(np.nanmedian(region))


def compute_reference_value(
    image: np.ndarray,
    mode: ReferenceMode,
) -> float:
    """Return the normalization reference value for the image."""
    if mode == ReferenceMode.CENTER:
        ref = _center_region_median(image)
    else:
        ref = float(np.median(image))

    if not np.isfinite(ref) or ref <= 0:
        raise ValueError(
            "Reference value is zero, negative, or non-finite. "
            "Cannot compute illumination percentages."
        )
    return ref


def normalize_illumination(
    image: np.ndarray,
    mode: ReferenceMode = ReferenceMode.CENTER,
) -> np.ndarray:
    """Convert flat image to illumination percentage map."""
    ref = compute_reference_value(image, mode)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (image / ref) * 100.0
    return np.where(np.isfinite(pct), pct, np.nan)


def _finite_mean(region: np.ndarray, name: str) -> float:
    """Return the finite mean of a region or raise a useful error."""
    values = region[np.isfinite(region)]
    if values.size == 0:
        raise ValueError(f"{name} region contains no finite pixel values.")
    return float(np.mean(values))


def _region_slices(
    image: np.ndarray,
    region_size: int,
) -> tuple[slice, slice, slice, slice, slice, slice, slice, slice, slice, slice]:
    """Return center and corner slices for a fixed-size image region."""
    if region_size <= 0:
        raise ValueError("Falloff region size must be positive.")

    ny, nx = image.shape
    height = min(region_size, ny)
    width = min(region_size, nx)
    center_y = max(0, (ny - height) // 2)
    center_x = max(0, (nx - width) // 2)

    return (
        slice(center_y, center_y + height),
        slice(center_x, center_x + width),
        slice(0, height),
        slice(0, width),
        slice(0, height),
        slice(nx - width, nx),
        slice(ny - height, ny),
        slice(0, width),
        slice(ny - height, ny),
        slice(nx - width, nx),
    )


def compute_corner_falloff(
    image: np.ndarray,
    region_size: int = FALLOFF_REGION_SIZE,
) -> CornerFalloffStats:
    """Compare four corner-region means with the central-region mean.

    This operates on the image supplied to it, before display smoothing. For
    Bayer flats, callers should supply the already 2x2-reduced analysis image.
    A positive falloff means the corner is dimmer than the center; a negative
    value means that corner is brighter than the center.
    """
    (
        center_y,
        center_x,
        top_left_y,
        top_left_x,
        top_right_y,
        top_right_x,
        bottom_left_y,
        bottom_left_x,
        bottom_right_y,
        bottom_right_x,
    ) = _region_slices(image, region_size)

    center_average = _finite_mean(image[center_y, center_x], "Center")
    if center_average <= 0:
        raise ValueError("Center falloff region has a zero or negative average.")

    averages = {
        "top_left": _finite_mean(image[top_left_y, top_left_x], "Top-left"),
        "top_right": _finite_mean(image[top_right_y, top_right_x], "Top-right"),
        "bottom_left": _finite_mean(image[bottom_left_y, bottom_left_x], "Bottom-left"),
        "bottom_right": _finite_mean(
            image[bottom_right_y, bottom_right_x],
            "Bottom-right",
        ),
    }
    percentages = {
        name: (average / center_average) * 100.0
        for name, average in averages.items()
    }
    falloffs = {
        name: 100.0 - percentage
        for name, percentage in percentages.items()
    }

    return CornerFalloffStats(
        region_size=min(region_size, image.shape[0], image.shape[1]),
        center_average=center_average,
        top_left_average=averages["top_left"],
        top_right_average=averages["top_right"],
        bottom_left_average=averages["bottom_left"],
        bottom_right_average=averages["bottom_right"],
        top_left_pct=percentages["top_left"],
        top_right_pct=percentages["top_right"],
        bottom_left_pct=percentages["bottom_left"],
        bottom_right_pct=percentages["bottom_right"],
        top_left_falloff_pct=falloffs["top_left"],
        top_right_falloff_pct=falloffs["top_right"],
        bottom_left_falloff_pct=falloffs["bottom_left"],
        bottom_right_falloff_pct=falloffs["bottom_right"],
    )


def smooth_illumination(
    illumination_pct: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply Gaussian smoothing to an illumination map (legacy; prefer compute_illumination_map)."""
    if sigma <= 0:
        return illumination_pct.copy()

    masked = np.nan_to_num(illumination_pct, nan=0.0)
    smoothed = gaussian_filter(masked, sigma=sigma)

    nan_mask = ~np.isfinite(illumination_pct)
    if nan_mask.any():
        smoothed = smoothed.astype(np.float64)
        smoothed[nan_mask] = np.nan

    return smoothed


def _filter_row_ranges(height: int, workers: int) -> list[tuple[int, int]]:
    """Split image rows into contiguous, non-empty worker ranges."""
    return [
        (height * index // workers, height * (index + 1) // workers)
        for index in range(workers)
    ]


def _effective_filter_workers(image: np.ndarray, requested: int) -> int:
    """Bound filter parallelism by image size and available row ranges."""
    if requested <= 1 or image.ndim != 2:
        return 1

    by_pixels = max(1, image.size // _MIN_FILTER_PIXELS_PER_WORKER)
    return max(1, min(int(requested), image.shape[0], by_pixels))


def _parallel_gaussian_filter(
    image: np.ndarray,
    sigma: float,
    workers: int,
    executor: Executor | None = None,
) -> np.ndarray:
    """Apply a Gaussian filter in independent row tiles.

    SciPy's Gaussian filter is separable but does not expose a worker-count
    argument. The vertical pass uses a radius-sized halo for each tile; the
    horizontal pass is independent for each core tile. This preserves the
    default reflect boundary behavior while allowing the two C-level filter
    passes to run concurrently.
    """
    worker_count = _effective_filter_workers(image, workers)
    if worker_count <= 1:
        return gaussian_filter(
            image,
            sigma=sigma,
            truncate=_GAUSSIAN_TRUNCATE,
        )

    radius = int(round(_GAUSSIAN_TRUNCATE * sigma))
    ranges = _filter_row_ranges(image.shape[0], worker_count)

    def vertical_pass(row_range: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        start, end = row_range
        extended_start = max(0, start - radius)
        extended_end = min(image.shape[0], end + radius)
        extended = image[extended_start:extended_end]
        filtered = gaussian_filter1d(
            extended,
            sigma=sigma,
            axis=0,
            mode="reflect",
            truncate=_GAUSSIAN_TRUNCATE,
        )
        core_start = start - extended_start
        core_end = end - extended_start
        return start, end, filtered[core_start:core_end]

    def run_filter(active_executor: Executor) -> np.ndarray:
        vertical = np.empty_like(image, dtype=np.float64)
        for start, end, filtered in active_executor.map(vertical_pass, ranges):
            vertical[start:end] = filtered

        def horizontal_pass(row_range: tuple[int, int]) -> tuple[int, int, np.ndarray]:
            start, end = row_range
            filtered = gaussian_filter1d(
                vertical[start:end],
                sigma=sigma,
                axis=1,
                mode="reflect",
                truncate=_GAUSSIAN_TRUNCATE,
            )
            return start, end, filtered

        result = np.empty_like(image, dtype=np.float64)
        for start, end, filtered in active_executor.map(horizontal_pass, ranges):
            result[start:end] = filtered
        return result

    if executor is not None:
        return run_filter(executor)

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="preview-filter",
    ) as active_executor:
        return run_filter(active_executor)


def compute_illumination_map(
    image: np.ndarray,
    mode: ReferenceMode = ReferenceMode.CENTER,
    sigma: float = 0,
    *,
    workers: int = 1,
    executor: Executor | None = None,
) -> np.ndarray:
    """Smooth the raw flat, then normalize to illumination percentages.

    Smoothing is applied to pixel intensities before normalization so the
    reference point (center or median) stays at 100% after smoothing.
    """
    processed = image.astype(np.float64, copy=False)
    if sigma > 0:
        processed = _parallel_gaussian_filter(
            processed,
            sigma=sigma,
            workers=workers,
            executor=executor,
        )
    return normalize_illumination(processed, mode)


def compute_contour_levels(
    max_level: float,
    min_level: float,
    step: float,
) -> np.ndarray:
    """Generate descending contour levels from max to min."""
    if step <= 0:
        raise ValueError("Contour step must be positive.")
    levels = np.arange(max_level, min_level - step * 0.5, -step)
    return levels[levels >= min_level]


def effective_contour_levels(
    illumination_pct: np.ndarray,
    requested_levels: np.ndarray,
    step: float,
) -> np.ndarray:
    """Return contour levels that intersect the actual illumination data range."""
    finite = illumination_pct[np.isfinite(illumination_pct)]
    if finite.size == 0:
        return np.array([])

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))

    levels = np.sort(requested_levels.astype(float))
    levels = levels[(levels >= data_min) & (levels <= data_max)]

    if levels.size == 0 and step > 0:
        top = step * np.floor(data_max / step)
        bottom = step * np.ceil(data_min / step)
        if top >= bottom:
            # Matplotlib requires strictly increasing contour levels
            levels = np.arange(bottom, top + step * 0.5, step)
            levels = levels[(levels >= data_min) & (levels <= data_max)]

    return np.unique(levels.astype(float))


def compute_stats(illumination_pct: np.ndarray) -> IlluminationStats:
    """Compute illumination summary statistics."""
    finite = illumination_pct[np.isfinite(illumination_pct)]
    if finite.size == 0:
        raise ValueError("No finite illumination values to summarize.")

    ny, nx = illumination_pct.shape
    region = min(CORNER_REGION_SIZE, ny // 4, nx // 4)
    region = max(region, 1)

    corners = [
        illumination_pct[:region, :region],
        illumination_pct[:region, -region:],
        illumination_pct[-region:, :region],
        illumination_pct[-region:, -region:],
    ]
    corner_vals = np.concatenate([c[np.isfinite(c)] for c in corners])
    corner_avg = float(np.mean(corner_vals)) if corner_vals.size else float("nan")

    return IlluminationStats(
        min_pct=float(np.min(finite)),
        max_pct=float(np.max(finite)),
        median_pct=float(np.median(finite)),
        corner_avg_pct=corner_avg,
        center_pct=_center_region_median(illumination_pct),
    )


def compute_export_dimensions(
    source_width: int,
    source_height: int,
    export_width: int,
    min_width: int = 64,
) -> ExportDimensions:
    """Compute export height from width, locking aspect ratio to source FITS."""
    aspect = source_width / source_height
    clamped_width = int(np.clip(export_width, min_width, source_width))
    export_height = max(1, round(clamped_width / aspect))
    return ExportDimensions(width=clamped_width, height=export_height)


def resample_for_export(
    illumination_pct: np.ndarray,
    export_width: int,
    export_height: int,
) -> np.ndarray:
    """Resample illumination map to export dimensions using bilinear zoom.

    The preallocated output keeps the existing ``scipy.ndimage.zoom``
    interpolation and coordinate mapping while avoiding its result copy.
    Non-finite values retain the historical replacement-with-zero behavior.
    """
    ny, nx = illumination_pct.shape
    if export_width == nx and export_height == ny:
        return illumination_pct.copy()

    zoom_y = export_height / ny
    zoom_x = export_width / nx

    from scipy.ndimage import zoom

    if np.isfinite(illumination_pct).all():
        filled = illumination_pct
    else:
        filled = np.nan_to_num(illumination_pct, nan=0.0)

    resampled = np.empty(
        (export_height, export_width),
        dtype=np.float64,
    )
    zoom(
        filled,
        (zoom_y, zoom_x),
        output=resampled,
        order=1,
    )

    if resampled.shape != (export_height, export_width):
        resampled = resampled[:export_height, :export_width]

    return resampled.astype(np.float64, copy=False)


def reduce_for_preview(
    image: np.ndarray,
    max_dimension: int = 900,
) -> tuple[np.ndarray, int]:
    """Reduce an image by integer-sized finite-pixel block averaging.

    The returned factor is the number of source pixels represented by each
    preview pixel along the reduced dimensions. Edge blocks may contain fewer
    pixels when the source dimensions are not divisible by the factor.
    """
    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("Preview reduction requires a two-dimensional image.")
    if max_dimension <= 0:
        raise ValueError("Preview maximum dimension must be positive.")

    ny, nx = source.shape
    if ny == 0 or nx == 0:
        raise ValueError("Preview reduction requires a non-empty image.")

    factor = max(1, int(np.ceil(max(ny, nx) / max_dimension)))
    if factor == 1:
        return source.copy(), factor

    y_starts = np.arange(0, ny, factor, dtype=int)
    x_starts = np.arange(0, nx, factor, dtype=int)
    finite = np.isfinite(source)
    values = np.where(finite, source, 0.0)

    summed = np.add.reduceat(values, y_starts, axis=0)
    summed = np.add.reduceat(summed, x_starts, axis=1)
    counts = np.add.reduceat(finite.astype(np.int64), y_starts, axis=0)
    counts = np.add.reduceat(counts, x_starts, axis=1)

    reduced = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, counts, out=reduced, where=counts > 0)
    return reduced, factor


def downsample_for_preview(
    illumination_pct: np.ndarray,
    max_width: int = 1200,
) -> np.ndarray:
    """Downsample for UI preview."""
    reduced, _ = reduce_for_preview(illumination_pct, max_dimension=max_width)
    return reduced
