"""Stage-isolated benchmark for interactive preview generation.

Run from the project root, for example:

    python performance_data/preview_benchmark.py
    python performance_data/preview_benchmark.py performance_data/test_flat.fits \
        --targets 900,1200,1800 --workers 1,2,4,8

The benchmark uses Matplotlib's non-interactive Agg backend. Its draw timing
therefore measures the rasterization path without requiring a display; the
TkAgg canvas draw remains an application-level measurement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import scipy

from flat_analyzer.fits_loader import LoadedFlat, load_flat
from flat_analyzer.illumination import (
    compute_contour_levels,
    compute_corner_falloff,
    compute_illumination_map,
    compute_stats,
    reduce_for_preview,
)
from flat_analyzer.processing import compute_preview_result
from flat_analyzer.visualization import RenderSettings, build_preview_figure

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None  # type: ignore[assignment]


DEFAULT_TARGETS = (900, 1200, 1800)
DEFAULT_WORKERS = (1, 2, 4, 8)
DEFAULT_CONTOURS = (True, False)
DEFAULT_ITERATIONS = 3


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positives.")
    return values


def _parse_contour_modes(raw: str) -> tuple[bool, ...]:
    modes = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized in {"on", "true", "1"}:
            modes.append(True)
        elif normalized in {"off", "false", "0"}:
            modes.append(False)
        else:
            raise argparse.ArgumentTypeError(
                "Contour modes must be on/off, true/false, or 1/0."
            )
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one contour mode.")
    return tuple(modes)


def _median_time(function: Callable[[], Any], iterations: int) -> float:
    """Return a warm median duration in seconds."""
    function()
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def _median_wall_cpu_time(
    function: Callable[[], Any],
    iterations: int,
) -> tuple[float, float]:
    """Return warm median wall and process CPU durations."""
    function()
    samples = []
    for _ in range(iterations):
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        function()
        samples.append(
            (
                time.perf_counter() - wall_started,
                time.process_time() - cpu_started,
            )
        )
    return (
        statistics.median(sample[0] for sample in samples),
        statistics.median(sample[1] for sample in samples),
    )


def _peak_rss_mib() -> float | None:
    """Return process high-water RSS when the platform exposes it."""
    if resource is None:
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None

    # macOS reports bytes; Linux and most other Unix systems report KiB.
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _load_with_timing(path: Path, iterations: int) -> tuple[LoadedFlat, float]:
    """Load the canonical image and report warm FITS-load timing."""
    loaded = load_flat(path)
    load_seconds = _median_time(lambda: load_flat(path), iterations)
    return loaded, load_seconds


def _render_settings(show_contours: bool) -> RenderSettings:
    return RenderSettings(
        vmin=80.0,
        vmax=100.0,
        show_contours=show_contours,
        contour_levels=compute_contour_levels(100.0, 0.0, 2.0),
        contour_step=2.0,
        reference_label="center",
    )


def _benchmark_target(
    loaded: LoadedFlat,
    target: int,
    workers: Iterable[int],
    contour_modes: Iterable[bool],
    iterations: int,
    load_seconds: float,
) -> list[dict[str, Any]]:
    """Measure all preview stages for one target size."""
    reduction_seconds = _median_time(
        lambda: reduce_for_preview(loaded.image, max_dimension=target),
        iterations,
    )
    preview_image, reduction_factor = reduce_for_preview(
        loaded.image,
        max_dimension=target,
    )
    analysis_sigma = 100.0 / 2.0 if loaded.metadata.bayer_reduced else 100.0
    preview_sigma = analysis_sigma / reduction_factor
    results: list[dict[str, Any]] = []

    for worker_count in workers:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="benchmark-preview",
        ) as filter_executor:
            map_seconds, map_cpu_seconds = _median_wall_cpu_time(
                lambda: compute_illumination_map(
                    preview_image,
                    sigma=preview_sigma,
                    workers=worker_count,
                    executor=filter_executor,
                ),
                iterations,
            )
            illumination = compute_illumination_map(
                preview_image,
                sigma=preview_sigma,
                workers=worker_count,
                executor=filter_executor,
            )
            stats_seconds = _median_time(
                lambda: compute_stats(illumination),
                iterations,
            )
            preview_seconds, preview_cpu_seconds = _median_wall_cpu_time(
                lambda: compute_preview_result(
                    preview_image,
                    sigma=preview_sigma,
                    worker_count=worker_count,
                    executor=filter_executor,
                ),
                iterations,
            )

        for show_contours in contour_modes:
            settings = _render_settings(show_contours)
            figure_seconds = _median_time(
                lambda: _build_and_clear(illumination, settings),
                iterations,
            )
            figure = build_preview_figure(illumination, settings)
            canvas = FigureCanvasAgg(figure)
            draw_seconds = _median_time(canvas.draw, iterations)
            figure.clear()

            results.append(
                {
                    "target_px": target,
                    "worker_count": worker_count,
                    "contours": show_contours,
                    "shape": list(preview_image.shape),
                    "reduction_factor": reduction_factor,
                    "preview_sigma": preview_sigma,
                    "load_s": load_seconds,
                    "reduction_s": reduction_seconds,
                    "illumination_map_s": map_seconds,
                    "illumination_map_cpu_s": map_cpu_seconds,
                    "stats_s": stats_seconds,
                    "preview_result_s": preview_seconds,
                    "preview_result_cpu_s": preview_cpu_seconds,
                    "figure_build_s": figure_seconds,
                    "agg_draw_s": draw_seconds,
                    "peak_rss_mib": _peak_rss_mib(),
                }
            )

    return results


def _build_and_clear(
    illumination: np.ndarray,
    settings: RenderSettings,
) -> None:
    figure = build_preview_figure(illumination, settings)
    figure.clear()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("test_flat.fits"),
        help="FITS file to benchmark.",
    )
    parser.add_argument(
        "--targets",
        type=_parse_int_list,
        default=DEFAULT_TARGETS,
        help="Comma-separated preview long-edge targets.",
    )
    parser.add_argument(
        "--workers",
        type=_parse_int_list,
        default=DEFAULT_WORKERS,
        help="Comma-separated Gaussian worker counts.",
    )
    parser.add_argument(
        "--contours",
        type=_parse_contour_modes,
        default=DEFAULT_CONTOURS,
        help="Comma-separated contour modes: on, off, or both.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of timed iterations after one warm-up.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of formatted text.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if not args.path.is_file():
        raise SystemExit(f"FITS file not found: {args.path}")

    gc.collect()
    loaded, load_seconds = _load_with_timing(args.path, args.iterations)
    corner_seconds = _median_time(
        lambda: compute_corner_falloff(loaded.image),
        args.iterations,
    )
    rows = []
    for target in args.targets:
        rows.extend(
            _benchmark_target(
                loaded,
                target,
                args.workers,
                args.contours,
                args.iterations,
                load_seconds,
            )
        )

    report = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "cpu_count": os.cpu_count(),
        "path": str(args.path),
        "source_shape": list(loaded.image.shape),
        "source_bayer_reduced": loaded.metadata.bayer_reduced,
        "corner_falloff_s": corner_seconds,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Preview benchmark: {args.path} · source {loaded.image.shape} · "
            f"CPU count {os.cpu_count()}"
        )
        print(
            f"Versions: Python {report['python']}, NumPy {report['numpy']}, "
            f"SciPy {report['scipy']}, Matplotlib {report['matplotlib']}"
        )
        print(f"Corner falloff: {corner_seconds:.6f} s")
        for row in rows:
            print(
                f"{row['target_px']:4d}px "
                f"workers={row['worker_count']:2d} "
                f"contours={'on' if row['contours'] else 'off'} "
                f"shape={tuple(row['shape'])} "
                f"map={row['illumination_map_s']:.6f}s "
                f"mapCPU/wall={row['illumination_map_cpu_s'] / row['illumination_map_s']:.2f} "
                f"stats={row['stats_s']:.6f}s "
                f"preview={row['preview_result_s']:.6f}s "
                f"figure={row['figure_build_s']:.6f}s "
                f"AggDraw={row['agg_draw_s']:.6f}s "
                f"RSS={row['peak_rss_mib']:.1f}MiB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
