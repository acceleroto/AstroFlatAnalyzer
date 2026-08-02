"""Stage-isolated benchmark for full-resolution image export.

Run from the project root, for example:

    .venv/bin/python performance_data/export_benchmark.py path/to/flat.fits \
        --targets native,50%,1920w --workers 1,2,4,8 --formats png,jpeg \
        --contours on,off --iterations 1

The benchmark uses the same numerical, Agg-rendering, and Pillow paths as the
application, but performs the final replace into a temporary benchmark
directory. It reports both cold exports (map calculation included) and cached
exports (the illumination map is reused).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image
import scipy

from flat_analyzer.fits_loader import LoadedFlat, load_flat
from flat_analyzer.illumination import (
    compute_contour_levels,
    compute_export_dimensions,
    compute_illumination_map,
    resample_for_export,
)
from flat_analyzer.processing import compute_export_result
from flat_analyzer.visualization import (
    EXPORT_DPI,
    RenderSettings,
    build_export_figure,
    save_figure,
)

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None  # type: ignore[assignment]


DEFAULT_WORKERS = (1, 2, 4, 8)
DEFAULT_FORMATS = ("png", "jpeg")
DEFAULT_CONTOURS = (True, False)
DEFAULT_ITERATIONS = 1


@dataclass(frozen=True)
class _Target:
    name: str
    width: int
    height: int


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


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized in {"jpg", "jpeg"}:
            normalized = "jpeg"
        if normalized not in {"png", "jpeg"}:
            raise argparse.ArgumentTypeError("Formats must be png, jpg, or jpeg.")
        if normalized not in formats:
            formats.append(normalized)
    if not formats:
        raise argparse.ArgumentTypeError("Expected at least one output format.")
    return tuple(formats)


def _parse_targets(raw: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("Expected at least one export target.")
    allowed = {"native", "50%", "1920w"}
    invalid = sorted(set(targets) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown targets: {', '.join(invalid)}. "
            "Use native, 50%, or 1920w."
        )
    return targets


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


def _median_time(function: Callable[[], Any], iterations: int) -> float:
    return _median_wall_cpu_time(function, iterations)[0]


def _peak_rss_mib() -> float | None:
    """Return process high-water RSS when the platform exposes it."""
    if resource is None:
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _render_settings(show_contours: bool) -> RenderSettings:
    return RenderSettings(
        vmin=80.0,
        vmax=100.0,
        show_contours=show_contours,
        contour_levels=compute_contour_levels(100.0, 0.0, 2.0),
        contour_step=2.0,
        reference_label="center",
    )


def _target_dimensions(loaded: LoadedFlat, target: str) -> _Target:
    native = compute_export_dimensions(
        loaded.metadata.nx,
        loaded.metadata.ny,
        loaded.metadata.nx,
    )
    if target == "native":
        dimensions = native
    elif target == "50%":
        dimensions = compute_export_dimensions(
            loaded.metadata.nx,
            loaded.metadata.ny,
            round(loaded.metadata.nx * 0.5),
        )
    else:
        dimensions = compute_export_dimensions(
            loaded.metadata.nx,
            loaded.metadata.ny,
            1920,
        )
    return _Target(target, dimensions.width, dimensions.height)


def _encode_to_bytes(
    figure: Any,
    fmt: str,
    jpg_quality: int,
) -> bytes:
    buffer = BytesIO()
    if fmt == "jpeg":
        figure.savefig(
            buffer,
            format="jpeg",
            dpi=EXPORT_DPI,
            pad_inches=0,
            pil_kwargs={"quality": jpg_quality},
        )
    else:
        figure.savefig(
            buffer,
            format="png",
            dpi=EXPORT_DPI,
            pad_inches=0,
        )
    return buffer.getvalue()


def _decode_metadata(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        return image.size, image.mode


def _benchmark_render_stages(
    illumination: np.ndarray,
    target: _Target,
    settings: RenderSettings,
    fmt: str,
    iterations: int,
    output_directory: Path,
    jpg_quality: int,
) -> dict[str, Any]:
    """Measure resampling through atomic filesystem commit for one output."""
    resample_wall, resample_cpu = _median_wall_cpu_time(
        lambda: resample_for_export(
            illumination,
            target.width,
            target.height,
        ),
        iterations,
    )
    export_data = resample_for_export(
        illumination,
        target.width,
        target.height,
    )

    figure_wall, figure_cpu = _median_wall_cpu_time(
        lambda: _build_and_clear(export_data, settings, target),
        iterations,
    )

    def draw_figure() -> None:
        figure = build_export_figure(
            export_data,
            settings,
            target.width,
            target.height,
        )
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        figure.clear()

    draw_wall, draw_cpu = _median_wall_cpu_time(draw_figure, iterations)

    def encode() -> bytes:
        figure = build_export_figure(
            export_data,
            settings,
            target.width,
            target.height,
        )
        FigureCanvasAgg(figure)
        try:
            return _encode_to_bytes(figure, fmt, jpg_quality)
        finally:
            figure.clear()

    encode_wall, encode_cpu = _median_wall_cpu_time(encode, iterations)
    encoded = encode()

    final_path = output_directory / f"{target.name}-{fmt}.{'jpg' if fmt == 'jpeg' else 'png'}"
    temp_path = output_directory / f".{final_path.name}.tmp"
    temp_path.unlink(missing_ok=True)

    def write_and_replace() -> None:
        temp_path.write_bytes(encoded)
        os.replace(temp_path, final_path)

    write_wall, write_cpu = _median_wall_cpu_time(
        write_and_replace,
        iterations,
    )

    render_temp_path = output_directory / f".{final_path.name}.render.tmp"

    def render_and_replace() -> None:
        render_temp_path.unlink(missing_ok=True)
        figure = build_export_figure(
            export_data,
            settings,
            target.width,
            target.height,
        )
        save_figure(
            figure,
            render_temp_path,
            fmt=fmt,
            jpg_quality=jpg_quality,
        )
        os.replace(render_temp_path, final_path)

    render_save_wall, render_save_cpu = _median_wall_cpu_time(
        render_and_replace,
        iterations,
    )
    dimensions, mode = _decode_metadata(final_path)

    return {
        "resample_s": resample_wall,
        "resample_cpu_s": resample_cpu,
        "figure_build_s": figure_wall,
        "figure_build_cpu_s": figure_cpu,
        "agg_draw_s": draw_wall,
        "agg_draw_cpu_s": draw_cpu,
        "encode_s": encode_wall,
        "encode_cpu_s": encode_cpu,
        "filesystem_write_s": write_wall,
        "filesystem_write_cpu_s": write_cpu,
        "render_save_s": render_save_wall,
        "render_save_cpu_s": render_save_cpu,
        "output_bytes": final_path.stat().st_size,
        "output_dimensions": list(dimensions),
        "output_mode": mode,
        "output_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "output_path": str(final_path),
    }


def _build_and_clear(
    export_data: np.ndarray,
    settings: RenderSettings,
    target: _Target,
) -> None:
    figure = build_export_figure(
        export_data,
        settings,
        target.width,
        target.height,
    )
    figure.clear()


def _benchmark_target(
    loaded: LoadedFlat,
    target: _Target,
    workers: Iterable[int],
    contour_modes: Iterable[bool],
    formats: Iterable[str],
    iterations: int,
    output_directory: Path,
    jpg_quality: int,
) -> list[dict[str, Any]]:
    """Measure cold and cached full-export paths for one target."""
    rows: list[dict[str, Any]] = []
    serial_reference: np.ndarray | None = None
    analysis_sigma = 100.0 / 2.0 if loaded.metadata.bayer_reduced else 100.0

    for worker_count in workers:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="benchmark-export",
        ) as filter_executor:
            map_wall, map_cpu = _median_wall_cpu_time(
                lambda: compute_illumination_map(
                    loaded.image,
                    sigma=analysis_sigma,
                    workers=worker_count,
                    executor=filter_executor,
                ),
                iterations,
            )
            computed = compute_export_result(
                loaded.image,
                analysis_sigma,
                target.width,
                target.height,
                worker_count=worker_count,
                executor=filter_executor,
            )
            illumination = computed.illumination
            if serial_reference is None:
                serial_reference = illumination.copy()

        difference = np.abs(illumination - serial_reference)
        finite_difference = difference[np.isfinite(difference)]
        map_difference = (
            float(np.max(finite_difference))
            if finite_difference.size
            else 0.0
        )
        map_digest = _sha256_array(illumination)
        reference_digest = _sha256_array(serial_reference)

        for cache_hit in (False, True):
            for show_contours in contour_modes:
                settings = _render_settings(show_contours)
                for fmt in formats:
                    stages = _benchmark_render_stages(
                        illumination,
                        target,
                        settings,
                        fmt,
                        iterations,
                        output_directory,
                        jpg_quality,
                    )
                    gc.collect()
                    rows.append(
                        {
                            "target": target.name,
                            "shape": list(illumination.shape),
                            "export_dimensions": [target.width, target.height],
                            "worker_count": worker_count,
                            "cache_hit": cache_hit,
                            "contours": show_contours,
                            "format": fmt,
                            "analysis_sigma": analysis_sigma,
                            "illumination_map_s": 0.0 if cache_hit else map_wall,
                            "illumination_map_cpu_s": 0.0 if cache_hit else map_cpu,
                            "map_sha256": map_digest,
                            "serial_reference_map_sha256": reference_digest,
                            "map_max_abs_difference": map_difference,
                            "peak_rss_mib": _peak_rss_mib(),
                            **stages,
                        }
                    )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="FITS file to benchmark.")
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=("native", "50%", "1920w"),
        help="Comma-separated targets: native, 50%%, or 1920w.",
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
        "--formats",
        type=_parse_formats,
        default=DEFAULT_FORMATS,
        help="Comma-separated output formats: png, jpg, or jpeg.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of timed iterations after one warm-up.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality passed unchanged to Pillow.",
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
    if not 1 <= args.jpg_quality <= 100:
        raise SystemExit("--jpg-quality must be between 1 and 100")
    if not args.path.is_file():
        raise SystemExit(f"FITS file not found: {args.path}")

    gc.collect()
    load_started = time.perf_counter()
    loaded = load_flat(args.path)
    load_seconds = time.perf_counter() - load_started
    targets = tuple(_target_dimensions(loaded, target) for target in args.targets)

    with tempfile.TemporaryDirectory(prefix="astroflat-export-benchmark-") as raw_dir:
        output_directory = Path(raw_dir)
        rows: list[dict[str, Any]] = []
        for target in targets:
            rows.extend(
                _benchmark_target(
                    loaded,
                    target,
                    args.workers,
                    args.contours,
                    args.formats,
                    args.iterations,
                    output_directory,
                    args.jpg_quality,
                )
            )

    report = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
        "cpu_count": os.cpu_count(),
        "path": str(args.path),
        "source_shape": list(loaded.image.shape),
        "source_metadata_dimensions": [loaded.metadata.nx, loaded.metadata.ny],
        "source_bayer_reduced": loaded.metadata.bayer_reduced,
        "load_s": load_seconds,
        "jpg_quality": args.jpg_quality,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Full export benchmark: {args.path} · source {loaded.image.shape} · "
            f"metadata {loaded.metadata.nx}x{loaded.metadata.ny} · "
            f"CPU count {os.cpu_count()}"
        )
        print(
            f"Versions: Python {report['python']}, NumPy {report['numpy']}, "
            f"SciPy {report['scipy']}, Matplotlib {report['matplotlib']}, "
            f"Pillow {report['pillow']}"
        )
        print(f"Load: {load_seconds:.6f}s")
        for row in rows:
            print(
                f"{row['target']:7s} "
                f"{row['format']:5s} "
                f"contours={'on' if row['contours'] else 'off'} "
                f"cache={'hit' if row['cache_hit'] else 'cold'} "
                f"workers={row['worker_count']:2d} "
                f"map={row['illumination_map_s']:.6f}s "
                f"resample={row['resample_s']:.6f}s "
                f"figure={row['figure_build_s']:.6f}s "
                f"draw={row['agg_draw_s']:.6f}s "
                f"encode={row['encode_s']:.6f}s "
                f"write={row['filesystem_write_s']:.6f}s "
                f"size={row['output_bytes']} "
                f"dims={tuple(row['output_dimensions'])} "
                f"RSS={row['peak_rss_mib']!s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
