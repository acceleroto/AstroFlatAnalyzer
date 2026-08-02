"""Matplotlib rendering for preview and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from flat_analyzer.illumination import effective_contour_levels

COLORMAP = "gray"
EXPORT_DPI = 100
CONTOUR_COLOR = "#00e5ff"
CONTOUR_OUTLINE_COLOR = "#000000"


@dataclass(frozen=True)
class RenderSettings:
    """Visual settings shared by preview and export."""

    vmin: float
    vmax: float
    show_contours: bool
    contour_levels: np.ndarray
    contour_step: float = 2.0
    reference_label: str = "center"


def _label_fontsize(nx: int, ny: int, for_export: bool) -> float:
    short_side = min(nx, ny)
    if for_export:
        return max(32.0, short_side / 100.0)
    return max(9.0, short_side / 120.0)


def _contour_linewidth(nx: int, ny: int, for_export: bool, outline: bool = False) -> float:
    short_side = min(nx, ny)
    if for_export:
        base = max(2.5, short_side / 1800.0)
        return base * 2.2 if outline else base
    return 2.0 if outline else 1.0


def _draw_contours(
    ax: plt.Axes,
    illumination_pct: np.ndarray,
    settings: RenderSettings,
    extent: tuple[float, float, float, float],
    for_export: bool = False,
) -> None:
    """Draw contour lines aligned with the image extent."""
    ny, nx = illumination_pct.shape
    levels = effective_contour_levels(
        illumination_pct,
        settings.contour_levels,
        settings.contour_step,
    )
    levels = np.sort(np.unique(levels))
    if levels.size == 0:
        return

    fontsize = _label_fontsize(nx, ny, for_export)
    lw_outline = _contour_linewidth(nx, ny, for_export, outline=True)
    lw_main = _contour_linewidth(nx, ny, for_export, outline=False)

    contour_kwargs = dict(
        origin="upper",
        extent=extent,
        levels=levels,
    )

    ax.contour(
        illumination_pct,
        colors=CONTOUR_OUTLINE_COLOR,
        linewidths=lw_outline,
        alpha=0.95,
        zorder=5,
        **contour_kwargs,
    )
    cs = ax.contour(
        illumination_pct,
        colors=CONTOUR_COLOR,
        linewidths=lw_main,
        alpha=1.0,
        zorder=6,
        **contour_kwargs,
    )

    # inline=False keeps the contour line from crossing through label text
    labels = ax.clabel(
        cs,
        inline=False,
        fontsize=fontsize,
        fmt="%g%%",
        colors="black",
        inline_spacing=10,
    )
    stroke_w = max(2.0, fontsize / 5.0)
    for label in labels:
        label.set_path_effects(
            [
                pe.Stroke(linewidth=stroke_w, foreground="white"),
                pe.Normal(),
            ]
        )
        label.set_zorder(10)


def build_figure(
    illumination_pct: np.ndarray,
    settings: RenderSettings,
    figsize_inches: tuple[float, float],
    show_colorbar: bool = True,
    tight_layout: bool = True,
    for_export: bool = False,
) -> Figure:
    """Build a matplotlib figure for the illumination map."""
    ny, nx = illumination_pct.shape
    extent = (0.0, float(nx), float(ny), 0.0)

    if show_colorbar:
        fig = Figure(figsize=figsize_inches, dpi=EXPORT_DPI)
        ax = fig.add_axes([0.02, 0.02, 0.82, 0.96])
    else:
        fig = Figure(figsize=figsize_inches, dpi=EXPORT_DPI)
        ax = fig.add_axes([0, 0, 1, 1])

    im = ax.imshow(
        illumination_pct,
        cmap=COLORMAP,
        vmin=settings.vmin,
        vmax=settings.vmax,
        origin="upper",
        extent=extent,
        interpolation="bilinear",
        zorder=1,
    )

    if settings.show_contours:
        _draw_contours(ax, illumination_pct, settings, extent, for_export=for_export)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)

    if show_colorbar:
        cax = fig.add_axes([0.86, 0.05, 0.03, 0.9])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Illumination (% of {settings.reference_label})")
    elif not tight_layout:
        fig.subplots_adjust(0, 0, 1, 1)

    return fig


def build_preview_figure(
    illumination_pct: np.ndarray,
    settings: RenderSettings,
    max_width_px: int = 900,
    max_height_px: int = 700,
) -> Figure:
    """Build a preview-sized figure for embedding in the UI."""
    ny, nx = illumination_pct.shape
    scale = min(max_width_px / nx, max_height_px / ny)
    width_in = (nx * scale) / EXPORT_DPI
    height_in = (ny * scale) / EXPORT_DPI
    return build_figure(
        illumination_pct,
        settings,
        (width_in, height_in),
        show_colorbar=True,
        for_export=False,
    )


def build_export_figure(
    illumination_pct: np.ndarray,
    settings: RenderSettings,
    export_width: int,
    export_height: int,
) -> Figure:
    """Build an export figure at exact pixel dimensions."""
    width_in = export_width / EXPORT_DPI
    height_in = export_height / EXPORT_DPI
    return build_figure(
        illumination_pct,
        settings,
        (width_in, height_in),
        show_colorbar=False,
        tight_layout=False,
        for_export=True,
    )


def save_figure(
    fig: Figure,
    path: str | Path,
    fmt: str = "png",
    jpg_quality: int = 95,
) -> None:
    """Save figure to disk as PNG or JPEG."""
    path = Path(path)
    if fmt.lower() in ("jpg", "jpeg"):
        fig.savefig(
            path,
            format="jpeg",
            dpi=EXPORT_DPI,
            pad_inches=0,
            pil_kwargs={"quality": jpg_quality},
        )
    else:
        fig.savefig(path, format="png", dpi=EXPORT_DPI, pad_inches=0)
    plt.close(fig)
