"""Main application window for AstroFlatAnalyzer."""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import math
import os
from queue import Empty, Queue
import tkinter as tk
from pathlib import Path
import tempfile
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import matplotlib
from tkinterdnd2 import COPY, DND_FILES, TkinterDnD

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from flat_analyzer.fits_loader import FitsLoadError, LoadedFlat, load_flat
from flat_analyzer.illumination import (
    CornerFalloffStats,
    compute_contour_levels,
    compute_corner_falloff,
    compute_export_dimensions,
    reduce_for_preview,
    resample_for_export,
)
from flat_analyzer.processing import (
    PreviewResult,
    compute_export_result,
    compute_preview_result,
    preview_worker_capacity,
)
from flat_analyzer.visualization import (
    RenderSettings,
    build_export_figure,
    build_preview_figure,
    MATPLOTLIB_RENDER_LOCK,
    save_figure,
)

FITS_FILETYPES = [
    ("FITS files", "*.fits *.fit *.fts"),
    ("All files", "*.*"),
]

SLIDER_PREVIEW_DELAY_MS = 1000
PREVIEW_MAX_DIMENSION = 1200
WORKER_POLL_INTERVAL_MS = 50


@dataclass(frozen=True)
class _PreviewRequest:
    """Snapshot of a preview computation request."""

    request_id: int
    generation: int
    key: tuple[int, float]
    image: np.ndarray
    sigma: float


@dataclass(frozen=True)
class _ExportRequest:
    """Snapshot of an export computation request."""

    generation: int
    key: tuple[int, float]
    image: np.ndarray
    sigma: float
    export_width: int
    export_height: int
    path: str
    fmt: str
    jpg_quality: int
    settings: RenderSettings


@dataclass(frozen=True)
class _ExportJobResult:
    """Worker result containing a staged file and optional new map cache."""

    illumination: np.ndarray | None
    temporary_path: str


class _ExportJobError(Exception):
    """Export failure that can preserve a newly computed map cache."""

    def __init__(
        self,
        cause: Exception,
        illumination: np.ndarray | None,
        temporary_path: str,
    ) -> None:
        self.cause = cause
        self.illumination = illumination
        self.temporary_path = temporary_path
        super().__init__(str(cause))


def _remove_export_file(path: str | Path | None) -> None:
    """Best-effort cleanup for a staged export file."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _create_export_temp_path(path: str | Path) -> Path:
    """Create a closed temporary sibling path for atomic final replacement."""
    final_path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=final_path.suffix,
        dir=final_path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _run_export_job(
    image: np.ndarray | None,
    sigma: float | None,
    export_width: int,
    export_height: int,
    settings: RenderSettings,
    temporary_path: str,
    fmt: str,
    jpg_quality: int,
    cached_illumination: np.ndarray | None,
    filter_executor: Executor | None,
) -> _ExportJobResult:
    """Prepare, render, and stage an export without touching Tk."""
    illumination_to_cache: np.ndarray | None = None
    try:
        if cached_illumination is None:
            if image is None or sigma is None:
                raise ValueError("An uncached export requires an image and sigma.")
            computed = compute_export_result(
                image,
                sigma,
                export_width,
                export_height,
                worker_count=preview_worker_capacity(),
                executor=filter_executor,
            )
            illumination_to_cache = computed.illumination
            export_data = computed.export_data
        else:
            export_data = resample_for_export(
                cached_illumination,
                export_width,
                export_height,
            )

        with MATPLOTLIB_RENDER_LOCK:
            figure = build_export_figure(
                export_data,
                settings,
                export_width,
                export_height,
            )
            save_figure(
                figure,
                temporary_path,
                fmt=fmt,
                jpg_quality=jpg_quality,
            )
    except Exception as exc:
        _remove_export_file(temporary_path)
        raise _ExportJobError(
            exc,
            illumination_to_cache,
            temporary_path,
        ) from exc

    return _ExportJobResult(
        illumination=illumination_to_cache,
        temporary_path=temporary_path,
    )


class MainWindow(ctk.CTk):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("AstroFlatAnalyzer")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self._loaded: LoadedFlat | None = None
        self._corner_falloff: CornerFalloffStats | None = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._last_valid_export_width: int = 0
        self._pending_preview_after: str | None = None
        self._preview_image: np.ndarray | None = None
        self._preview_reduction_factor = 1
        self._image_generation = 0
        self._preview_cache: dict[tuple[int, float], PreviewResult] = {}
        self._full_illumination_cache_key: tuple[int, float] | None = None
        self._full_illumination_cache: np.ndarray | None = None
        self._preview_request_id = 0
        self._latest_preview_request_id = 0
        self._preview_pending_request: _PreviewRequest | None = None
        self._preview_inflight_request: _PreviewRequest | None = None
        self._preview_future: Future[Any] | None = None
        self._export_request: _ExportRequest | None = None
        self._export_future: Future[Any] | None = None
        self._export_temp_path: Path | None = None
        self._worker_results: Queue[tuple[str, Future[Any]]] = Queue()
        self._worker_poll_after: str | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="analysis"
        )
        self._preview_filter_executor = ThreadPoolExecutor(
            max_workers=preview_worker_capacity(),
            thread_name_prefix="preview-filter",
        )
        self._progress_window: ctk.CTkToplevel | None = None
        self._progress_label: ctk.CTkLabel | None = None
        self._progress_bar: ctk.CTkProgressBar | None = None
        self._closing = False

        self._build_layout()
        self._setup_drag_drop()
        self._bind_events()
        self._update_export_height_label()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_worker_poll()

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Left control panel ---
        panel = ctk.CTkScrollableFrame(self, width=280, label_text="Controls")
        panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        ctk.CTkButton(panel, text="Open FITS…", command=self._open_file).pack(
            fill="x", padx=8, pady=(8, 4)
        )
        self._path_label = ctk.CTkLabel(
            panel, text="No file loaded", wraplength=240, justify="left"
        )
        self._path_label.pack(fill="x", padx=8, pady=(0, 8))

        ctk.CTkLabel(panel, text="Smoothing σ (pixels)").pack(anchor="w", padx=8)
        self._sigma_slider = ctk.CTkSlider(panel, from_=25, to=200, number_of_steps=175)
        self._sigma_slider.set(100)
        self._sigma_slider.pack(fill="x", padx=8, pady=(0, 2))
        sigma_value_frame = ctk.CTkFrame(panel, fg_color="transparent")
        sigma_value_frame.pack(fill="x", padx=8, pady=(0, 8))
        self._sigma_entry = ctk.CTkEntry(sigma_value_frame, width=80)
        self._sigma_entry.insert(0, "100.0")
        self._sigma_entry.pack(side="left")
        ctk.CTkLabel(sigma_value_frame, text="px").pack(side="left", padx=(4, 0))

        ctk.CTkLabel(panel, text="Display range (%)").pack(anchor="w", padx=8)
        range_frame = ctk.CTkFrame(panel, fg_color="transparent")
        range_frame.pack(fill="x", padx=8, pady=(0, 8))
        self._vmin_entry = ctk.CTkEntry(range_frame, width=80, placeholder_text="Min")
        self._vmin_entry.insert(0, "80")
        self._vmin_entry.pack(side="left", padx=(0, 4))
        self._vmax_entry = ctk.CTkEntry(range_frame, width=80, placeholder_text="Max")
        self._vmax_entry.insert(0, "100")
        self._vmax_entry.pack(side="left")

        self._contour_check = ctk.CTkCheckBox(panel, text="Show contours")
        self._contour_check.select()
        self._contour_check.pack(anchor="w", padx=8, pady=(4, 0))

        ctk.CTkLabel(panel, text="Contour step (%)").pack(anchor="w", padx=8)
        self._contour_step_slider = ctk.CTkSlider(panel, from_=1, to=10, number_of_steps=9)
        self._contour_step_slider.set(2)
        self._contour_step_slider.pack(fill="x", padx=8, pady=(0, 2))
        contour_step_value_frame = ctk.CTkFrame(panel, fg_color="transparent")
        contour_step_value_frame.pack(fill="x", padx=8, pady=(0, 8))
        self._contour_step_entry = ctk.CTkEntry(contour_step_value_frame, width=80)
        self._contour_step_entry.insert(0, "2")
        self._contour_step_entry.pack(side="left")
        ctk.CTkLabel(contour_step_value_frame, text="%").pack(
            side="left", padx=(4, 0)
        )

        ctk.CTkLabel(panel, text="Export resolution").pack(anchor="w", padx=8)
        self._preset_var = ctk.StringVar(value="Native")
        self._preset_seg = ctk.CTkSegmentedButton(
            panel,
            values=["Native", "50%", "1920w"],
            variable=self._preset_var,
            command=self._apply_resolution_preset,
        )
        self._preset_seg.pack(fill="x", padx=8, pady=(0, 4))

        export_frame = ctk.CTkFrame(panel, fg_color="transparent")
        export_frame.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(export_frame, text="Width:").pack(side="left")
        self._export_width_entry = ctk.CTkEntry(export_frame, width=80)
        self._export_width_entry.insert(0, "0")
        self._export_width_entry.pack(side="left", padx=4)
        ctk.CTkLabel(export_frame, text="px").pack(side="left")

        self._export_dims_label = ctk.CTkLabel(panel, text="→ — × — px")
        self._export_dims_label.pack(anchor="w", padx=8, pady=(0, 8))

        ctk.CTkLabel(panel, text="Export format").pack(anchor="w", padx=8)
        self._format_var = ctk.StringVar(value="PNG")
        self._format_menu = ctk.CTkOptionMenu(
            panel, values=["PNG", "JPG"], variable=self._format_var
        )
        self._format_menu.pack(fill="x", padx=8, pady=(0, 4))

        self._jpg_quality_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self._jpg_quality_label = ctk.CTkLabel(self._jpg_quality_frame, text="JPG quality")
        self._jpg_quality_label.pack(anchor="w")
        self._jpg_quality_slider = ctk.CTkSlider(
            self._jpg_quality_frame, from_=85, to=100, number_of_steps=15
        )
        self._jpg_quality_slider.set(95)
        self._jpg_quality_slider.pack(fill="x", pady=(0, 2))
        jpg_quality_value_frame = ctk.CTkFrame(
            self._jpg_quality_frame, fg_color="transparent"
        )
        jpg_quality_value_frame.pack(fill="x", pady=(0, 8))
        self._jpg_quality_entry = ctk.CTkEntry(jpg_quality_value_frame, width=80)
        self._jpg_quality_entry.insert(0, "95")
        self._jpg_quality_entry.pack(side="left")
        ctk.CTkLabel(jpg_quality_value_frame, text="%").pack(
            side="left", padx=(4, 0)
        )

        ctk.CTkButton(panel, text="Save image…", command=self._save_image).pack(
            fill="x", padx=8, pady=(12, 8)
        )

        # --- Right preview panel ---
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._stats_label = ctk.CTkLabel(
            right,
            text="Load a FITS flat frame to begin.",
            justify="left",
            anchor="w",
        )
        self._stats_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        self._preview_frame = ctk.CTkFrame(right)
        self._preview_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._preview_frame.grid_rowconfigure(0, weight=1)
        self._preview_frame.grid_columnconfigure(0, weight=1)

        self._placeholder = ctk.CTkLabel(
            self._preview_frame,
            text="Preview will appear here\n\nDrag & drop a .fits file",
            font=ctk.CTkFont(size=14),
        )
        self._placeholder.grid(row=0, column=0)

    def _setup_drag_drop(self) -> None:
        # CustomTkinter owns the Tk root, so initialize TkDnD on the existing
        # root rather than replacing it with TkinterDnD.Tk. TkDnD methods are
        # added to descendant widgets, not to tkinter.Tk itself.
        try:
            TkinterDnD.require(self)
            registered = 0
            for widget in self._iter_descendant_widgets(self):
                if not hasattr(widget, "drop_target_register"):
                    continue
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
                registered += 1
        except (tk.TclError, RuntimeError) as exc:
            raise RuntimeError(
                "Drag-and-drop initialization failed. "
                "The bundled TkDnD library could not be loaded."
            ) from exc

        if registered == 0:
            raise RuntimeError("Drag-and-drop could not find a Tk widget target.")

    @staticmethod
    def _iter_descendant_widgets(widget: tk.Misc):
        """Yield a widget and all Tk descendants that can accept drops."""
        for child in widget.winfo_children():
            yield child
            yield from MainWindow._iter_descendant_widgets(child)

    def _bind_events(self) -> None:
        self._sigma_slider.configure(command=self._on_sigma_drag)
        self._sigma_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._contour_step_slider.configure(command=self._on_contour_step_drag)
        self._contour_step_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._jpg_quality_slider.configure(command=self._on_jpg_quality_drag)
        self._contour_check.configure(command=self._on_display_setting_change)
        self._format_menu.configure(command=self._on_format_change)

        for widget, handler in (
            (self._sigma_entry, self._on_sigma_entry_change),
            (self._contour_step_entry, self._on_contour_step_entry_change),
            (self._jpg_quality_entry, self._on_jpg_quality_entry_change),
        ):
            widget.bind("<Return>", handler)
            widget.bind("<FocusOut>", handler)

        for widget in (self._vmin_entry, self._vmax_entry):
            widget.bind("<Return>", self._on_display_setting_change)
            widget.bind("<FocusOut>", self._on_display_setting_change)

        self._export_width_entry.bind("<Return>", self._on_export_width_change)
        self._export_width_entry.bind("<FocusOut>", self._on_export_width_change)

    def _on_format_change(self, _: str) -> None:
        if self._format_var.get() == "JPG":
            self._jpg_quality_frame.pack(fill="x", padx=8, pady=(0, 8))
        else:
            self._jpg_quality_frame.pack_forget()

    def _on_display_setting_change(self, _event=None) -> None:
        self._refresh_preview()

    @staticmethod
    def _set_entry_value(entry: ctk.CTkEntry, value: str) -> None:
        if entry.get() == value:
            return
        entry.delete(0, "end")
        entry.insert(0, value)

    @staticmethod
    def _parse_slider_value(
        raw_value: str, minimum: float, maximum: float
    ) -> float | None:
        try:
            value = float(raw_value)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        value = float(round(value))
        return min(max(value, minimum), maximum)

    def _update_sigma_value(self, value: float) -> None:
        self._set_entry_value(self._sigma_entry, f"{value:.1f}")

    def _update_contour_step_value(self, value: float) -> None:
        self._set_entry_value(self._contour_step_entry, f"{int(value)}")

    def _update_jpg_quality_value(self, value: float) -> None:
        self._set_entry_value(self._jpg_quality_entry, f"{int(value)}")

    def _on_sigma_drag(self, value: float) -> None:
        self._update_sigma_value(value)
        self._schedule_preview_refresh()

    def _on_contour_step_drag(self, value: float) -> None:
        self._update_contour_step_value(value)
        self._schedule_preview_refresh()

    def _on_jpg_quality_drag(self, value: float) -> None:
        self._update_jpg_quality_value(value)

    def _on_slider_release(self, _event: tk.Event | None = None) -> None:
        """Recompute the preview immediately after a slider is released."""
        self._cancel_scheduled_preview()
        self._update_sigma_value(self._sigma_slider.get())
        self._update_contour_step_value(self._contour_step_slider.get())
        self._refresh_preview()

    def _schedule_preview_refresh(self) -> None:
        self._cancel_scheduled_preview()
        self._pending_preview_after = self.after(
            SLIDER_PREVIEW_DELAY_MS, self._run_scheduled_preview_refresh
        )

    def _cancel_scheduled_preview(self) -> None:
        if self._pending_preview_after is None:
            return
        try:
            self.after_cancel(self._pending_preview_after)
        except tk.TclError:
            pass
        self._pending_preview_after = None

    def _run_scheduled_preview_refresh(self) -> None:
        self._pending_preview_after = None
        self._refresh_preview()

    def _on_sigma_entry_change(self, _event=None) -> None:
        value = self._parse_slider_value(
            self._sigma_entry.get(), minimum=25, maximum=200
        )
        if value is None:
            self._update_sigma_value(self._sigma_slider.get())
            return
        self._sigma_slider.set(value)
        self._update_sigma_value(self._sigma_slider.get())
        self._cancel_scheduled_preview()
        self._refresh_preview()

    def _on_contour_step_entry_change(self, _event=None) -> None:
        value = self._parse_slider_value(
            self._contour_step_entry.get(), minimum=1, maximum=10
        )
        if value is None:
            self._update_contour_step_value(self._contour_step_slider.get())
            return
        self._contour_step_slider.set(value)
        self._update_contour_step_value(self._contour_step_slider.get())
        self._cancel_scheduled_preview()
        self._refresh_preview()

    def _on_jpg_quality_entry_change(self, _event=None) -> None:
        value = self._parse_slider_value(
            self._jpg_quality_entry.get(), minimum=85, maximum=100
        )
        if value is None:
            self._update_jpg_quality_value(self._jpg_quality_slider.get())
            return
        self._jpg_quality_slider.set(value)
        self._update_jpg_quality_value(self._jpg_quality_slider.get())

    def _on_drop(self, event: tk.Event) -> str:
        path = self._parse_drop_paths(str(event.data))
        if path:
            self._load_path(path)
        return COPY

    @staticmethod
    def _parse_drop_paths(data: str) -> Path | None:
        paths = []
        current = ""
        in_brace = False
        for char in data:
            if char == "{":
                in_brace = True
            elif char == "}":
                in_brace = False
            elif char == " " and not in_brace:
                if current:
                    paths.append(current)
                    current = ""
            else:
                current += char
        if current:
            paths.append(current)

        for p in paths:
            path = Path(p)
            if path.suffix.lower() in {".fits", ".fit", ".fts"}:
                return path
        return Path(paths[0]) if paths else None

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=FITS_FILETYPES)
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            loaded = load_flat(path)
        except FitsLoadError as exc:
            messagebox.showerror("Load error", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Load error", f"Unexpected error:\n{exc}")
            return

        self._image_generation += 1
        self._cancel_scheduled_preview()
        self._preview_pending_request = None
        self._preview_cache.clear()
        self._full_illumination_cache_key = None
        self._full_illumination_cache = None
        if self._export_request is not None:
            self._close_progress()
        self._cleanup_export_temp_path()
        self._export_request = None
        self._loaded = loaded
        try:
            self._corner_falloff = compute_corner_falloff(loaded.image)
        except ValueError:
            self._corner_falloff = None
        self._preview_image, self._preview_reduction_factor = reduce_for_preview(
            loaded.image, max_dimension=PREVIEW_MAX_DIMENSION
        )
        self._path_label.configure(text=str(loaded.metadata.path.name))
        self._last_valid_export_width = loaded.metadata.nx
        self._export_width_entry.delete(0, "end")
        self._export_width_entry.insert(0, str(loaded.metadata.nx))
        self._preset_var.set("Native")
        self._update_export_height_label()
        self._refresh_preview()

    def _get_render_settings(self) -> RenderSettings | None:
        try:
            vmin = float(self._vmin_entry.get())
            vmax = float(self._vmax_entry.get())
            contour_step = float(self._contour_step_slider.get())
        except ValueError:
            return None

        if vmin >= vmax:
            return None

        levels = compute_contour_levels(vmax, 0.0, contour_step)

        return RenderSettings(
            vmin=vmin,
            vmax=vmax,
            show_contours=self._contour_check.get() == 1,
            contour_levels=levels,
            contour_step=contour_step,
            reference_label="center",
        )

    def _get_analysis_sigma(self) -> float:
        """Return the smoothing sigma in analysis-image pixels."""
        sigma = float(self._sigma_slider.get())
        if self._loaded is not None and self._loaded.metadata.bayer_reduced:
            # Slider is expressed in original sensor pixels; the Bayer
            # analysis image is half-sized in each dimension.
            sigma /= 2.0
        return sigma

    def _refresh_preview(self) -> None:
        """Request a preview refresh without blocking the Tk event loop."""
        if self._loaded is None or self._closing:
            return

        settings = self._get_render_settings()
        if settings is None:
            return
        if self._preview_image is None:
            self._preview_image, self._preview_reduction_factor = reduce_for_preview(
                self._loaded.image, max_dimension=PREVIEW_MAX_DIMENSION
            )

        sigma = self._get_analysis_sigma() / self._preview_reduction_factor
        key = (self._image_generation, float(self._sigma_slider.get()))
        self._preview_request_id += 1
        self._latest_preview_request_id = self._preview_request_id

        cached = self._preview_cache.get(key)
        if cached is not None:
            self._preview_pending_request = None
            self._render_preview(cached, settings)
            return

        self._preview_pending_request = _PreviewRequest(
            request_id=self._preview_request_id,
            generation=self._image_generation,
            key=key,
            image=self._preview_image,
            sigma=sigma,
        )
        self._submit_pending_preview()

    def _render_preview(
        self,
        result: PreviewResult,
        settings: RenderSettings,
    ) -> None:
        self._update_stats(result.stats, self._corner_falloff)

        try:
            with MATPLOTLIB_RENDER_LOCK:
                fig = build_preview_figure(result.illumination, settings)
                self._show_figure(fig)
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def _submit_pending_preview(self) -> None:
        if self._closing or self._preview_pending_request is None:
            return

        request = self._preview_pending_request
        cached = self._preview_cache.get(request.key)
        if cached is not None:
            self._preview_pending_request = None
            if request.request_id == self._latest_preview_request_id:
                settings = self._get_render_settings()
                if settings is not None:
                    self._render_preview(cached, settings)
            return

        if self._preview_future is not None:
            return

        self._preview_pending_request = None
        self._preview_inflight_request = request
        filter_executor = self.__dict__.get("_preview_filter_executor")
        preview_task = partial(
            compute_preview_result,
            request.image,
            request.sigma,
            executor=filter_executor,
        )
        future = self._executor.submit(
            preview_task,
        )
        self._preview_future = future
        self._track_worker_future("preview", future)

    def _track_worker_future(self, kind: str, future: Future[Any]) -> None:
        def handle_completion(completed: Future[Any]) -> None:
            if kind == "export" and self.__dict__.get("_closing", False):
                self._cleanup_export_future(completed)
            self._worker_results.put((kind, completed))

        future.add_done_callback(
            handle_completion
        )

    def _schedule_worker_poll(self) -> None:
        if self._closing or self._worker_poll_after is not None:
            return
        self._worker_poll_after = self.after(
            WORKER_POLL_INTERVAL_MS, self._poll_worker_results
        )

    def _poll_worker_results(self) -> None:
        self._worker_poll_after = None
        if self._closing:
            return

        while True:
            try:
                kind, future = self._worker_results.get_nowait()
            except Empty:
                break
            if kind == "preview":
                self._handle_preview_result(future)
            elif kind == "export":
                self._handle_export_result(future)

        self._schedule_worker_poll()

    def _handle_preview_result(self, future: Future[Any]) -> None:
        request = self._preview_inflight_request
        self._preview_inflight_request = None
        self._preview_future = None
        if request is None:
            self._submit_pending_preview()
            return

        try:
            result = future.result()
        except Exception as exc:
            if (
                request.generation == self._image_generation
                and request.request_id == self._latest_preview_request_id
            ):
                messagebox.showerror("Preview error", str(exc))
        else:
            if not isinstance(result, PreviewResult):
                if (
                    request.generation == self._image_generation
                    and request.request_id == self._latest_preview_request_id
                ):
                    messagebox.showerror(
                        "Preview error",
                        "Preview worker returned an invalid result.",
                    )
            elif request.generation == self._image_generation:
                self._preview_cache[request.key] = result
                while len(self._preview_cache) > 2:
                    self._preview_cache.pop(next(iter(self._preview_cache)))

                if request.request_id == self._latest_preview_request_id:
                    settings = self._get_render_settings()
                    if settings is not None:
                        self._render_preview(result, settings)

        self._submit_pending_preview()

    def _update_stats(
        self,
        stats,
        corner_falloff: CornerFalloffStats | None,
    ) -> None:
        meta = self._loaded.metadata  # type: ignore[union-attr]
        if meta.bayer_reduced:
            bayer_note = (
                f"\nBayer pattern: {meta.bayerpat} · 2×2 cells averaged "
                f"to {meta.analysis_nx} × {meta.analysis_ny}"
            )
        elif meta.bayerpat:
            bayer_note = f"\nBayer pattern: {meta.bayerpat} (not reduced)"
        else:
            bayer_note = ""
        rgb_note = " (RGB → luminance)" if meta.is_rgb else ""
        if corner_falloff is not None:
            illumination_note = (
                f"\nIllumination:\n"
                f"Top Left {corner_falloff.top_left_pct:.1f}% · "
                f"Top Right {corner_falloff.top_right_pct:.1f}%\n"
                f"Bottom Left {corner_falloff.bottom_left_pct:.1f}% · "
                f"Bottom Right {corner_falloff.bottom_right_pct:.1f}%"
            )
        else:
            illumination_note = "\nIllumination: unavailable"
        self._stats_label.configure(
            text=(
                f"{meta.nx} × {meta.ny} px{rgb_note}\n"
                f"Illumination: min {stats.min_pct:.1f}%  ·  "
                f"median {stats.median_pct:.1f}%  ·  max {stats.max_pct:.1f}%\n"
                f"Center: {stats.center_pct:.1f}%  ·  "
                f"Corners (avg): {stats.corner_avg_pct:.1f}%"
                f"{bayer_note}"
                f"{illumination_note}"
            )
        )

    def _show_figure(self, fig: Figure) -> None:
        if self._placeholder.winfo_ismapped():
            self._placeholder.grid_forget()

        if self._canvas is None:
            self._canvas = FigureCanvasTkAgg(fig, master=self._preview_frame)
            widget = self._canvas.get_tk_widget()
            widget.grid(row=0, column=0, sticky="nsew")
            self._preview_frame.update_idletasks()
        else:
            old_figure = self._canvas.figure
            old_figure.clear()
            old_figure.set_canvas(None)
            self._canvas.figure = fig
            fig.set_canvas(self._canvas)

        widget = self._canvas.get_tk_widget()
        width = widget.winfo_width()
        height = widget.winfo_height()
        if width > 1 and height > 1:
            fig.set_size_inches(
                width / fig.dpi,
                height / fig.dpi,
                forward=False,
            )
        self._canvas.draw()

    def _get_export_dimensions(self) -> tuple[int, int] | None:
        if self._loaded is None:
            return None
        try:
            width = int(self._export_width_entry.get())
        except ValueError:
            width = self._last_valid_export_width

        dims = compute_export_dimensions(
            self._loaded.metadata.nx,
            self._loaded.metadata.ny,
            width,
        )
        return dims.width, dims.height

    def _update_export_height_label(self) -> None:
        dims = self._get_export_dimensions()
        if dims is None:
            self._export_dims_label.configure(text="→ — × — px")
            return
        w, h = dims
        self._export_dims_label.configure(text=f"→ {w} × {h} px")

    def _on_export_width_change(self, _event=None) -> None:
        if self._loaded is None:
            return
        try:
            width = int(self._export_width_entry.get())
        except ValueError:
            self._export_width_entry.delete(0, "end")
            self._export_width_entry.insert(0, str(self._last_valid_export_width))
            return

        dims = compute_export_dimensions(
            self._loaded.metadata.nx,
            self._loaded.metadata.ny,
            width,
        )
        if dims.width != width:
            self._export_width_entry.delete(0, "end")
            self._export_width_entry.insert(0, str(dims.width))

        self._last_valid_export_width = dims.width
        self._update_export_height_label()

    def _apply_resolution_preset(self, value: str) -> None:
        if self._loaded is None:
            return
        nx = self._loaded.metadata.nx
        if value == "Native":
            width = nx
        elif value == "50%":
            width = max(64, nx // 2)
        elif value == "1920w":
            width = min(1920, nx)
        else:
            return

        self._export_width_entry.delete(0, "end")
        self._export_width_entry.insert(0, str(width))
        self._last_valid_export_width = width
        self._update_export_height_label()

    def _cleanup_export_temp_path(self) -> None:
        """Remove the currently tracked staged export, if any."""
        temporary_path = self.__dict__.get("_export_temp_path")
        _remove_export_file(temporary_path)
        self.__dict__["_export_temp_path"] = None

    @staticmethod
    def _cleanup_export_future(future: Future[Any]) -> None:
        """Remove a staged file from a completed or failed export future."""
        try:
            result = future.result()
        except _ExportJobError as exc:
            _remove_export_file(exc.temporary_path)
        except Exception:
            return
        else:
            if isinstance(result, _ExportJobResult):
                _remove_export_file(result.temporary_path)

    def _save_image(self) -> None:
        if self._loaded is None:
            messagebox.showwarning("Save", "Load a FITS file first.")
            return
        if self._export_request is not None or self._export_future is not None:
            messagebox.showinfo("Save", "An image export is already in progress.")
            return

        settings = self._get_render_settings()
        dims = self._get_export_dimensions()

        if settings is None or dims is None:
            messagebox.showerror("Save", "Invalid display or export settings.")
            return

        fmt = self._format_var.get()
        ext = ".jpg" if fmt == "JPG" else ".png"
        filetypes = [("JPEG", "*.jpg"), ("All", "*.*")] if fmt == "JPG" else [
            ("PNG", "*.png"),
            ("All", "*.*"),
        ]

        default_name = self._loaded.metadata.path.stem + "_illumination" + ext
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialfile=default_name,
            filetypes=filetypes,
        )
        if not path:
            return

        export_w, export_h = dims
        try:
            temporary_path = _create_export_temp_path(path)
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            return

        request = _ExportRequest(
            generation=self._image_generation,
            key=(self._image_generation, float(self._sigma_slider.get())),
            image=self._loaded.image,
            sigma=self._get_analysis_sigma(),
            export_width=export_w,
            export_height=export_h,
            path=path,
            fmt=fmt.lower(),
            jpg_quality=int(self._jpg_quality_slider.get()),
            settings=settings,
        )
        self._export_request = request
        self._export_temp_path = temporary_path
        cached = (
            self._full_illumination_cache
            if self._full_illumination_cache_key == request.key
            else None
        )
        self._show_progress(
            "Rendering and saving image…" if cached is not None else "Preparing image…"
        )
        filter_executor = self.__dict__.get("_preview_filter_executor")
        export_task = partial(
            _run_export_job,
            request.image if cached is None else None,
            request.sigma if cached is None else None,
            request.export_width,
            request.export_height,
            request.settings,
            str(temporary_path),
            request.fmt,
            request.jpg_quality,
            cached,
            filter_executor,
        )
        try:
            future = self._executor.submit(export_task)
        except Exception as exc:
            self._cleanup_export_temp_path()
            self._show_save_error(exc)
            return

        self._export_future = future
        self._track_worker_future("export", future)

    def _handle_export_result(self, future: Future[Any]) -> None:
        request = self._export_request
        self._export_future = None
        if request is None:
            self._cleanup_export_future(future)
            return

        try:
            result = future.result()
        except _ExportJobError as exc:
            if request.generation != self._image_generation:
                self._cleanup_export_temp_path()
                self._close_progress()
                self._export_request = None
                return
            if exc.illumination is not None:
                self._full_illumination_cache_key = request.key
                self._full_illumination_cache = exc.illumination
            self._show_save_error(exc.cause)
            return
        except Exception as exc:
            if request.generation != self._image_generation:
                self._cleanup_export_temp_path()
                self._close_progress()
                self._export_request = None
                return
            self._show_save_error(exc)
            return

        if request.generation != self._image_generation:
            if isinstance(result, _ExportJobResult):
                _remove_export_file(result.temporary_path)
            else:
                self._cleanup_export_temp_path()
            self._close_progress()
            self._export_request = None
            return

        if not isinstance(result, _ExportJobResult):
            self._show_save_error(
                RuntimeError("Export worker returned an invalid result.")
            )
            return

        if result.illumination is not None:
            self._full_illumination_cache_key = request.key
            self._full_illumination_cache = result.illumination
        try:
            os.replace(result.temporary_path, request.path)
        except Exception as exc:
            _remove_export_file(result.temporary_path)
            self._show_save_error(exc)
            return

        self._export_temp_path = None
        self._close_progress()
        self._export_request = None

    def _show_progress(self, message: str) -> None:
        if self._progress_window is None:
            window = ctk.CTkToplevel(self)
            window.title("AstroFlatAnalyzer")
            window.geometry("320x120")
            window.resizable(False, False)
            window.transient(self)
            window.protocol("WM_DELETE_WINDOW", lambda: None)

            label = ctk.CTkLabel(window, text=message)
            label.pack(fill="x", padx=20, pady=(20, 8))
            progress = ctk.CTkProgressBar(window, mode="indeterminate")
            progress.pack(fill="x", padx=20, pady=(0, 20))

            self._progress_window = window
            self._progress_label = label
            self._progress_bar = progress
            window.grab_set()
        else:
            self._set_progress_message(message)

        if self._progress_bar is not None:
            self._progress_bar.start()
        if self._progress_window is not None:
            self._progress_window.update_idletasks()

    def _set_progress_message(self, message: str) -> None:
        if self._progress_label is not None:
            self._progress_label.configure(text=message)
        if self._progress_window is not None:
            self._progress_window.update_idletasks()

    def _close_progress(self) -> None:
        progress_bar = self.__dict__.get("_progress_bar")
        progress_window = self.__dict__.get("_progress_window")
        if progress_bar is not None:
            progress_bar.stop()
        if progress_window is not None:
            try:
                progress_window.grab_release()
            except tk.TclError:
                pass
            progress_window.destroy()
        self.__dict__["_progress_window"] = None
        self.__dict__["_progress_label"] = None
        self.__dict__["_progress_bar"] = None

    def _show_save_error(self, exc: Exception) -> None:
        self._cleanup_export_temp_path()
        self._close_progress()
        self._export_request = None
        messagebox.showerror("Save error", str(exc))

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._cancel_scheduled_preview()
        if self._worker_poll_after is not None:
            try:
                self.after_cancel(self._worker_poll_after)
            except tk.TclError:
                pass
            self._worker_poll_after = None
        self._cleanup_export_temp_path()
        self._close_progress()
        self._export_request = None
        preview_filter_executor = self.__dict__.get("_preview_filter_executor")
        if preview_filter_executor is not None:
            preview_filter_executor.shutdown(wait=False, cancel_futures=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


def run_app() -> None:
    app = MainWindow()
    app.mainloop()
