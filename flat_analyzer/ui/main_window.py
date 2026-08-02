"""Main application window for AstroFlatAnalyzer."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from flat_analyzer.fits_loader import FitsLoadError, LoadedFlat, load_flat
from flat_analyzer.illumination import (
    CornerFalloffStats,
    ReferenceMode,
    compute_contour_levels,
    compute_corner_falloff,
    compute_export_dimensions,
    compute_illumination_map,
    compute_stats,
    downsample_for_preview,
    resample_for_export,
)
from flat_analyzer.visualization import (
    RenderSettings,
    build_export_figure,
    build_preview_figure,
    save_figure,
)

FITS_FILETYPES = [
    ("FITS files", "*.fits *.fit *.fts"),
    ("All files", "*.*"),
]


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

        self._build_layout()
        self._setup_drag_drop()
        self._bind_events()
        self._update_export_height_label()

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
        self._sigma_label = ctk.CTkLabel(panel, text="100.0 px")
        self._sigma_label.pack(anchor="w", padx=8, pady=(0, 8))

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
        self._contour_step_label = ctk.CTkLabel(panel, text="2%")
        self._contour_step_label.pack(anchor="w", padx=8, pady=(0, 8))

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
        self._jpg_quality_slider.pack(fill="x", pady=(0, 8))

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
        try:
            self.drop_target_register("DND_Files")  # type: ignore[attr-defined]
            self.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]
        except (tk.TclError, AttributeError):
            pass

    def _bind_events(self) -> None:
        self._sigma_slider.configure(command=self._on_sigma_drag)
        self._sigma_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._contour_step_slider.configure(command=self._on_contour_step_drag)
        self._contour_step_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._contour_check.configure(command=lambda _: self._refresh_preview())
        self._format_menu.configure(command=self._on_format_change)

        for widget in (self._vmin_entry, self._vmax_entry):
            widget.bind("<Return>", lambda _: self._refresh_preview())
            widget.bind("<FocusOut>", lambda _: self._refresh_preview())

        self._export_width_entry.bind("<Return>", self._on_export_width_change)
        self._export_width_entry.bind("<FocusOut>", self._on_export_width_change)

    def _on_format_change(self, _: str) -> None:
        if self._format_var.get() == "JPG":
            self._jpg_quality_frame.pack(fill="x", padx=8, pady=(0, 8))
        else:
            self._jpg_quality_frame.pack_forget()

    def _on_sigma_drag(self, value: float) -> None:
        self._sigma_label.configure(text=f"{value:.1f} px")

    def _on_contour_step_drag(self, value: float) -> None:
        self._contour_step_label.configure(text=f"{int(value)}%")

    def _on_slider_release(self, _event: tk.Event | None = None) -> None:
        """Recompute preview after the user releases a slider."""
        self._on_sigma_drag(self._sigma_slider.get())
        self._on_contour_step_drag(self._contour_step_slider.get())
        self._refresh_preview()

    def _on_drop(self, event: tk.Event) -> None:
        path = self._parse_drop_paths(str(event.data))
        if path:
            self._load_path(path)

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

        self._loaded = loaded
        try:
            self._corner_falloff = compute_corner_falloff(loaded.image)
        except ValueError:
            self._corner_falloff = None
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

    def _compute_illumination(self) -> np.ndarray | None:
        if self._loaded is None:
            return None
        settings = self._get_render_settings()
        if settings is None:
            return None

        try:
            sigma = self._sigma_slider.get()
            if self._loaded.metadata.bayer_reduced:
                # Slider is expressed in original sensor pixels; the Bayer
                # analysis image is half-size in each dimension.
                sigma /= 2.0
            return compute_illumination_map(
                self._loaded.image,
                mode=ReferenceMode.CENTER,
                sigma=sigma,
            )
        except ValueError:
            return None

    def _refresh_preview(self) -> None:
        if self._loaded is None:
            return

        illumination = self._compute_illumination()
        settings = self._get_render_settings()
        if illumination is None or settings is None:
            return

        try:
            stats = compute_stats(illumination)
        except ValueError:
            return

        self._update_stats(stats, self._corner_falloff)
        preview_data = downsample_for_preview(illumination)

        try:
            fig = build_preview_figure(preview_data, settings)
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))
            return

        self._show_figure(fig)

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

        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

        self._canvas = FigureCanvasTkAgg(fig, master=self._preview_frame)
        widget = self._canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
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

    def _save_image(self) -> None:
        if self._loaded is None:
            messagebox.showwarning("Save", "Load a FITS file first.")
            return

        illumination = self._compute_illumination()
        settings = self._get_render_settings()
        dims = self._get_export_dimensions()

        if illumination is None or settings is None or dims is None:
            messagebox.showerror("Save", "Invalid display or export settings.")
            return

        export_w, export_h = dims
        export_data = resample_for_export(illumination, export_w, export_h)

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

        try:
            fig = build_export_figure(export_data, settings, export_w, export_h)
            save_figure(
                fig,
                path,
                fmt=fmt.lower(),
                jpg_quality=int(self._jpg_quality_slider.get()),
            )
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
            return


def run_app() -> None:
    app = MainWindow()
    app.mainloop()
