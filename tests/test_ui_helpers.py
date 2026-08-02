"""Tests for desktop UI helper behavior that does not require a window."""

from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from flat_analyzer.processing import compute_preview_result
from flat_analyzer.ui.main_window import (
    _ExportJobError,
    MainWindow,
    _ExportJobResult,
    _ExportRequest,
    _run_export_job,
)
from flat_analyzer.visualization import RenderSettings


def test_parse_drop_paths_preserves_spaces_in_braced_paths():
    dropped = "{/Users/test data/flat frame.fits} {/Users/other.fit}"

    assert MainWindow._parse_drop_paths(dropped) == Path(
        "/Users/test data/flat frame.fits"
    )


def test_parse_drop_paths_accepts_unbraced_paths():
    assert MainWindow._parse_drop_paths("/tmp/flat.fits") == Path("/tmp/flat.fits")


def test_parse_slider_value_rounds_and_clamps():
    assert MainWindow._parse_slider_value("101.6", 25, 200) == 102
    assert MainWindow._parse_slider_value("10", 25, 200) == 25
    assert MainWindow._parse_slider_value("250", 25, 200) == 200


def test_parse_slider_value_rejects_non_finite_or_invalid_values():
    assert MainWindow._parse_slider_value("not a number", 1, 10) is None
    assert MainWindow._parse_slider_value("nan", 1, 10) is None


class _ValueWidget:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _ImmediateExecutor:
    def __init__(self):
        self.submit_count = 0

    def submit(self, function, *args):
        self.submit_count += 1
        future = Future()
        future.set_result(function(*args))
        return future


class _DeferredExecutor:
    def __init__(self, events=None):
        self.futures = []
        self.events = events

    def submit(self, function, *args):
        if self.events is not None:
            self.events.append("submit")
        future = Future()
        self.futures.append((future, function, args))
        return future


class _ShutdownRecorder:
    def __init__(self):
        self.calls = []

    def shutdown(self, *, wait, cancel_futures):
        self.calls.append((wait, cancel_futures))


def _make_preview_window(executor):
    image = np.ones((40, 60), dtype=float) * 1000
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    window = MainWindow.__new__(MainWindow)
    window._loaded = SimpleNamespace(
        image=image,
        metadata=SimpleNamespace(bayer_reduced=False),
    )
    window._closing = False
    window._image_generation = 1
    window._preview_image = image
    window._preview_reduction_factor = 1
    window._preview_cache = {}
    window._sigma_slider = _ValueWidget(3)
    window._preview_request_id = 0
    window._latest_preview_request_id = 0
    window._preview_pending_request = None
    window._preview_inflight_request = None
    window._preview_future = None
    window._worker_results = Queue()
    window._executor = executor
    window._get_render_settings = lambda: settings
    window.rendered_results = []
    window._render_preview = lambda result, _settings: (
        window.rendered_results.append(result)
    )
    return window


def test_preview_result_is_cached_for_display_only_refreshes():
    executor = _ImmediateExecutor()
    window = _make_preview_window(executor)

    window._refresh_preview()
    window._handle_preview_result(window._preview_future)
    window._refresh_preview()

    assert executor.submit_count == 1
    assert len(window.rendered_results) == 2


def test_stale_preview_result_is_discarded_before_new_request_runs():
    executor = _DeferredExecutor()
    window = _make_preview_window(executor)

    window._refresh_preview()
    first_future = window._preview_future
    window._sigma_slider.value = 4
    window._refresh_preview()

    first_future.set_result(compute_preview_result(window._preview_image, sigma=3))
    window._handle_preview_result(first_future)

    assert len(executor.futures) == 2
    assert window.rendered_results == []

    second_future = window._preview_future
    second_future.set_result(compute_preview_result(window._preview_image, sigma=4))
    window._handle_preview_result(second_future)

    assert len(window.rendered_results) == 1


def test_close_shuts_down_request_and_preview_filter_executors():
    request_executor = _ShutdownRecorder()
    filter_executor = _ShutdownRecorder()
    window = MainWindow.__new__(MainWindow)
    window._closing = False
    window._pending_preview_after = None
    window._worker_poll_after = None
    window._executor = request_executor
    window._preview_filter_executor = filter_executor
    window._close_progress = lambda: None
    window.destroyed = False
    window.destroy = lambda: setattr(window, "destroyed", True)

    window._on_close()

    assert request_executor.calls == [(False, True)]
    assert filter_executor.calls == [(False, True)]
    assert window.destroyed is True


def test_close_cleans_staged_export_for_active_request(tmp_path):
    staged_path = tmp_path / ".active.png.tmp"
    staged_path.write_bytes(b"active")
    request_executor = _ShutdownRecorder()
    filter_executor = _ShutdownRecorder()
    window = MainWindow.__new__(MainWindow)
    window._closing = False
    window._pending_preview_after = None
    window._worker_poll_after = None
    window._executor = request_executor
    window._preview_filter_executor = filter_executor
    window._export_request = object()
    window._export_temp_path = staged_path
    window._close_progress = lambda: None
    window.destroy = lambda: None

    window._on_close()

    assert not staged_path.exists()
    assert window._export_request is None


def test_save_dialog_opens_before_export_computation(monkeypatch):
    events = []
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    window = MainWindow.__new__(MainWindow)
    window._loaded = SimpleNamespace(
        image=np.ones((40, 60), dtype=float) * 1000,
        metadata=SimpleNamespace(
            path=Path("/tmp/example.fits"),
            bayer_reduced=False,
            nx=60,
            ny=40,
        ),
    )
    window._export_request = None
    window._export_future = None
    window._export_temp_path = None
    window._full_illumination_cache_key = None
    window._full_illumination_cache = None
    window._image_generation = 1
    window._get_render_settings = lambda: settings
    window._get_export_dimensions = lambda: (60, 40)
    window._get_analysis_sigma = lambda: 3.0
    window._sigma_slider = _ValueWidget(3)
    window._format_var = _ValueWidget("PNG")
    window._jpg_quality_slider = _ValueWidget(95)
    window._show_progress = lambda _message: events.append("progress")
    window._track_worker_future = lambda _kind, _future: None
    window._executor = _DeferredExecutor(events)

    def choose_path(**_kwargs):
        events.append("dialog")
        return "/tmp/output.png"

    monkeypatch.setattr(
        "flat_analyzer.ui.main_window.filedialog.asksaveasfilename",
        choose_path,
    )

    window._save_image()

    assert events.index("dialog") < events.index("submit")
    window._cleanup_export_temp_path()


def test_export_result_commits_staged_file_and_populates_full_resolution_cache(
    tmp_path,
):
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    image = np.ones((4, 6), dtype=float) * 1000
    request = _ExportRequest(
        generation=2,
        key=(2, 3.0),
        image=image,
        sigma=3.0,
        export_width=6,
        export_height=4,
        path=str(tmp_path / "output.png"),
        fmt="png",
        jpg_quality=95,
        settings=settings,
    )
    staged_path = tmp_path / "staged.png"
    staged_path.write_bytes(b"staged")
    result = _ExportJobResult(
        illumination=image,
        temporary_path=str(staged_path),
    )
    future = Future()
    future.set_result(result)

    window = MainWindow.__new__(MainWindow)
    window._export_request = request
    window._export_future = future
    window._image_generation = 2
    window._full_illumination_cache_key = None
    window._full_illumination_cache = None
    window._export_temp_path = staged_path
    window._progress_window = None
    window._progress_label = None
    window._progress_bar = None

    window._handle_export_result(future)

    assert window._full_illumination_cache_key == request.key
    assert window._full_illumination_cache is result.illumination
    assert Path(request.path).read_bytes() == b"staged"
    assert not staged_path.exists()
    assert window._export_request is None


def test_export_job_renders_to_staged_file_without_tk(tmp_path):
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    staged_path = tmp_path / ".output.png.tmp"
    result = _run_export_job(
        np.ones((40, 60), dtype=float) * 1000,
        0.0,
        60,
        40,
        settings,
        str(staged_path),
        "png",
        95,
        None,
        None,
    )

    assert isinstance(result, _ExportJobResult)
    assert result.illumination is not None
    assert staged_path.exists()
    assert staged_path.stat().st_size > 0


def test_cached_export_job_reuses_map_without_recomputing(monkeypatch, tmp_path):
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    cached = np.ones((40, 60), dtype=float) * 100
    staged_path = tmp_path / ".cached.png.tmp"

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("cached export should not recompute the map")

    monkeypatch.setattr(
        "flat_analyzer.ui.main_window.compute_export_result",
        fail_compute,
    )
    result = _run_export_job(
        None,
        None,
        60,
        40,
        settings,
        str(staged_path),
        "png",
        95,
        cached,
        None,
    )

    assert result.illumination is None
    assert staged_path.exists()


def test_export_job_cleans_staged_file_and_preserves_map_on_render_error(
    monkeypatch,
    tmp_path,
):
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    staged_path = tmp_path / ".output.png.tmp"
    staged_path.write_bytes(b"placeholder")

    def fail_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("flat_analyzer.ui.main_window.save_figure", fail_save)

    with pytest.raises(_ExportJobError) as raised:
        _run_export_job(
            np.ones((40, 60), dtype=float) * 1000,
            0.0,
            60,
            40,
            settings,
            str(staged_path),
            "png",
            95,
            None,
            None,
        )

    assert str(raised.value.cause) == "disk full"
    assert raised.value.illumination is not None
    assert not staged_path.exists()


def test_stale_export_result_removes_staged_file_without_commit(tmp_path):
    settings = RenderSettings(
        vmin=80,
        vmax=100,
        show_contours=False,
        contour_levels=np.array([]),
    )
    staged_path = tmp_path / ".stale.png.tmp"
    staged_path.write_bytes(b"stale")
    request = _ExportRequest(
        generation=1,
        key=(1, 3.0),
        image=np.ones((4, 6), dtype=float),
        sigma=3.0,
        export_width=6,
        export_height=4,
        path=str(tmp_path / "output.png"),
        fmt="png",
        jpg_quality=95,
        settings=settings,
    )
    future = Future()
    future.set_result(
        _ExportJobResult(
            illumination=None,
            temporary_path=str(staged_path),
        )
    )

    window = MainWindow.__new__(MainWindow)
    window._export_request = request
    window._export_future = future
    window._export_temp_path = staged_path
    window._image_generation = 2
    window._progress_window = None
    window._progress_label = None
    window._progress_bar = None

    window._handle_export_result(future)

    assert not staged_path.exists()
    assert not Path(request.path).exists()
    assert window._export_request is None


class _FakePreviewWidget:
    def winfo_width(self):
        return 800

    def winfo_height(self):
        return 600


class _FakePreviewFrame:
    def update_idletasks(self):
        pass


class _FakePlaceholder:
    def winfo_ismapped(self):
        return False


class _FakeFigure:
    dpi = 100

    def __init__(self):
        self.canvas = None
        self.cleared = False
        self.size = None

    def set_canvas(self, canvas):
        self.canvas = canvas

    def clear(self):
        if self.canvas is None:
            raise AssertionError("Figure must remain attached while clearing.")
        self.cleared = True

    def set_size_inches(self, width, height, forward=False):
        self.size = (width, height, forward)


class _FakeCanvas:
    def __init__(self, figure):
        self.figure = figure
        figure.canvas = self
        self.widget = _FakePreviewWidget()
        self.draw_count = 0

    def get_tk_widget(self):
        return self.widget

    def draw(self):
        self.draw_count += 1


def test_show_figure_reuses_existing_canvas_widget():
    old_figure = _FakeFigure()
    new_figure = _FakeFigure()
    canvas = _FakeCanvas(old_figure)
    window = MainWindow.__new__(MainWindow)
    window._placeholder = _FakePlaceholder()
    window._preview_frame = _FakePreviewFrame()
    window._canvas = canvas

    window._show_figure(new_figure)

    assert window._canvas is canvas
    assert canvas.figure is new_figure
    assert old_figure.canvas is None
    assert old_figure.cleared is True
    assert new_figure.canvas is canvas
    assert new_figure.size == (8.0, 6.0, False)
    assert canvas.draw_count == 1
