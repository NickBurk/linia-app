import time
from threading import Event

import numpy as np
import pytest
import rasterio
from PySide6.QtCore import QSettings, Qt, QThread, QTimer
from PySide6.QtWidgets import QApplication
from rasterio.transform import from_origin

from lineament_analyzer.algorithms.results import build_detection_result
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.gui.project import load_project, save_project
from lineament_analyzer.gui.window import MainWindow


@pytest.fixture
def app(tmp_path):
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    settings = QSettings("NickBurk", "LineamentAnalyzer")
    settings.clear()
    settings.sync()
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    assert predicate(), "GUI task did not finish"


def test_result_delivered_on_gui_thread_and_event_loop_runs(app):
    window = MainWindow()
    started, release = Event(), Event()
    received, ticks = [], []
    timer = QTimer()
    timer.setInterval(1)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()

    def operation(progress, cancel):
        assert QThread.currentThread() != app.thread()
        started.set()
        release.wait(3)
        progress(50)
        return 42

    def receiver(value):
        received.append((value, QThread.currentThread() == app.thread()))

    window.start_task(operation, receiver)
    assert not window.open_button.isEnabled()
    wait_for(app, lambda: started.is_set() and len(ticks) >= 3)
    release.set()
    wait_for(app, lambda: window.task is None)
    assert received == [(42, True)]
    assert window.open_button.isEnabled()
    timer.stop()
    window.close()


def test_error_and_cancel(app, monkeypatch):
    window = MainWindow()
    errors, results = [], []
    monkeypatch.setattr(window, "show_error", errors.append)

    def fail(progress, cancel):
        raise ValueError("synthetic failure")

    window.start_task(fail, results.append)
    wait_for(app, lambda: window.task is None)
    assert "synthetic failure" in errors[0]
    started = Event()

    def cancellable(progress, cancel):
        started.set()
        while not cancel():
            time.sleep(0.001)
        check_cancel(cancel)

    window.start_task(cancellable, results.append)
    wait_for(app, started.is_set)
    window.cancel_task()
    wait_for(app, lambda: window.task is None)
    assert results == []
    assert "Cancelled" in window.statusBar().currentMessage()
    window.close()


def test_close_cancels_and_waits(app):
    window = MainWindow()
    started = Event()

    def operation(progress, cancel):
        started.set()
        while not cancel():
            time.sleep(0.001)
        check_cancel(cancel)

    window.show()
    window.start_task(operation, lambda value: None)
    wait_for(app, started.is_set)
    window.close()
    wait_for(app, lambda: window.task is None and not window.isVisible())


def test_real_gui_load_terrain_lsd(app, tmp_path):
    path = tmp_path / "dem.tif"
    y, x = np.indices((100, 100))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float64",
        crs="EPSG:32633",
        transform=from_origin(500000, 5000000, 10, 10),
    ) as dst:
        dst.write((x + y + (x > 50) * 100).astype(float), 1)
        dst.set_band_unit(1, "m")
    window = MainWindow()
    window.load_path(path)
    wait_for(app, lambda: window.task is None)
    assert window.dem.values.shape == (100, 100)
    window.layer.setCurrentIndex(1)
    wait_for(app, lambda: window.task is None)
    assert window.parameters["layer"] == "hillshade"
    window.run_lsd()
    wait_for(app, lambda: window.task is None)
    assert len(window.segments) > 0
    assert len(window.scene.items()) == 2
    window.close()


def test_gui_runs_and_compares_canny_hough_and_edlines(app, tmp_path):
    path = tmp_path / "line-dem.tif"
    image = np.zeros((160, 160), dtype=np.float32)
    image[76:84, 15:145] = 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=160,
        width=160,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500000, 4500000, 5, 5),
    ) as dst:
        dst.write(image, 1)
        dst.set_band_unit(1, "m")

    window = MainWindow()
    window.load_path(path)
    wait_for(app, lambda: window.task is None)
    window.algorithm_box.setCurrentIndex(window.algorithm_box.findData("canny-hough"))
    assert window.parameter_stack.currentIndex() == 1
    window.canny_min_length.setValue(200)
    window.hough_votes.setValue(10)
    window.run_detection()
    wait_for(app, lambda: window.task is None)
    assert window.results_by_algorithm["canny-hough"].segments

    window.algorithm_box.setCurrentIndex(window.algorithm_box.findData("edlines"))
    assert window.parameter_stack.currentIndex() == 2
    window.ed_min_length.setValue(200)
    window.ed_gradient.setValue(12)
    window.run_detection()
    wait_for(app, lambda: window.task is None)
    assert window.results_by_algorithm["edlines"].segments
    assert window.result_view_box.currentData() == "all"
    assert len(window.scene.items()) == 3

    window.result_view_box.setCurrentIndex(window.result_view_box.findData("canny-hough"))
    assert len(window.scene.items()) == 2
    window.result_view_box.setCurrentIndex(window.result_view_box.findData("all"))
    assert len(window.scene.items()) == 3
    window.language_box.setCurrentIndex(1)
    assert window.algorithm_box.currentData() == "edlines"
    assert window.result_view_box.itemText(0) == "All algorithms"
    window.close()


def test_gui_postprocessing_runs_in_worker_and_toggles_class_groups(app, tmp_path):
    path = tmp_path / "postprocess-dem.tif"
    image = np.zeros((120, 120), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=120,
        width=120,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500000, 4500000, 10, 10),
    ) as dst:
        dst.write(image, 1)
    window = MainWindow()
    window.load_path(path)
    wait_for(app, lambda: window.task is None)
    raw = build_detection_result(
        window.dem,
        [[10, 15, 45, 15], [70, 10, 70, 50]],
        algorithm_id="lsd",
        parameters={"min_length": 5},
    )
    window.accept_detection(raw)
    window.run_postprocessing()
    assert window.task is not None
    wait_for(app, lambda: window.task is None)
    processed = window.postprocessed_results_by_algorithm["lsd"]
    assert len(processed.segments) == 2
    assert "object_count" not in window.statistics_label.text()
    assert "Объекты: 2" in window.statistics_label.text()
    assert window.class_group_list.count() >= 2
    assert len(window.scene.items()) >= 3

    first = window.class_group_list.item(0)
    first.setCheckState(Qt.CheckState.Unchecked)
    assert len(window.scene.items()) >= 2
    window.language_box.setCurrentIndex(1)
    assert "Objects: 2" in window.statistics_label.text()
    window.close()


def write_dem(path, offset=0):
    values = np.arange(64, dtype=np.float32).reshape(8, 8) + offset
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=1,
        dtype="float32", crs="EPSG:32633", transform=from_origin(500000, 4500000, 5, 5),
    ) as dataset:
        dataset.write(values, 1)
    return values


def test_sequential_dem_load_is_transactional_and_repeatable(app, tmp_path, monkeypatch):
    first, second, corrupt = (tmp_path / name for name in ("first.tif", "second.tif", "broken.tif"))
    write_dem(first)
    write_dem(second, 100)
    corrupt.write_bytes(b"not a geotiff")
    window = MainWindow()
    errors = []
    monkeypatch.setattr(window, "show_error", errors.append)

    for source in (first, first):
        assert window.load_path(source)
        wait_for(app, lambda: window.task is None)
        assert window.source == str(source.resolve())
        assert window.dem.values.shape == (8, 8)

    result = build_detection_result(window.dem, [[0, 0, 7, 7]], algorithm_id="lsd", parameters={})
    window.accept_detection(result)
    assert window.load_path(second)
    wait_for(app, lambda: window.task is None)
    assert window.source == str(second.resolve())
    assert not window.results_by_algorithm

    before = window.dem
    assert window.load_path(corrupt)
    wait_for(app, lambda: window.task is None)
    assert window.dem is before
    assert window.source == str(second.resolve())
    assert errors
    assert window.load_path(second)
    wait_for(app, lambda: window.task is None)
    window.close()


def test_project_round_trip_and_gui_language_docks(app, tmp_path):
    path = tmp_path / "project-dem.tif"
    write_dem(path)
    window = MainWindow()
    window.load_path(path)
    wait_for(app, lambda: window.task is None)
    result = build_detection_result(window.dem, [[0, 0, 7, 7]], algorithm_id="lsd", parameters={"min_length": 1})
    window.accept_detection(result)
    project_file = tmp_path / "analysis.laproj"
    save_project(project_file, {
        "dem_path": window.source,
        "detection_results": window.results_by_algorithm,
        "postprocessed_results": {},
        "class_visibility": {"example": False},
        "display": {"algorithm": "lsd", "layer": "elevation", **window._ui_parameters()},
    })
    restored = load_project(project_file)
    assert restored["dem_path"] == str(path.resolve())
    assert restored["detection_results"]["lsd"].segments == result.segments
    assert restored["class_visibility"] == {"example": False}
    assert restored["display"]["detector_parameters"]["min_length"] == window.min_length.value()
    window.min_length.setValue(77)
    window._apply_ui_parameters(restored["display"])
    assert window.min_length.value() != 77
    assert window.property_tabs.count() == 4
    window.language_box.setCurrentIndex(1)
    assert window.property_tabs.tabText(0) == "Data"
    assert window.file_menu.title() == "File"
    window.language_box.setCurrentIndex(0)
    assert window.property_tabs.tabText(0) == "Данные"
    window.close()


def test_opt_in_restore_reloads_dem_and_missing_path_is_safe(app, tmp_path):
    source = tmp_path / "restore.tif"
    write_dem(source)
    first = MainWindow()
    first.load_path(source)
    wait_for(app, lambda: first.task is None)
    first.min_length.setValue(23)
    first.restore_action.setChecked(True)
    first.close()

    restored = MainWindow()
    wait_for(app, lambda: restored.task is None)
    assert restored.source == str(source.resolve())
    assert restored.min_length.value() == 23
    restored.close()

    settings = QSettings("NickBurk", "LineamentAnalyzer")
    settings.setValue("lastDem", str(tmp_path / "moved-away.tif"))
    settings.setValue("autoRestore", True)
    missing = MainWindow()
    assert missing.dem is None
    assert "moved-away.tif" in missing.statusBar().currentMessage()
    missing.close()
