import time
from threading import Event

import numpy as np
import pytest
import rasterio
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication
from rasterio.transform import from_origin

from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.gui.window import MainWindow


@pytest.fixture
def app():
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
