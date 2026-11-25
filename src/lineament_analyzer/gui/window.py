"""Main Qt window for raster inspection and LSD detection."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lineament_analyzer.algorithms.lsd import LsdParameters, detect
from lineament_analyzer.core.windowed import check_cancel, process_dem
from lineament_analyzer.data.dem import Dem, load_preview
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.gui.tasks import Task

TEXT = {
    "ru": [
        "Открыть GeoTIFF",
        "Слой",
        "Высота",
        "Теневой рельеф",
        "Уклон",
        "Экспозиция",
        "TPI",
        "LSD",
        "Минимальная длина",
        "Линеаменты",
        "Язык",
        "Экспорт GeoJSON",
    ],
    "en": [
        "Open GeoTIFF",
        "Layer",
        "Elevation",
        "Hillshade",
        "Slope",
        "Aspect",
        "TPI",
        "LSD",
        "Minimum length",
        "Lineaments",
        "Language",
        "Export GeoJSON",
    ],
}


class MainWindow(QMainWindow):
    """Interactive DEM viewer with localized controls."""

    def __init__(self) -> None:
        super().__init__()
        self.task = None
        self.closing = False
        self.source = None
        self.cache = TemporaryDirectory(prefix="lineament-")
        self.parameters = {}
        self.dem: Dem | None = None
        self.image: np.ndarray | None = None
        self.segments = np.empty((0, 4), dtype=float)
        self.language = "ru"
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self.open_file)
        self.layer_label = QLabel()
        self.layer = QComboBox()
        self.layer.currentIndexChanged.connect(self.render)
        self.length_label = QLabel()
        self.min_length = QDoubleSpinBox()
        self.min_length.setRange(1, 100000)
        self.min_length.setValue(10)
        self.detect_button = QPushButton()
        self.detect_button.clicked.connect(self.run_lsd)
        self.export_button = QPushButton()
        self.export_button.clicked.connect(self.save_geojson)
        self.language_label = QLabel()
        self.language_box = QComboBox()
        self.language_box.addItems(["Русский", "English"])
        self.language_box.currentIndexChanged.connect(self.change_language)
        controls = QHBoxLayout()
        for widget in (
            self.open_button,
            self.layer_label,
            self.layer,
            self.length_label,
            self.min_length,
            self.detect_button,
            self.export_button,
            self.language_label,
            self.language_box,
        ):
            controls.addWidget(widget)
        layout = QVBoxLayout()
        layout.addLayout(controls)
        self.unit_box = QComboBox()
        self.unit_box.addItems(["Auto", "m", "ft", "us survey foot", "cm", "mm"])
        self.z_factor = QDoubleSpinBox()
        self.z_factor.setRange(0.0001, 10000)
        self.z_factor.setDecimals(4)
        self.z_factor.setValue(1)
        self.unit_label = QLabel()
        self.z_label = QLabel("Z-factor")
        self.apply_button = QPushButton()
        self.apply_button.clicked.connect(self.render)
        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self.cancel_task)
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        processing = QHBoxLayout()
        for widget in (
            self.unit_label,
            self.unit_box,
            self.z_label,
            self.z_factor,
            self.apply_button,
            self.progress,
            self.cancel_button,
        ):
            processing.addWidget(widget)
        layout.addLayout(processing)
        layout.addWidget(self.view)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1100, 750)
        self.change_language(0)

    def change_language(self, index: int) -> None:
        """Update visible text without restarting Qt."""
        self.language = "ru" if index == 0 else "en"
        self.unit_label.setText("Единицы высот" if index == 0 else "Elevation units")
        self.apply_button.setText("Применить" if index == 0 else "Apply")
        self.cancel_button.setText("Отмена" if index == 0 else "Cancel")
        labels = TEXT[self.language]
        self.setWindowTitle("Lineament Analyzer")
        self.open_button.setText(labels[0])
        self.layer_label.setText(labels[1])
        self.length_label.setText(
            labels[8] + (" (px предпросмотра)" if index == 0 else " (preview px)")
        )
        self.detect_button.setText(labels[9])
        self.language_label.setText(labels[10])
        self.export_button.setText(labels[11])
        selected = self.layer.currentIndex()
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItems(labels[2:7])
        self.layer.setCurrentIndex(max(0, selected))
        self.layer.blockSignals(False)

    def set_busy(self, busy):
        for widget in (
            self.open_button,
            self.layer,
            self.detect_button,
            self.export_button,
            self.unit_box,
            self.z_factor,
            self.apply_button,
            self.min_length,
        ):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 100)
        self.progress.setValue(0)

    def start_task(self, operation, receiver):
        if self.task is not None:
            return
        self.set_busy(True)
        self.task = Task(operation, self)
        self.task.result.connect(receiver)
        self.task.failed.connect(self.show_error)
        self.task.notices.connect(self.show_notice)
        self.task.cancelled.connect(self.show_cancelled)
        self.task.progress.connect(self.update_progress)
        self.task.finished.connect(self.task_finished)
        self.task.start()

    def show_error(self, message):
        self.statusBar().showMessage(message)
        if not self.closing:
            QMessageBox.warning(self, "Lineament Analyzer", message)

    def show_notice(self, message):
        if not self.closing:
            QMessageBox.warning(self, "Units / Единицы измерения", message)

    def update_progress(self, value):
        self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def show_cancelled(self):
        self.statusBar().showMessage("Отменено / Cancelled")

    def cancel_task(self):
        if self.task is not None:
            self.task.cancel()
            self.cancel_button.setEnabled(False)

    def task_finished(self):
        self.task.deleteLater()
        self.task = None
        self.set_busy(False)
        if self.dem is not None:
            self.layer.blockSignals(True)
            kind = self.parameters.get("layer", "elevation")
            self.layer.setCurrentIndex(
                ["elevation", "hillshade", "slope", "aspect", "tpi"].index(kind)
            )
            self.layer.blockSignals(False)
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.task is not None:
            self.closing = True
            self.cancel_task()
            event.ignore()
            return
        self.cache.cleanup()
        event.accept()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        # Shared entry point for file dialog and automated GUI checks.
        self.start_task(
            lambda progress, cancel: self.prepare_layer(path, "elevation", {}, cancel, progress),
            self.accept_layer,
        )

    def prepare_layer(self, path, kind, parameters, cancel, progress):
        preview = load_preview(path)
        check_cancel(cancel)
        if kind == "elevation":
            image = preview.values
        else:
            target = Path(self.cache.name) / "derivative.tif"
            process_dem(path, target, kind, progress=progress, cancel=cancel, **parameters)
            image = load_preview(target).values
        valid = preview.valid & np.isfinite(image)
        gray = np.zeros(image.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(image[valid], [2, 98])
            gray[valid] = np.clip((image[valid] - low) / max(high - low, 1e-12) * 255, 0, 255)
        check_cancel(cancel)
        return path, kind, parameters, preview, image, gray

    def render(self):
        if self.source is None:
            return
        kind = ["elevation", "hillshade", "slope", "aspect", "tpi"][self.layer.currentIndex()]
        parameters = {"vertical_scale": self.z_factor.value()}
        if self.unit_box.currentIndex() != 0:
            parameters["elevation_unit"] = self.unit_box.currentText()
        path = self.source
        self.start_task(
            lambda progress, cancel: self.prepare_layer(path, kind, parameters, cancel, progress),
            self.accept_layer,
        )

    def accept_layer(self, result):
        if self.closing:
            return
        path, kind, parameters, self.dem, self.image, gray = result
        self.source = path
        self.parameters = {"layer": kind, **parameters, "preview_shape": list(self.image.shape)}
        self.segments = np.empty((0, 4), dtype=float)
        self.layer.blockSignals(True)
        self.layer.setCurrentIndex(["elevation", "hillshade", "slope", "aspect", "tpi"].index(kind))
        self.layer.blockSignals(False)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimage)))
        self.scene.setSceneRect(0, 0, width, height)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def run_lsd(self):
        if self.dem is None or self.image is None:
            return
        image, valid = self.image, self.dem.valid
        params = LsdParameters(self.min_length.value())
        self.start_task(lambda progress, cancel: detect(image, valid, params), self.accept_lines)

    def accept_lines(self, lines):
        if self.closing:
            return
        self.segments = lines
        # A single graphics path keeps the scene responsive for many segments.
        for item in self.scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        path = QPainterPath()
        for x1, y1, x2, y2 in lines:
            path.moveTo(x1 + 0.5, y1 + 0.5)
            path.lineTo(x2 + 0.5, y2 + 0.5)
        pen = QPen(Qt.GlobalColor.red)
        pen.setCosmetic(True)
        self.scene.addPath(path, pen)
        self.parameters.update(min_length=self.min_length.value(), scale=0.8)
        self.statusBar().showMessage(f"LSD: {len(lines)} | preview pixels / пиксели предпросмотра")

    def save_geojson(self):
        if self.dem is None or len(self.segments) == 0:
            return
        path, _ = QFileDialog.getSaveFileName(self, "GeoJSON", "", "GeoJSON (*.geojson)")
        if path:
            dem, segments, parameters = self.dem, self.segments.copy(), self.parameters.copy()
            self.start_task(
                lambda progress, cancel: export_geojson(path, dem, segments, parameters=parameters),
                self.export_finished,
            )

    def export_finished(self, result):
        self.statusBar().showMessage("GeoJSON сохранён / saved")
