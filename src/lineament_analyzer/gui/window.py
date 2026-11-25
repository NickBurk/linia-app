"""Main Qt window for raster inspection and LSD detection."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPen, QPixmap
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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lineament_analyzer.algorithms.lsd import LsdParameters, detect
from lineament_analyzer.core.terrain import derivative
from lineament_analyzer.data.dem import Dem, load_dem
from lineament_analyzer.export.geojson import export_geojson

TEXT = {
    "ru": ["Открыть GeoTIFF", "Слой", "Высота", "Теневой рельеф", "Уклон", "Экспозиция",
           "TPI", "LSD", "Минимальная длина", "Линеаменты", "Язык", "Экспорт GeoJSON"],
    "en": ["Open GeoTIFF", "Layer", "Elevation", "Hillshade", "Slope", "Aspect",
           "TPI", "LSD", "Minimum length", "Lineaments", "Language", "Export GeoJSON"],
}


class MainWindow(QMainWindow):
    """Interactive DEM viewer with localized controls."""

    def __init__(self) -> None:
        super().__init__()
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
        for widget in (self.open_button, self.layer_label, self.layer, self.length_label,
                       self.min_length, self.detect_button, self.export_button,
                       self.language_label, self.language_box):
            controls.addWidget(widget)
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.view)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1100, 750)
        self.change_language(0)

    def change_language(self, index: int) -> None:
        """Update visible text without restarting Qt."""
        self.language = "ru" if index == 0 else "en"
        labels = TEXT[self.language]
        self.setWindowTitle("Lineament Analyzer")
        self.open_button.setText(labels[0])
        self.layer_label.setText(labels[1])
        self.length_label.setText(labels[8])
        self.detect_button.setText(labels[9])
        self.language_label.setText(labels[10])
        self.export_button.setText(labels[11])
        selected = self.layer.currentIndex()
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItems(labels[2:7])
        self.layer.setCurrentIndex(max(0, selected))
        self.layer.blockSignals(False)

    def open_file(self) -> None:
        """Choose a GeoTIFF DEM from disk."""
        path, _ = QFileDialog.getOpenFileName(self, "GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.dem = load_dem(path)
            self.segments = np.empty((0, 4), dtype=float)
            self.render()

    def render(self) -> None:
        """Display the selected raster layer."""
        if self.dem is None:
            return
        kind = ["elevation", "hillshade", "slope", "aspect", "tpi"][self.layer.currentIndex()]
        self.image = self.dem.values if kind == "elevation" else derivative(self.dem, kind)
        valid = self.dem.valid & np.isfinite(self.image)
        if not np.any(valid):
            return
        low, high = np.percentile(self.image[valid], [2, 98])
        pixels = np.clip((self.image - low) / max(high - low, 1e-12) * 255, 0, 255)
        gray = np.ascontiguousarray(np.where(valid, pixels, 0).astype(np.uint8))
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimage)))
        self.scene.setSceneRect(0, 0, width, height)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def run_lsd(self) -> None:
        """Draw detected segments over the current DEM layer."""
        if self.dem is None or self.image is None:
            return
        self.render()
        lines = detect(self.image, self.dem.valid, LsdParameters(self.min_length.value()))
        self.segments = lines
        pen = QPen(Qt.GlobalColor.red)
        pen.setCosmetic(True)
        for x1, y1, x2, y2 in lines:
            self.scene.addLine(x1, y1, x2, y2, pen)
        self.statusBar().showMessage(f"LSD: {len(lines)}")

    def save_geojson(self) -> None:
        """Save current LSD candidates in WGS84 GeoJSON."""
        if self.dem is None or len(self.segments) == 0:
            return
        path, _ = QFileDialog.getSaveFileName(self, "GeoJSON", "", "GeoJSON (*.geojson)")
        if path:
            export_geojson(path, self.dem, self.segments)
