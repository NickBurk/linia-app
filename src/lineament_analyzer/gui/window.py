"""Main Qt window for georeferenced line detection and comparison."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lineament_analyzer.algorithms.canny_hough import (
    CannyHoughDetector,
    CannyHoughParameters,
)
from lineament_analyzer.algorithms.edlines import EdLinesDetector, EdLinesParameters
from lineament_analyzer.algorithms.lsd import LsdDetector, LsdParameters
from lineament_analyzer.algorithms.results import DetectionResult
from lineament_analyzer.core.windowed import check_cancel, process_dem
from lineament_analyzer.data.dem import Dem, load_preview
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.gui.tasks import Task

TEXT = {
    "ru": {
        "open": "Открыть GeoTIFF",
        "layer": "Слой",
        "algorithm": "Алгоритм",
        "algorithms": ["LSD", "Canny + Hough", "EDLines"],
        "run": "Обнаружить",
        "results": "Сравнение результатов",
        "all_results": "Все алгоритмы",
        "export": "Экспорт GeoJSON",
        "language": "Язык",
        "length": "Минимальная длина (пиксели предпросмотра)",
        "canny_low": "Порог Canny: нижний",
        "canny_high": "Порог Canny: верхний",
        "canny_aperture": "Размер апертуры",
        "canny_l2": "Градиент L2",
        "hough_votes": "Порог голосов Hough",
        "minimum_m": "Минимальная длина (м)",
        "gap_m": "Максимальный разрыв (м)",
        "ed_sigma": "Sigma",
        "ed_gradient": "Порог градиента",
        "ed_anchor": "Порог anchor",
        "ed_path": "Минимальная длина пути (px)",
        "ed_line_fit": "Порог ошибки аппроксимации",
        "ed_max_error": "Максимальная ошибка",
        "ed_distance": "Максимальное расстояние (px)",
        "ed_nfa": "NFA validation",
        "ed_pf": "PF mode",
        "ed_operator": "Оператор границ",
        "ed_scan": "Интервал сканирования",
        "units": "Единицы высот",
        "apply": "Применить",
        "cancel": "Отмена",
        "saved": "GeoJSON сохранён",
        "cancelled": "Отменено",
    },
    "en": {
        "open": "Open GeoTIFF",
        "layer": "Layer",
        "algorithm": "Algorithm",
        "algorithms": ["LSD", "Canny + Hough", "EDLines"],
        "run": "Detect",
        "results": "Compare results",
        "all_results": "All algorithms",
        "export": "Export GeoJSON",
        "language": "Language",
        "length": "Minimum length (preview pixels)",
        "canny_low": "Canny low threshold",
        "canny_high": "Canny high threshold",
        "canny_aperture": "Aperture size",
        "canny_l2": "L2 gradient",
        "hough_votes": "Hough vote threshold",
        "minimum_m": "Minimum segment length (m)",
        "gap_m": "Maximum gap (m)",
        "ed_sigma": "Sigma",
        "ed_gradient": "Gradient threshold",
        "ed_anchor": "Anchor threshold",
        "ed_path": "Minimum path length (px)",
        "ed_line_fit": "Line fit error threshold",
        "ed_max_error": "Maximum error threshold",
        "ed_distance": "Maximum line distance (px)",
        "ed_nfa": "NFA validation",
        "ed_pf": "PF mode",
        "ed_operator": "Edge operator",
        "ed_scan": "Scan interval",
        "units": "Elevation units",
        "apply": "Apply",
        "cancel": "Cancel",
        "saved": "GeoJSON saved",
        "cancelled": "Cancelled",
    },
}

ALGORITHM_NAMES = {"lsd": "LSD", "canny-hough": "Canny + Hough", "edlines": "EDLines"}
ALGORITHM_COLORS = {
    "lsd": Qt.GlobalColor.red,
    "canny-hough": Qt.GlobalColor.darkGreen,
    "edlines": Qt.GlobalColor.blue,
}


class MainWindow(QMainWindow):
    """DEM viewer with three asynchronous line detectors and result overlays."""

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
        self.results_by_algorithm: dict[str, DetectionResult] = {}
        self.language = "ru"
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.open_button = QPushButton()
        self.open_button.clicked.connect(self.open_file)
        self.layer_label = QLabel()
        self.layer = QComboBox()
        self.layer.currentIndexChanged.connect(self.render)
        self.algorithm_label = QLabel()
        self.algorithm_box = QComboBox()
        self.algorithm_box.currentIndexChanged.connect(self.algorithm_changed)
        self.detect_button = QPushButton()
        self.detect_button.clicked.connect(self.run_detection)
        self.result_label = QLabel()
        self.result_view_box = QComboBox()
        self.result_view_box.currentIndexChanged.connect(self.draw_result_overlays)
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
            self.algorithm_label,
            self.algorithm_box,
            self.detect_button,
            self.result_label,
            self.result_view_box,
            self.export_button,
            self.language_label,
            self.language_box,
        ):
            controls.addWidget(widget)

        self.parameter_stack = QStackedWidget()
        self._build_lsd_parameters()
        self._build_canny_parameters()
        self._build_edlines_parameters()

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

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.parameter_stack)
        layout.addLayout(processing)
        layout.addWidget(self.view)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1250, 820)
        self.change_language(0)

    def _spin(self, minimum, maximum, value, *, decimals=0, step=None):
        control = QDoubleSpinBox() if decimals else QSpinBox()
        control.setRange(minimum, maximum)
        if decimals:
            control.setDecimals(decimals)
        control.setValue(value)
        if step is not None:
            control.setSingleStep(step)
        return control

    def _build_lsd_parameters(self):
        page = QWidget()
        form = QFormLayout(page)
        self.length_label = QLabel()
        self.min_length = self._spin(1, 100000, 10, decimals=1, step=1)
        form.addRow(self.length_label, self.min_length)
        self.parameter_stack.addWidget(page)

    def _build_canny_parameters(self):
        page = QWidget()
        form = QFormLayout(page)
        self.canny_low = self._spin(0, 255, 30, decimals=1)
        self.canny_high = self._spin(1, 255, 90, decimals=1)
        self.canny_aperture = QComboBox()
        self.canny_aperture.addItems(["3", "5", "7"])
        self.canny_l2 = QCheckBox()
        self.canny_l2.setChecked(True)
        self.canny_l2_label = QLabel()
        self.hough_votes = self._spin(1, 10000, 20)
        self.canny_min_length = self._spin(0.1, 10000000, 100, decimals=1, step=10)
        self.canny_max_gap = self._spin(0, 1000000, 20, decimals=1, step=5)
        self.canny_labels = {
            "canny_low": QLabel(),
            "canny_high": QLabel(),
            "canny_aperture": QLabel(),
            "hough_votes": QLabel(),
            "minimum_m": QLabel(),
            "gap_m": QLabel(),
        }
        form.addRow(self.canny_labels["canny_low"], self.canny_low)
        form.addRow(self.canny_labels["canny_high"], self.canny_high)
        form.addRow(self.canny_labels["canny_aperture"], self.canny_aperture)
        form.addRow(self.canny_l2_label, self.canny_l2)
        form.addRow(self.canny_labels["hough_votes"], self.hough_votes)
        form.addRow(self.canny_labels["minimum_m"], self.canny_min_length)
        form.addRow(self.canny_labels["gap_m"], self.canny_max_gap)
        self.parameter_stack.addWidget(page)

    def _build_edlines_parameters(self):
        page = QWidget()
        form = QFormLayout(page)
        self.ed_sigma = self._spin(0.1, 10, 1, decimals=2, step=0.1)
        self.ed_gradient = self._spin(1, 255, 20)
        self.ed_anchor = self._spin(0, 255, 0)
        self.ed_path = self._spin(1, 100000, 10)
        self.ed_min_length = self._spin(0.1, 10000000, 100, decimals=1, step=10)
        self.ed_line_fit_error = self._spin(0.01, 1000, 1, decimals=2, step=0.1)
        self.ed_max_error = self._spin(0.01, 1000, 1.3, decimals=2, step=0.1)
        self.ed_max_distance = self._spin(0, 10000, 6, decimals=1, step=1)
        self.ed_nfa = QCheckBox()
        self.ed_nfa.setChecked(True)
        self.ed_pf = QCheckBox()
        self.ed_operator = QComboBox()
        self.ed_operator.addItems(["PREWITT", "SOBEL", "SCHARR"])
        self.ed_scan = self._spin(1, 1000, 1)
        self.ed_labels = {
            key: QLabel()
            for key in (
                "ed_sigma",
                "ed_gradient",
                "ed_anchor",
                "ed_path",
                "minimum_m",
                "ed_line_fit",
                "ed_max_error",
                "ed_distance",
                "ed_nfa",
                "ed_pf",
                "ed_operator",
                "ed_scan",
            )
        }
        rows = (
            ("ed_sigma", self.ed_sigma),
            ("ed_gradient", self.ed_gradient),
            ("ed_anchor", self.ed_anchor),
            ("ed_path", self.ed_path),
            ("minimum_m", self.ed_min_length),
            ("ed_line_fit", self.ed_line_fit_error),
            ("ed_max_error", self.ed_max_error),
            ("ed_distance", self.ed_max_distance),
            ("ed_nfa", self.ed_nfa),
            ("ed_pf", self.ed_pf),
            ("ed_operator", self.ed_operator),
            ("ed_scan", self.ed_scan),
        )
        for key, control in rows:
            form.addRow(self.ed_labels[key], control)
        self.parameter_stack.addWidget(page)

    def change_language(self, index: int) -> None:
        """Update controls while preserving the selected detector and result view."""
        self.language = "ru" if index == 0 else "en"
        labels = TEXT[self.language]
        self.setWindowTitle("Lineament Analyzer")
        self.open_button.setText(labels["open"])
        self.layer_label.setText(labels["layer"])
        self.algorithm_label.setText(labels["algorithm"])
        self.detect_button.setText(labels["run"])
        self.result_label.setText(labels["results"])
        self.export_button.setText(labels["export"])
        self.language_label.setText(labels["language"])
        self.length_label.setText(labels["length"])
        self.unit_label.setText(labels["units"])
        self.apply_button.setText(labels["apply"])
        self.cancel_button.setText(labels["cancel"])

        selected_algorithm = self.algorithm_box.currentData() or "lsd"
        self.algorithm_box.blockSignals(True)
        self.algorithm_box.clear()
        for algorithm_id, name in zip(ALGORITHM_NAMES, labels["algorithms"], strict=True):
            self.algorithm_box.addItem(name, algorithm_id)
        self.algorithm_box.setCurrentIndex(max(0, self.algorithm_box.findData(selected_algorithm)))
        self.algorithm_box.blockSignals(False)

        selected_layer = max(0, self.layer.currentIndex())
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItems(
            ["Высота", "Теневой рельеф", "Уклон", "Экспозиция", "TPI"]
            if index == 0
            else ["Elevation", "Hillshade", "Slope", "Aspect", "TPI"]
        )
        self.layer.setCurrentIndex(selected_layer)
        self.layer.blockSignals(False)
        self.canny_l2_label.setText(labels["canny_l2"])
        self.ed_nfa.setText(labels["ed_nfa"])
        self.ed_pf.setText(labels["ed_pf"])
        for key, label in self.canny_labels.items():
            label.setText(labels[key])
        for key, label in self.ed_labels.items():
            label.setText(labels[key])
        self.refresh_result_choices()
        self.algorithm_changed(self.algorithm_box.currentIndex())

    def algorithm_changed(self, index):
        self.parameter_stack.setCurrentIndex(max(0, index))

    def refresh_result_choices(self):
        previous = self.result_view_box.currentData() if self.result_view_box.count() else "all"
        self.result_view_box.blockSignals(True)
        self.result_view_box.clear()
        self.result_view_box.addItem(TEXT[self.language]["all_results"], "all")
        for algorithm_id, result in self.results_by_algorithm.items():
            self.result_view_box.addItem(
                f"{ALGORITHM_NAMES[algorithm_id]} ({len(result.segments)})", algorithm_id
            )
        index = self.result_view_box.findData(previous)
        self.result_view_box.setCurrentIndex(max(index, 0))
        self.result_view_box.blockSignals(False)
        self.draw_result_overlays()

    def set_busy(self, busy):
        for widget in (
            self.open_button,
            self.layer,
            self.algorithm_box,
            self.parameter_stack,
            self.detect_button,
            self.result_view_box,
            self.export_button,
            self.unit_box,
            self.z_factor,
            self.apply_button,
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
        message = TEXT[self.language]["cancelled"]
        if self.language == "ru":
            message += " / Cancelled"
        self.statusBar().showMessage(message)

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
        path, kind, parameters, preview, image, gray = result
        if path != self.source:
            self.results_by_algorithm.clear()
            self.segments = np.empty((0, 4), dtype=float)
        self.dem, self.image = preview, image
        self.source = path
        self.parameters = {"layer": kind, **parameters, "preview_shape": list(self.image.shape)}
        self.layer.blockSignals(True)
        self.layer.setCurrentIndex(["elevation", "hillshade", "slope", "aspect", "tpi"].index(kind))
        self.layer.blockSignals(False)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimage)))
        self.scene.setSceneRect(0, 0, width, height)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.refresh_result_choices()

    def _selected_parameters(self, algorithm_id):
        if algorithm_id == "lsd":
            return LsdParameters(self.min_length.value())
        if algorithm_id == "canny-hough":
            return CannyHoughParameters(
                canny_low_threshold=self.canny_low.value(),
                canny_high_threshold=self.canny_high.value(),
                aperture_size=int(self.canny_aperture.currentText()),
                l2_gradient=self.canny_l2.isChecked(),
                hough_threshold=self.hough_votes.value(),
                min_segment_length_m=self.canny_min_length.value(),
                max_gap_m=self.canny_max_gap.value(),
            )
        return EdLinesParameters(
            sigma=self.ed_sigma.value(),
            gradient_threshold=self.ed_gradient.value(),
            anchor_threshold=self.ed_anchor.value(),
            min_path_length_px=self.ed_path.value(),
            min_segment_length_m=self.ed_min_length.value(),
            line_fit_error_threshold=self.ed_line_fit_error.value(),
            max_error_threshold=self.ed_max_error.value(),
            max_distance_between_lines_px=self.ed_max_distance.value(),
            nfa_validation=self.ed_nfa.isChecked(),
            pf_mode=self.ed_pf.isChecked(),
            edge_operator=self.ed_operator.currentText(),
            scan_interval=self.ed_scan.value(),
        )

    def run_detection(self):
        if self.dem is None or self.image is None:
            return
        algorithm_id = self.algorithm_box.currentData()
        parameters = self._selected_parameters(algorithm_id)
        detector = {
            "lsd": LsdDetector(),
            "canny-hough": CannyHoughDetector(),
            "edlines": EdLinesDetector(),
        }[algorithm_id]
        analysis_dem = replace(
            self.dem,
            values=self.image,
            valid=self.dem.valid & np.isfinite(self.image),
        )
        self.start_task(
            lambda progress, cancel: detector.run(
                analysis_dem, parameters, progress=progress, cancel=cancel
            ),
            self.accept_detection,
        )

    def run_lsd(self):
        """Backward-compatible entry point retained for existing callers/tests."""
        self.algorithm_box.setCurrentIndex(self.algorithm_box.findData("lsd"))
        self.run_detection()

    def accept_detection(self, result: DetectionResult):
        if self.closing:
            return
        self.results_by_algorithm[result.algorithm_id] = result
        self.segments = result.as_pixel_array()
        self.parameters.update(
            algorithm_id=result.algorithm_id,
            detector_parameters=dict(result.parameters),
        )
        self.refresh_result_choices()
        self.statusBar().showMessage(
            f"{ALGORITHM_NAMES[result.algorithm_id]}: {len(result.segments)}"
        )

    def draw_result_overlays(self, *_):
        for item in self.scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        selected = self.result_view_box.currentData() if self.result_view_box.count() else "all"
        algorithm_ids = (
            list(self.results_by_algorithm)
            if selected == "all"
            else [selected]
            if selected in self.results_by_algorithm
            else []
        )
        for algorithm_id in algorithm_ids:
            path = QPainterPath()
            for segment in self.results_by_algorithm[algorithm_id].segments:
                x1, y1 = segment.pixel_start
                x2, y2 = segment.pixel_end
                path.moveTo(x1 + 0.5, y1 + 0.5)
                path.lineTo(x2 + 0.5, y2 + 0.5)
            pen = QPen(ALGORITHM_COLORS[algorithm_id])
            pen.setCosmetic(True)
            self.scene.addPath(path, pen)

    def save_geojson(self):
        if self.dem is None or not self.results_by_algorithm:
            return
        path, _ = QFileDialog.getSaveFileName(self, "GeoJSON", "", "GeoJSON (*.geojson)")
        if path:
            selected = self.result_view_box.currentData()
            results = (
                list(self.results_by_algorithm.values())
                if selected == "all"
                else [self.results_by_algorithm[selected]]
            )
            dem = self.dem
            self.start_task(
                lambda progress, cancel: export_geojson(path, dem, results),
                self.export_finished,
            )

    def export_finished(self, result):
        self.statusBar().showMessage(TEXT[self.language]["saved"])
