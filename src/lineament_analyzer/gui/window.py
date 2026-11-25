"""Main Qt window for georeferenced line detection and comparison."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from affine import Affine
from PySide6.QtCore import QPointF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lineament_analyzer.algorithms.canny_hough import CannyHoughParameters
from lineament_analyzer.algorithms.edlines import EdLinesParameters
from lineament_analyzer.algorithms.lsd import LsdParameters
from lineament_analyzer.algorithms.postprocessing import (
    PostprocessParameters,
    postprocess_result,
    summarize_results,
)
from lineament_analyzer.algorithms.results import DetectionResult, parameters_dict
from lineament_analyzer.core.windowed import check_cancel, process_dem
from lineament_analyzer.data.dem import Dem, load_preview
from lineament_analyzer.engine import (
    CONTROLS,
    METHODS,
    PROFILES,
    dataset_grid,
    detector_component,
    detector_response,
    run_detector,
)
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.export.geopackage import export_geopackage
from lineament_analyzer.export.shapefile import export_shapefile
from lineament_analyzer.gui.experiments import ExperimentPanel
from lineament_analyzer.gui.project import load_project, save_project
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
        "postprocess": "Постобработка",
        "run_postprocess": "Обработать результаты",
        "merge": "Объединять сегменты",
        "angle": "Угловое отклонение (°)",
        "endpoint_distance": "Расстояние концов (px)",
        "collinearity": "Допуск коллинеарности (px)",
        "maximum_gap": "Максимальный разрыв (px)",
        "minimum_length_filter": "Мин. длина (px)",
        "maximum_length_filter": "Макс. длина (px; 0 = без лимита)",
        "minimum_confidence": "Мин. confidence (не задан = выкл.)",
        "orientation_min": "Мин. осевой азимут (°; -1 = выкл.)",
        "orientation_max": "Макс. осевой азимут (°; -1 = выкл.)",
        "statistics": "Статистика линеаментов",
        "class_groups": "Группы классов (цвет/видимость)",
        "stat_count": "Объекты",
        "stat_total_length": "Суммарная длина",
        "stat_density": "Плотность / км²",
        "stat_rose": "Роза азимутов 0–180°",
        "stat_classes": "Классы",
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
        "saved": "Экспорт завершён",
        "cancelled": "Отменено",
        "missing_dem": "Файл DEM не найден: {}", "no_dem": "DEM не загружена",
        "menu_file": "Файл", "menu_view": "Вид", "menu_analysis": "Анализ",
        "menu_export": "Экспорт", "menu_help": "Справка", "new_project": "Новый проект",
        "open_project": "Открыть проект…", "save_project": "Сохранить проект",
        "save_as": "Сохранить проект как…", "close_dem": "Закрыть DEM",
        "restore": "Автовосстановление последнего DEM", "zoom": "Масштабировать к данным",
        "reset_panels": "Сбросить панели", "terrain": "Анализ рельефа",
        "compare": "Сравнить результаты", "about": "О программе",
        "guide": "Руководство пользователя", "tab_data": "Данные",
        "tab_detection": "Обнаружение", "tab_post": "Постобработка",
        "tab_stats": "Статистика", "save_ok": "Проект сохранён",
        "project_error": "Не удалось открыть проект: {}", "layers": "Панель слоёв",
        "properties": "Панель свойств", "exit": "Выход", "shapefile": "Shapefile",
        "open_dem_title": "Открыть DEM", "open_project_title": "Открыть проект",
        "save_project_title": "Сохранить проект", "about_text": "Анализ DEM и геологических линеаментов",
        "guide_text": "Откройте GeoTIFF DEM, выберите алгоритм и запустите обнаружение.",
        "resolution": "Разрешение", "range": "Диапазон",
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
        "postprocess": "Postprocessing",
        "run_postprocess": "Process results",
        "merge": "Merge segments",
        "angle": "Angular tolerance (°)",
        "endpoint_distance": "Endpoint distance (px)",
        "collinearity": "Collinearity tolerance (px)",
        "maximum_gap": "Maximum gap (px)",
        "minimum_length_filter": "Minimum length (px)",
        "maximum_length_filter": "Maximum length (px; 0 = unlimited)",
        "minimum_confidence": "Minimum confidence (unset = off)",
        "orientation_min": "Minimum axial azimuth (°; -1 = off)",
        "orientation_max": "Maximum axial azimuth (°; -1 = off)",
        "statistics": "Lineament statistics",
        "class_groups": "Class groups (color/visibility)",
        "stat_count": "Objects",
        "stat_total_length": "Total length",
        "stat_density": "Density / km²",
        "stat_rose": "Axial azimuth rose 0–180°",
        "stat_classes": "Classes",
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
        "saved": "Export complete",
        "cancelled": "Cancelled",
        "missing_dem": "DEM file not found: {}", "no_dem": "No DEM loaded",
        "menu_file": "File", "menu_view": "View", "menu_analysis": "Analysis",
        "menu_export": "Export", "menu_help": "Help", "new_project": "New project",
        "open_project": "Open project…", "save_project": "Save project",
        "save_as": "Save project as…", "close_dem": "Close DEM",
        "restore": "Restore last DEM automatically", "zoom": "Zoom to data",
        "reset_panels": "Reset panels", "terrain": "Terrain analysis",
        "compare": "Compare results", "about": "About", "guide": "User guide",
        "tab_data": "Data", "tab_detection": "Detection",
        "tab_post": "Postprocessing", "tab_stats": "Statistics",
        "save_ok": "Project saved", "project_error": "Could not open project: {}",
        "layers": "Layers panel", "properties": "Properties panel",
        "exit": "Exit", "shapefile": "Shapefile",
        "open_dem_title": "Open DEM", "open_project_title": "Open project",
        "save_project_title": "Save project", "about_text": "DEM and geological lineament analysis",
        "guide_text": "Open a GeoTIFF DEM, choose a detector and start detection.",
        "resolution": "Resolution", "range": "Range",
    },
}

ALGORITHM_NAMES = {key: method.name for key, method in METHODS.items()}
ALGORITHM_NAMES["consensus"] = "Algorithm Consensus"
FILTER_METHODS = {"directional", "wavelet", "log-gabor", "shearlet"}
TERRAIN_MODES = ("elevation", "hillshade", "slope", "aspect", "tpi")


ALGORITHM_COLORS = {
    "lsd": Qt.GlobalColor.red,
    "canny-hough": Qt.GlobalColor.darkGreen,
    "edlines": Qt.GlobalColor.blue,
}


class MapCanvas(QGraphicsView):
    """Aspect-preserving raster canvas with wheel zoom and hand panning."""

    pixelPicked = Signal(object)
    picking = False

    def mousePressEvent(self, event):
        if self.picking and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            self.pixelPicked.emit((point.x() - 0.5, point.y() - 0.5))
            event.accept()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()


class MainWindow(QMainWindow):
    """DEM viewer with asynchronous detection, validation and result overlays."""

    def __init__(self) -> None:
        super().__init__()
        self.task = None
        self.closing = False
        self.source = None
        self.native_grid = None
        self.analysis_path = None
        self.endpoints = []
        self.cache = TemporaryDirectory(prefix="lineament-")
        self.parameters = {}
        self.dem: Dem | None = None
        self.image: np.ndarray | None = None
        self.segments = np.empty((0, 4), dtype=float)
        self.results_by_algorithm: dict[str, DetectionResult] = {}
        self.postprocessed_results_by_algorithm: dict[str, DetectionResult] = {}
        self.class_visibility: dict[str, bool] = {}
        self.unique_candidates = {}
        self.last_statistics = None
        self.language = "ru"
        self.settings = QSettings("NickBurk", "LineamentAnalyzer")
        self.project_path = None
        self.pending_project = None
        self.scene = QGraphicsScene(self)
        self.view = MapCanvas(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.pixelPicked.connect(self.accept_astar_point)

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
        self.export_format = QComboBox()
        self.export_format.addItems(["GeoJSON", "Shapefile", "GeoPackage"])
        self.postprocess_button = QPushButton()
        self.postprocess_button.clicked.connect(self.run_postprocessing)
        self.language_label = QLabel()
        self.language_box = QComboBox()
        self.language_box.addItems(["Русский", "English"])
        self.language_box.currentIndexChanged.connect(self.change_language)

        self.parameter_stack = QStackedWidget()
        self._build_lsd_parameters()
        self._build_canny_parameters()
        self._build_edlines_parameters()
        self.advanced_controls = {}
        self.advanced_labels = {}
        for algorithm, method in list(METHODS.items())[3:]:
            form = QWidget()
            layout = QFormLayout(form)
            controls = self.advanced_controls[algorithm] = {}
            for key, default in vars(method.parameters()).items():
                _, _, minimum, maximum = CONTROLS[key]
                if algorithm == "ransac" and key == "max_gap_px":
                    minimum, maximum = 1, 32
                field_type = method.parameters.__dataclass_fields__[key].type
                decimals = 0 if field_type is int else 3
                control = self._spin(minimum, maximum, default, decimals=decimals)
                label = QLabel()
                self.advanced_labels[algorithm, key] = label
                controls[key] = control
                layout.addRow(label, control)
            self.parameter_stack.addWidget(form)

        self.resolution_mode = QComboBox()
        self.resolution_mode.addItem("", "preview")
        self.resolution_mode.addItem("", "full")
        self.resolution_label = QLabel()
        self.memory_budget = self._spin(16, 32768, 1024)
        self.memory_budget.setSuffix(" MiB")
        self.memory_label = QLabel()
        self.profile_box = QComboBox()
        self.profile_label = QLabel()
        self.apply_profile_button = QPushButton()
        self.apply_profile_button.clicked.connect(self.apply_profile)
        self.save_profile_button = QPushButton()
        self.save_profile_button.clicked.connect(self.save_profile)
        self.response_button = QPushButton()
        self.response_button.clicked.connect(self.show_response)
        self.orientation_response = QCheckBox()
        self.response_band = self._spin(0, 512, 0)
        self.response_band.setPrefix("№ ")
        self.response_band_label = QLabel()
        self.astar_manual = QPushButton()
        self.astar_manual.clicked.connect(self.pick_astar_endpoints)

        self.postprocess_group = QGroupBox()
        post_form = QFormLayout(self.postprocess_group)
        self.merge_enabled = QCheckBox()
        self.merge_enabled.setChecked(True)
        self.post_angle = self._spin(0, 90, 12, decimals=1)
        self.post_endpoint = self._spin(0, 10000, 15, decimals=1)
        self.post_collinearity = self._spin(0, 10000, 4, decimals=1)
        self.post_gap = self._spin(0, 10000, 12, decimals=1)
        self.post_min_length = self._spin(0, 100000, 0, decimals=1)
        self.post_max_length = self._spin(0, 100000, 0, decimals=1)
        self.post_confidence = self._spin(0, 1, 0.5, decimals=3)
        self.post_confidence_enabled = QCheckBox()
        self.post_orientation_min = self._spin(-1, 179.9, -1, decimals=1)
        self.post_orientation_max = self._spin(-1, 179.9, -1, decimals=1)
        self.post_labels = {key: QLabel() for key in (
            "merge", "angle", "endpoint_distance", "collinearity", "maximum_gap",
            "minimum_length_filter", "maximum_length_filter", "minimum_confidence",
            "orientation_min", "orientation_max",
        )}
        post_form.addRow(self.post_labels["merge"], self.merge_enabled)
        for key, control in (
            ("angle", self.post_angle),
            ("endpoint_distance", self.post_endpoint),
            ("collinearity", self.post_collinearity),
            ("maximum_gap", self.post_gap),
            ("minimum_length_filter", self.post_min_length),
            ("maximum_length_filter", self.post_max_length),
            ("minimum_confidence", self.post_confidence),
            ("", self.post_confidence_enabled),
            ("orientation_min", self.post_orientation_min),
            ("orientation_max", self.post_orientation_max),
        ):
            post_form.addRow(self.post_labels[key] if key else QLabel(), control)
        self.class_label = QLabel()
        self.class_group_list = QListWidget()
        self.class_group_list.setMaximumHeight(100)
        self.class_group_list.itemChanged.connect(self.class_visibility_changed)
        self.statistics_label = QLabel()
        self.statistics_label.setWordWrap(True)

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

        processing = QHBoxLayout()
        for widget in (
            self.unit_label,
            self.unit_box,
            self.z_label,
            self.z_factor,
            self.apply_button,
        ):
            processing.addWidget(widget)

        self._build_workspace(processing)
        self.resize(1250, 820)
        language_index = 1 if self.settings.value("language", "ru") == "en" else 0
        self.language_box.blockSignals(True)
        self.language_box.setCurrentIndex(language_index)
        self.language_box.blockSignals(False)
        self.change_language(language_index)
        algorithm_index = self.algorithm_box.findData(self.settings.value("algorithm", "lsd"))
        if algorithm_index >= 0:
            self.algorithm_box.setCurrentIndex(algorithm_index)
        saved_ui = self.settings.value("uiParameters", "")
        if saved_ui:
            try:
                self._apply_ui_parameters(json.loads(saved_ui))
            except (ValueError, TypeError, json.JSONDecodeError):
                self.statusBar().showMessage("Saved controls were invalid; defaults are in use")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        self.restore_action.setChecked(self.settings.value("autoRestore", False, type=bool))
        last_dem = self.settings.value("lastDem", "")
        if self.restore_action.isChecked() and last_dem:
            if Path(last_dem).is_file():
                self.load_path(last_dem)
            else:
                self.statusBar().showMessage(TEXT[self.language]["missing_dem"].format(last_dem))

    def _build_workspace(self, processing):
        """Arrange the existing controls around a large, aspect-safe map canvas."""
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setCentralWidget(self.view)

        self.layer_tree = QListWidget()
        self.layer_tree.itemDoubleClicked.connect(self._activate_layer_item)
        self.layer_tree.setToolTip("DEM и вычисляемые представления рельефа")
        layer_dock = QDockWidget(self)
        layer_dock.setObjectName("layersDock")
        layer_dock.setWidget(self.layer_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, layer_dock)
        self.layers_dock = layer_dock

        tabs = QTabWidget()
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.addWidget(self.open_button)
        data_layout.addWidget(self.language_label)
        data_layout.addWidget(self.language_box)
        self.data_info = QLabel("DEM не загружена")
        self.data_info.setWordWrap(True)
        data_layout.addWidget(self.data_info)
        data_layout.addStretch(1)
        tabs.addTab(data_page, "Данные")

        terrain_page = QWidget()
        terrain_layout = QVBoxLayout(terrain_page)
        terrain_form = QFormLayout()
        terrain_form.addRow(self.layer_label, self.layer)
        terrain_layout.addLayout(terrain_form)
        terrain_layout.addLayout(processing)
        terrain_layout.addStretch(1)
        self.terrain_tab = tabs.addTab(terrain_page, "")

        detection_page = QWidget()
        detect_layout = QVBoxLayout(detection_page)
        detect_form = QFormLayout()
        detect_form.addRow(self.algorithm_label, self.algorithm_box)
        detect_form.addRow(self.resolution_label, self.resolution_mode)
        detect_form.addRow(self.memory_label, self.memory_budget)
        detect_form.addRow(self.profile_label, self.profile_box)
        profile_buttons = QHBoxLayout()
        profile_buttons.addWidget(self.apply_profile_button)
        profile_buttons.addWidget(self.save_profile_button)
        detect_form.addRow(profile_buttons)
        detect_form.addRow(self.parameter_stack)
        detect_form.addRow(self.result_label, self.result_view_box)
        detect_layout.addLayout(detect_form)
        detect_layout.addWidget(self.detect_button)
        detect_layout.addWidget(self.astar_manual)
        detect_layout.addWidget(self.orientation_response)
        band_form = QFormLayout()
        band_form.addRow(self.response_band_label, self.response_band)
        detect_layout.addLayout(band_form)
        detect_layout.addWidget(self.response_button)
        detect_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(detection_page)
        self.detection_tab = tabs.addTab(scroll, "Обнаружение")
        self.comparison = ExperimentPanel(self, False)
        self.comparison_tab = tabs.addTab(self.comparison, "")

        post_page = QWidget()
        post_layout = QVBoxLayout(post_page)
        post_layout.addWidget(self.postprocess_group)
        post_layout.addWidget(self.postprocess_button)
        post_layout.addWidget(self.class_label)
        post_layout.addWidget(self.class_group_list)
        post_layout.addStretch(1)
        self.post_tab = tabs.addTab(post_page, "Постобработка")
        self.validation = ExperimentPanel(self, True)
        validation_scroll = QScrollArea()
        validation_scroll.setWidgetResizable(True)
        validation_scroll.setWidget(self.validation)
        self.validation_tab = tabs.addTab(validation_scroll, "")

        stats_page = QWidget()
        stats_layout = QVBoxLayout(stats_page)
        stats_layout.addWidget(self.statistics_label)
        stats_layout.addStretch(1)
        self.stats_tab = tabs.addTab(stats_page, "Статистика")
        export_page = QWidget()
        export_layout = QVBoxLayout(export_page)
        export_layout.addWidget(self.export_format)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch(1)
        self.export_tab = tabs.addTab(export_page, "")
        tabs.setUsesScrollButtons(True)
        props_dock = QDockWidget(self)
        props_dock.setObjectName("propertiesDock")
        props_dock.setWidget(tabs)
        props_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, props_dock)
        self.properties_dock = props_dock
        self.property_tabs = tabs

        self._build_menus()
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_dem_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.zoom_action)
        toolbar.addAction(self.detect_action)
        toolbar.addAction(self.postprocess_action)
        toolbar.addAction(self.export_action)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(160)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)

    def _build_menus(self):
        self.file_menu = self.menuBar().addMenu("")
        self.new_project_action = self.file_menu.addAction("")
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action = self.file_menu.addAction("")
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action = self.file_menu.addAction("")
        self.save_project_action.triggered.connect(self.save_project)
        self.save_as_action = self.file_menu.addAction("")
        self.save_as_action.triggered.connect(self.save_project_as)
        self.file_menu.addSeparator()
        self.open_dem_action = self.file_menu.addAction("")
        self.open_dem_action.triggered.connect(self.open_file)
        self.close_dem_action = self.file_menu.addAction("")
        self.close_dem_action.triggered.connect(self.close_dem)
        self.file_menu.addSeparator()
        self.restore_action = self.file_menu.addAction("")
        self.restore_action.setCheckable(True)
        self.restore_action.toggled.connect(lambda value: self.settings.setValue("autoRestore", value))
        self.exit_action = self.file_menu.addAction("")
        self.exit_action.triggered.connect(self.close)

        self.view_menu = self.menuBar().addMenu("")
        self.layers_toggle_action = self.layers_dock.toggleViewAction()
        self.properties_toggle_action = self.properties_dock.toggleViewAction()
        self.view_menu.addAction(self.layers_toggle_action)
        self.view_menu.addAction(self.properties_toggle_action)
        self.zoom_action = self.view_menu.addAction("")
        self.zoom_action.triggered.connect(self.zoom_to_dem)
        self.reset_panels_action = self.view_menu.addAction("")
        self.reset_panels_action.triggered.connect(self.reset_panels)

        self.analysis_menu = self.menuBar().addMenu("")
        self.terrain_action = self.analysis_menu.addAction("")
        self.terrain_action.triggered.connect(lambda: self.property_tabs.setCurrentIndex(self.terrain_tab))
        self.detect_action = self.analysis_menu.addAction("")
        self.detect_action.triggered.connect(self.run_detection)
        self.postprocess_action = self.analysis_menu.addAction("")
        self.postprocess_action.triggered.connect(self.run_postprocessing)
        self.compare_action = self.analysis_menu.addAction("")
        self.compare_action.triggered.connect(lambda: self.property_tabs.setCurrentIndex(self.comparison_tab))

        export_menu = self.menuBar().addMenu("")
        self.export_action = export_menu.addAction("")
        self.export_action.triggered.connect(self.save_geojson)
        self.shapefile_action = export_menu.addAction("Shapefile")
        self.shapefile_action.triggered.connect(self._select_shapefile)
        self.help_menu = self.menuBar().addMenu("")
        self.about_action = self.help_menu.addAction("")
        self.about_action.triggered.connect(self.about)
        self.guide_action = self.help_menu.addAction("")
        self.guide_action.triggered.connect(self.user_guide)

    def _activate_layer_item(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key in {"elevation", "hillshade", "slope", "aspect", "tpi"}:
            self.layer.setCurrentIndex(["elevation", "hillshade", "slope", "aspect", "tpi"].index(key))
        elif key in ALGORITHM_NAMES or isinstance(key, str) and key.startswith("post:"):
            index = self.result_view_box.findData(key)
            if index >= 0:
                self.result_view_box.setCurrentIndex(index)

    def _refresh_layers(self):
        self.layer_tree.clear()
        if self.source:
            item = QListWidgetItem(f"DEM — {Path(self.source).name}")
            item.setData(Qt.ItemDataRole.UserRole, "elevation")
            self.layer_tree.addItem(item)
            kind = self.parameters.get("layer", "elevation")
            names = {"hillshade": "Hillshade", "slope": "Slope", "aspect": "Aspect", "tpi": "TPI"}
            if kind in names:
                item = QListWidgetItem(names[kind])
                item.setData(Qt.ItemDataRole.UserRole, kind)
                self.layer_tree.addItem(item)
        for algorithm_id in self.results_by_algorithm:
            item = QListWidgetItem(ALGORITHM_NAMES[algorithm_id])
            item.setData(Qt.ItemDataRole.UserRole, algorithm_id)
            self.layer_tree.addItem(item)
        for algorithm_id in self.postprocessed_results_by_algorithm:
            item = QListWidgetItem(f"{ALGORITHM_NAMES.get(algorithm_id, algorithm_id)} · post")
            item.setData(Qt.ItemDataRole.UserRole, f"post:{algorithm_id}")
            self.layer_tree.addItem(item)
    def _select_shapefile(self):
        self.export_format.setCurrentIndex(1)
        self.save_geojson()

    def about(self):
        QMessageBox.about(self, "Lineament Analyzer", TEXT[self.language]["about_text"])

    def user_guide(self):
        QMessageBox.information(self, "Lineament Analyzer", TEXT[self.language]["guide_text"])

    def reset_panels(self):
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.layers_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        self.layers_dock.show()
        self.properties_dock.show()

    def zoom_to_dem(self):
        if self.scene.sceneRect().isValid():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def close_dem(self):
        if self.task is not None:
            return
        self.source = self.dem = self.image = None
        self.native_grid = self.analysis_path = None
        self.clear_experiments()
        self.parameters.clear()
        self.results_by_algorithm.clear()
        self.postprocessed_results_by_algorithm.clear()
        self.scene.clear()
        self.scene.setSceneRect(0, 0, 1, 1)
        self.data_info.setText(TEXT[self.language]["no_dem"])
        self._refresh_layers()
        self.refresh_result_choices()

    def new_project(self):
        self.close_dem()
        self.project_path = None

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
        self.postprocess_group.setTitle(labels["postprocess"])
        self.postprocess_button.setText(labels["run_postprocess"])
        self.class_label.setText(labels["class_groups"])
        for key, label in self.post_labels.items():
            label.setText(labels[key])
        self.post_confidence_enabled.setText("Применять" if index == 0 else "Apply")
        self.language_label.setText(labels["language"])
        self.length_label.setText(("Минимальная длина (px)", "Minimum length (px)")[index])
        self.unit_label.setText(labels["units"])
        self.apply_button.setText(labels["apply"])
        self.cancel_button.setText(labels["cancel"])
        self.settings.setValue("language", self.language)
        self.file_menu.setTitle(labels["menu_file"])
        self.view_menu.setTitle(labels["menu_view"])
        self.analysis_menu.setTitle(labels["menu_analysis"])
        self.menuBar().actions()[3].setText(labels["menu_export"])
        self.help_menu.setTitle(labels["menu_help"])
        for action, key in (
            (self.new_project_action, "new_project"), (self.open_project_action, "open_project"),
            (self.save_project_action, "save_project"), (self.save_as_action, "save_as"),
            (self.open_dem_action, "open"), (self.close_dem_action, "close_dem"),
            (self.restore_action, "restore"), (self.zoom_action, "zoom"),
            (self.detect_action, "run"), (self.postprocess_action, "run_postprocess"),
            (self.export_action, "export"),
            (self.layers_toggle_action, "layers"), (self.properties_toggle_action, "properties"),
            (self.reset_panels_action, "reset_panels"), (self.terrain_action, "terrain"),
            (self.compare_action, "compare"), (self.about_action, "about"),
            (self.guide_action, "guide"), (self.shapefile_action, "shapefile"),
            (self.exit_action, "exit"),
        ):
            action.setText(labels[key])
        for tab_index, key in (
            (0, "tab_data"), (self.terrain_tab, "terrain"),
            (self.detection_tab, "tab_detection"), (self.comparison_tab, "compare"),
            (self.post_tab, "tab_post"), (self.stats_tab, "tab_stats"),
            (self.export_tab, "menu_export"),
        ):
            self.property_tabs.setTabText(tab_index, labels[key])
        self.property_tabs.setTabText(self.validation_tab, ("Валидация", "Validation")[index])
        self.export_button.setText(("Экспорт линий", "Export lines")[index])
        self.export_action.setText(self.export_button.text())
        self.resolution_label.setText(("Режим разрешения", "Resolution mode")[index])
        self.resolution_mode.setItemText(0, ("Предпросмотр", "Preview")[index])
        self.resolution_mode.setItemText(1, ("Полное разрешение", "Full resolution")[index])
        self.memory_label.setText(("Бюджет памяти", "Memory budget")[index])
        self.profile_label.setText(("Профиль", "Profile")[index])
        self.apply_profile_button.setText(("Применить", "Apply")[index])
        self.save_profile_button.setText(("Сохранить", "Save")[index])
        self.astar_manual.setText(("Выбрать две точки A*", "Pick two A* endpoints")[index])
        self.response_button.setText(("Показать отклик", "Show response")[index])
        self.orientation_response.setText(("Цвет по направлению", "Color by direction")[index])
        label = ("Компонент (0 = сводный)", "Component (0 = aggregate)")[index]
        self.response_band_label.setText(label)
        self.response_band.setToolTip(label)
        for (algorithm, key), label in self.advanced_labels.items():
            text = CONTROLS[key][index]
            label.setText(text)
            control = self.advanced_controls[algorithm][key]
            control.setToolTip(f"{text} · {control.minimum():g}–{control.maximum():g}")
        self.refresh_profiles()
        self.validation.localize()
        self.comparison.localize()

        selected_algorithm = self.algorithm_box.currentData() or "lsd"
        self.algorithm_box.blockSignals(True)
        self.algorithm_box.clear()
        for algorithm_id, method in METHODS.items():
            name = "Shearlet (экспериментальный)" if index == 0 and algorithm_id == "shearlet" else method.name
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
        if self.last_statistics is not None:
            self._display_statistics(self.last_statistics)
        self.algorithm_changed(self.algorithm_box.currentIndex())

    def algorithm_changed(self, index):
        self.parameter_stack.setCurrentIndex(max(0, index))
        method = self.algorithm_box.currentData()
        self.astar_manual.setVisible(method == "astar")
        self.response_button.setVisible(method in FILTER_METHODS)
        self.orientation_response.setVisible(method in FILTER_METHODS)
        self.response_band.setVisible(method in FILTER_METHODS)
        self.response_band_label.setVisible(method in FILTER_METHODS)
        self.response_band.setValue(0)
        self.view.picking = False
        self.endpoints.clear()

    def refresh_result_choices(self):
        previous = self.result_view_box.currentData() if self.result_view_box.count() else "all"
        self.result_view_box.blockSignals(True)
        self.result_view_box.clear()
        self.result_view_box.addItem(TEXT[self.language]["all_results"], "all")
        for algorithm_id, result in self.results_by_algorithm.items():
            self.result_view_box.addItem(
                f"{ALGORITHM_NAMES[algorithm_id]} ({len(result.segments)})", algorithm_id
            )
        for algorithm_id, result in self.postprocessed_results_by_algorithm.items():
            self.result_view_box.addItem(
                f"{ALGORITHM_NAMES.get(algorithm_id, algorithm_id)} · post ({len(result.segments)})",
                f"post:{algorithm_id}",
            )
        for algorithm, ids in self.unique_candidates.items():
            if ids and algorithm in self.results_by_algorithm:
                title = ("уникальные", "unique")[self.language == "en"]
                self.result_view_box.addItem(
                    f"{ALGORITHM_NAMES[algorithm]} · {title} ({len(ids)})", f"unique:{algorithm}"
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
            self.export_format,
            self.postprocess_button,
            self.postprocess_group,
            self.class_group_list,
            self.validation, self.comparison, self.resolution_mode, self.memory_budget,
            self.astar_manual, self.response_button, self.orientation_response, self.response_band, self.profile_box,
            self.apply_profile_button, self.save_profile_button,
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
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("algorithm", self.algorithm_box.currentData() or "lsd")
        self.settings.setValue("uiParameters", json.dumps(self._ui_parameters()))
        if self.source:
            self.settings.setValue("lastDem", self.source)
        self.settings.sync()
        event.accept()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.load_path(path)

    def load_path(self, path, *, kind='elevation', parameters=None):
        if self.task is not None:
            return False
        path = str(Path(path).expanduser().resolve())
        if not Path(path).is_file():
            self.show_error(TEXT[self.language]["missing_dem"].format(path))
            return False
        self.start_task(
            lambda progress, cancel: self.prepare_layer(path, kind, parameters or {}, cancel, progress),
            self.accept_layer,
        )
        return True

    def prepare_layer(self, path, kind, parameters, cancel, progress):
        preview = load_preview(path)
        with rasterio.open(path) as dataset:
            native_grid = dataset_grid(dataset)
        check_cancel(cancel)
        if kind == "elevation":
            image = preview.values
            analysis_path = path
        else:
            info = Path(path).stat()
            key = json.dumps([str(path), info.st_size, info.st_mtime_ns, kind, parameters],
                             sort_keys=True)
            token = hashlib.sha256(key.encode()).hexdigest()
            target = Path(self.cache.name) / f"{token}.tif"
            if not target.exists():
                process_dem(path, target, kind, progress=progress, cancel=cancel, **parameters)
                # Bound session cache to three source-resolution derivatives.
                cached = sorted(Path(self.cache.name).glob("*.tif"),
                                key=lambda entry: entry.stat().st_mtime_ns)
                for stale in cached[:-3]:
                    if stale != self.analysis_path:
                        stale.unlink(missing_ok=True)
            else:
                target.touch()
            image = load_preview(target).values
            analysis_path = target
        valid = preview.valid & np.isfinite(image)
        gray = np.zeros(image.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(image[valid], [2, 98])
            gray[valid] = np.clip((image[valid] - low) / max(high - low, 1e-12) * 255, 0, 255)
        check_cancel(cancel)
        return str(Path(path).resolve()), kind, parameters, preview, image, gray, native_grid, analysis_path

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
        path, kind, parameters, preview, image, gray, self.native_grid, self.analysis_path = result
        if path != self.source:
            self.clear_experiments()
            self.results_by_algorithm.clear()
            self.postprocessed_results_by_algorithm.clear()
            self.statistics_label.clear()
            self.last_statistics = None
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
        self._refresh_layers()
        crs = self.dem.crs.to_string() if self.dem.crs else "unknown"
        valid_values = self.dem.values[self.dem.valid]
        labels = TEXT[self.language]
        details = f"{Path(path).name}\n{width} × {height}\nCRS: {crs}\n{labels['resolution']}: {self.dem.resolution[0]:g} × {self.dem.resolution[1]:g}"
        if valid_values.size:
            details += f"\n{labels['range']}: {float(valid_values.min()):g} – {float(valid_values.max()):g}"
        self.data_info.setText(details)
        self.statusBar().showMessage(f"{Path(path).name} · {crs} · {width}×{height}")
        self.settings.setValue("lastDem", path)
        if self.pending_project:
            state = self.pending_project
            self.pending_project = None
            results = list(state["detection_results"].values()) + list(state["postprocessed_results"].values())
            preview_grid = (
                self.dem.values.shape, tuple(self.dem.transform)[:6],
                self.dem.crs.to_wkt() if self.dem.crs else None,
            )
            matches = all(
                (value.source_shape, value.source_transform, value.source_crs_wkt)
                in (preview_grid, self.native_grid) for value in results
            )
            if matches:
                self.results_by_algorithm = state["detection_results"]
                self.postprocessed_results_by_algorithm = state["postprocessed_results"]
                self.class_visibility = state["class_visibility"]
                self.project_path = state.get("project_path")
                display = state.get("display", {})
                self.algorithm_box.setCurrentIndex(max(0, self.algorithm_box.findData(display.get("algorithm"))))
                self._apply_ui_parameters(display)
                self._refresh_layers()
                self.refresh_result_choices()
                result_index = self.result_view_box.findData(display.get("result"))
                if result_index >= 0:
                    self.result_view_box.setCurrentIndex(result_index)
                    selected = display.get("result")
                    if isinstance(selected, str) and selected.startswith("post:"):
                        self._refresh_class_groups(self.postprocessed_results_by_algorithm[selected[5:]])
            else:
                self.show_error("Результаты проекта относятся к другой геометрии DEM")
        self.refresh_result_choices()

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, TEXT[self.language]["save_project_title"], "", "Project (*.laproj)")
        if path:
            if not path.endswith(".laproj"):
                path += ".laproj"
            self.project_path = path
            self.save_project()

    def save_project(self):
        if self.dem is None:
            return
        if not self.project_path:
            self.save_project_as()
            return
        try:
            save_project(self.project_path, {
                "dem_path": self.source,
                "detection_results": self.results_by_algorithm,
                "postprocessed_results": self.postprocessed_results_by_algorithm,
                "class_visibility": self.class_visibility,
                "display": {"layer": self.parameters.get("layer", "elevation"),
                            "algorithm": self.algorithm_box.currentData(),
                            "result": self.result_view_box.currentData(),
                            "terrain_parameters": {
                                key: self.parameters[key] for key in ("vertical_scale", "elevation_unit")
                                if key in self.parameters
                            },
                            **self._ui_parameters()},
            })
        except (OSError, TypeError, ValueError) as error:
            self.show_error(str(error))
            return
        self.statusBar().showMessage(TEXT[self.language]["save_ok"])

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, TEXT[self.language]["open_project_title"], "", "Project (*.laproj *.json)")
        if path:
            self.open_project_path(path)

    def open_project_path(self, path):
        try:
            state = load_project(path)
            state["project_path"] = path
            if not Path(state["dem_path"]).is_file():
                raise FileNotFoundError(TEXT[self.language]["missing_dem"].format(state["dem_path"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.show_error(TEXT[self.language]["project_error"].format(error))
            return
        self.pending_project = state
        display = state.get("display", {})
        kind = display.get("layer", "elevation")
        if kind not in TERRAIN_MODES:
            kind = "elevation"
        if not self.load_path(state["dem_path"], kind=kind,
                              parameters=display.get("terrain_parameters", {})):
            self.pending_project = None

    def _ui_parameters(self):
        algorithm = self.algorithm_box.currentData() or "lsd"
        return {
            "detector_parameters": parameters_dict(self._selected_parameters(algorithm)),
            "all_detector_parameters": {
                key: parameters_dict(self._selected_parameters(key)) for key in METHODS
            },
            "resolution_mode": self.resolution_mode.currentData(),
            "memory_mib": self.memory_budget.value(),
            "postprocessing_parameters": parameters_dict(self._postprocess_parameters()),
        }

    def _apply_ui_parameters(self, state):
        algorithm = self.algorithm_box.currentData()
        all_parameters = state.get("all_detector_parameters", {
            algorithm: state.get("detector_parameters", {}),
        })
        for key, parameters in all_parameters.items():
            self.apply_detector_parameters(key, parameters)
        resolution = self.resolution_mode.findData(state.get("resolution_mode"))
        if resolution >= 0:
            self.resolution_mode.setCurrentIndex(resolution)
        if "memory_mib" in state:
            self.memory_budget.setValue(state["memory_mib"])
        post = state.get("postprocessing_parameters", {})
        self.apply_postprocess_parameters(post)

    def apply_detector_parameters(self, algorithm, detector):
        controls = {
            "lsd": {"min_length": self.min_length},
            "canny-hough": {
                "canny_low_threshold": self.canny_low,
                "canny_high_threshold": self.canny_high,
                "aperture_size": self.canny_aperture,
                "l2_gradient": self.canny_l2,
                "hough_threshold": self.hough_votes,
                "min_segment_length_m": self.canny_min_length,
                "max_gap_m": self.canny_max_gap,
            },
            "edlines": {
                "sigma": self.ed_sigma, "gradient_threshold": self.ed_gradient,
                "anchor_threshold": self.ed_anchor, "min_path_length_px": self.ed_path,
                "min_segment_length_m": self.ed_min_length,
                "line_fit_error_threshold": self.ed_line_fit_error,
                "max_error_threshold": self.ed_max_error,
                "max_distance_between_lines_px": self.ed_max_distance,
                "nfa_validation": self.ed_nfa, "pf_mode": self.ed_pf,
                "edge_operator": self.ed_operator, "scan_interval": self.ed_scan,
            },
        }
        for key, control in (controls | self.advanced_controls).get(algorithm, {}).items():
            if key not in detector:
                continue
            value = detector[key]
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                index = control.findText(str(value))
                if index >= 0:
                    control.setCurrentIndex(index)
            else:
                control.setValue(value)
    def apply_postprocess_parameters(self, post):
        post_controls = {
            "max_angle_difference_deg": self.post_angle,
            "max_endpoint_distance_px": self.post_endpoint,
            "max_collinearity_distance_px": self.post_collinearity,
            "max_gap_px": self.post_gap,
            "minimum_length_px": self.post_min_length,
            "maximum_length_px": self.post_max_length,
            "minimum_confidence": self.post_confidence,
            "minimum_orientation_deg": self.post_orientation_min,
            "maximum_orientation_deg": self.post_orientation_max,
        }
        for key, control in post_controls.items():
            value = post.get(key)
            if value is not None:
                control.setValue(value)
        if "merge_enabled" in post:
            self.merge_enabled.setChecked(bool(post["merge_enabled"]))
        if "minimum_confidence" in post:
            self.post_confidence_enabled.setChecked(post["minimum_confidence"] is not None)

    def _selected_parameters(self, algorithm_id):
        if algorithm_id in self.advanced_controls:
            return METHODS[algorithm_id].parameters(**{
                key: control.value() for key, control in self.advanced_controls[algorithm_id].items()
            })
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

    def clear_experiments(self):
        self.view.picking = False
        self.endpoints.clear()
        self.comparison.clear_results()
        self.validation.clear_results()
        self.unique_candidates.clear()

    def analysis_options(self):
        return {
            "resolution": self.resolution_mode.currentData(),
            "memory_mib": self.memory_budget.value(),
            "working_path": self.analysis_path,
            "preview": replace(
                self.dem, values=self.image, valid=self.dem.valid & np.isfinite(self.image),
            ),
        }

    def refresh_profiles(self):
        index = int(self.language == "en")
        selected = self.profile_box.currentData()
        self.profile_box.clear()
        for key, profile in PROFILES.items():
            self.profile_box.addItem(profile[index], ("preset", key))
        for key in self.user_profiles():
            self.profile_box.addItem(key, ("user", key))
        previously = self.profile_box.findData(selected)
        if previously >= 0:
            self.profile_box.setCurrentIndex(previously)

    def user_profiles(self):
        try:
            return json.loads(self.settings.value("profiles", "{}"))
        except (TypeError, ValueError):
            return {}

    def apply_profile(self):
        origin, key = self.profile_box.currentData()
        configurations = PROFILES[key][2] if origin == "preset" else self.user_profiles()[key]
        for algorithm, parameters in configurations.items():
            self.apply_detector_parameters(algorithm, parameters_dict(METHODS[algorithm].parameters(**parameters)))
        self.algorithm_box.setCurrentIndex(self.algorithm_box.findData(next(iter(configurations))))
        for panel in (self.comparison, self.validation):
            for index in range(panel.methods.count()):
                item = panel.methods.item(index)
                matches = item.data(Qt.ItemDataRole.UserRole) in configurations
                item.setCheckState(Qt.CheckState.Checked if matches else Qt.CheckState.Unchecked)

    def save_profile(self):
        title = ("Название профиля", "Profile name")[self.language == "en"]
        name, accepted = QInputDialog.getText(self, title, title)
        if accepted and name.strip():
            profiles = self.user_profiles()
            profiles[name.strip()] = self._ui_parameters()["all_detector_parameters"]
            self.settings.setValue("profiles", json.dumps(profiles, ensure_ascii=False))
            self.refresh_profiles()

    def pick_astar_endpoints(self):
        if self.dem is not None and self.task is None:
            self.endpoints.clear()
            self.view.picking = True
            self.statusBar().showMessage(("Выберите начало и конец на карте", "Pick start and end on the map")[self.language == "en"])

    def accept_astar_point(self, point):
        if self.dem is None or not (0 <= point[0] < self.dem.values.shape[1]
                                   and 0 <= point[1] < self.dem.values.shape[0]):
            return
        if self.resolution_mode.currentData() == "full":
            affine = Affine(*self.native_grid[1])
            x, y = self.dem.transform * tuple(np.asarray(point) + .5)
            point = tuple(np.asarray(~affine * (x, y)) - .5)
        self.endpoints.append(point)
        if len(self.endpoints) == 2:
            self.view.picking = False
            self.run_detection(endpoints=tuple(self.endpoints))
            self.endpoints.clear()

    def show_response(self):
        if self.dem is None:
            return
        algorithm = self.algorithm_box.currentData()
        p = self._selected_parameters(algorithm)
        dem = self.analysis_options()["preview"]
        orientation = self.orientation_response.isChecked()
        band = self.response_band.value()
        if band:
            self.start_task(
                lambda _progress, cancel: detector_component(dem, algorithm, band - 1, p, cancel=cancel),
                self.accept_component,
            )
            return
        self.start_task(
            lambda progress, cancel: (
                detector_response(dem, algorithm, p, progress=progress, cancel=cancel)[1],
                orientation,
            ),
            self.accept_response,
        )

    def accept_component(self, payload):
        angle, coefficient = payload
        valid = np.isfinite(coefficient)
        raster = np.zeros(coefficient.shape, np.uint8)
        if valid.any():
            limit = np.percentile(abs(coefficient[valid]), 98)
            raster[valid] = np.clip(127.5 * (1 + coefficient[valid] / max(limit, 1e-12)), 0, 255)
        self.show_response_raster(raster)
        note = (("Низкие частоты", "Low-pass")[self.language == "en"]
                if angle is None else f"{np.rad2deg(angle):.1f}°")
        self.statusBar().showMessage(f"{self.response_band.value()} · {note}")

    def accept_response(self, payload):
        response, orientation = payload
        if orientation:
            angle = response.tangent_rad
            image = np.stack([(0.5 + .5 * np.cos(2 * angle + phase)) * response.strength
                              for phase in (0, 2 * np.pi / 3, 4 * np.pi / 3)], axis=-1)
            raster = np.ascontiguousarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        else:
            raster = np.ascontiguousarray((response.strength * 255).astype(np.uint8))
        self.show_response_raster(raster)
        self.statusBar().showMessage(self.response_button.text())

    def show_response_raster(self, raster):
        height, width = raster.shape[:2]
        if raster.ndim == 3:
            qimage = QImage(raster.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(raster.data, width, height, width, QImage.Format.Format_Grayscale8)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimage.copy()))
        self.draw_result_overlays()

    def run_detection(self, _checked=False, *, endpoints=None):
        if self.dem is None or self.image is None:
            return
        algorithm_id = self.algorithm_box.currentData()
        parameters = self._selected_parameters(algorithm_id)
        source = self.source
        options = self.analysis_options()
        self.start_task(
            lambda progress, cancel: run_detector(
                source, algorithm_id, parameters, endpoints=endpoints, progress=progress,
                cancel=cancel, **options,
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
        self.comparison.clear_results()
        self.validation.clear_results()
        self.unique_candidates.clear()
        self.results_by_algorithm.pop("consensus", None)
        for key in (result.algorithm_id, "consensus"):
            self.postprocessed_results_by_algorithm.pop(key, None)
        self.results_by_algorithm[result.algorithm_id] = result
        self._refresh_layers()
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
        if selected == "all":
            overlays = [(key, value, False) for key, value in self.results_by_algorithm.items()]
        else:
            value = self.selected_result(selected)
            overlays = [(value.algorithm_id, value, bool(value.postprocessing_parameters))] if value else []
        for algorithm_id, result, is_postprocessed in overlays:
            paths = {}
            colors = {}
            world_to_preview = ~self.dem.transform if self.dem else ~Affine(*result.source_transform)
            for segment in result.segments:
                if is_postprocessed:
                    class_id = str(segment.properties.get("class_id", "unclassified"))
                    if not self.class_visibility.get(class_id, True):
                        continue
                    hue = int.from_bytes(hashlib.sha256(class_id.encode("utf-8")).digest()[:2], "big") % 360
                    color = QColor.fromHsv(hue, 220, 230)
                else:
                    class_id = algorithm_id
                    color = QColor(ALGORITHM_COLORS.get(algorithm_id, Qt.GlobalColor.magenta))
                    if algorithm_id not in ALGORITHM_COLORS:
                        hue = int.from_bytes(hashlib.sha256(algorithm_id.encode()).digest()[:2], "big") % 360
                        color = QColor.fromHsv(hue, 215, 235)
                path = paths.setdefault(class_id, QPainterPath())
                colors[class_id] = color
                path.addPolygon(QPolygonF([
                    QPointF(*(world_to_preview * coordinate)) for coordinate in segment.coordinates
                ]))
            for group_id, path in paths.items():
                pen = QPen(colors[group_id])
                pen.setCosmetic(True)
                self.scene.addPath(path, pen)

    def _postprocess_parameters(self):
        return PostprocessParameters(
            merge_enabled=self.merge_enabled.isChecked(),
            max_angle_difference_deg=self.post_angle.value(),
            max_endpoint_distance_px=self.post_endpoint.value(),
            max_collinearity_distance_px=self.post_collinearity.value(),
            max_gap_px=self.post_gap.value(),
            minimum_length_px=self.post_min_length.value(),
            maximum_length_px=self.post_max_length.value() or None,
            minimum_confidence=self.post_confidence.value()
            if self.post_confidence_enabled.isChecked()
            else None,
            minimum_orientation_deg=self.post_orientation_min.value()
            if self.post_orientation_min.value() >= 0
            else None,
            maximum_orientation_deg=self.post_orientation_max.value()
            if self.post_orientation_max.value() >= 0
            else None,
        )

    def run_postprocessing(self):
        algorithm_id = self.algorithm_box.currentData()
        selected = self.result_view_box.currentData()
        if selected == "consensus":
            algorithm_id = selected
        source = self.results_by_algorithm.get(algorithm_id)
        if source is None and isinstance(selected, str) and selected.startswith("post:"):
            source = self.postprocessed_results_by_algorithm.get(selected[5:])
        if source is None:
            return
        parameters = self._postprocess_parameters()

        def operation(progress, cancel):
            processed = postprocess_result(source, parameters, cancel=cancel)
            stats = summarize_results(processed)
            progress(100)
            return processed, stats

        self.start_task(operation, self.accept_postprocessing)

    def accept_postprocessing(self, response):
        if self.closing:
            return
        result, stats = response
        self.last_statistics = stats
        self.postprocessed_results_by_algorithm[result.algorithm_id] = result
        self._refresh_layers()
        self.class_visibility = {
            str(segment.properties.get("class_id", "unclassified")): self.class_visibility.get(
                str(segment.properties.get("class_id", "unclassified")), True
            )
            for segment in result.segments
        }
        self._refresh_class_groups(result)
        self._display_statistics(stats)
        self.refresh_result_choices()
        index = self.result_view_box.findData(f"post:{result.algorithm_id}")
        if index >= 0:
            self.result_view_box.setCurrentIndex(index)

    def _display_statistics(self, stats):
        labels = TEXT[self.language]
        self.statistics_label.setText(
            f"{labels['stat_count']}: {stats['object_count']}  "
            f"{labels['stat_total_length']}: {stats['total_length_m']} m  "
            f"{labels['stat_density']}: {stats['density_per_km2']}\n"
            f"{labels['stat_rose']}: {stats['rose_counts']}\n"
            f"{labels['stat_classes']}: {stats['class_statistics']}"
        )

    def _refresh_class_groups(self, result):
        classes = sorted({
            str(segment.properties.get("class_id", "unclassified"))
            for segment in result.segments
        })
        self.class_group_list.blockSignals(True)
        self.class_group_list.clear()
        for class_id in classes:
            hue = int.from_bytes(hashlib.sha256(class_id.encode("utf-8")).digest()[:2], "big") % 360
            item = QListWidgetItem(class_id)
            item.setForeground(QColor.fromHsv(hue, 220, 230))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if self.class_visibility.get(class_id, True)
                else Qt.CheckState.Unchecked
            )
            self.class_group_list.addItem(item)
        self.class_group_list.blockSignals(False)

    def class_visibility_changed(self, item):
        self.class_visibility[item.text()] = item.checkState() == Qt.CheckState.Checked
        self.draw_result_overlays()

    def selected_result(self, selection):
        if isinstance(selection, str) and selection.startswith("post:"):
            return self.postprocessed_results_by_algorithm.get(selection[5:])
        if isinstance(selection, str) and selection.startswith("unique:"):
            algorithm = selection[7:]
            result = self.results_by_algorithm.get(algorithm)
            if result:
                ids = set(self.unique_candidates.get(algorithm, ()))
                return replace(result, segments=tuple(
                    line for i, line in enumerate(result.segments)
                    if line.properties.get("object_id", i) in ids
                ))
            return None
        return self.results_by_algorithm.get(selection)

    def save_geojson(self):
        if self.dem is None or not (self.results_by_algorithm or self.postprocessed_results_by_algorithm):
            return
        title = self.export_format.currentText()
        file_filter, exporter = {
            "GeoJSON": ("GeoJSON (*.geojson)", export_geojson),
            "Shapefile": ("Shapefile (*.shp)", export_shapefile),
            "GeoPackage": ("GeoPackage (*.gpkg)", export_geopackage),
        }[title]
        path, _ = QFileDialog.getSaveFileName(self, title, "", file_filter)
        if path:
            selected = self.result_view_box.currentData()
            results = (
                list(self.results_by_algorithm.values()) + list(self.postprocessed_results_by_algorithm.values())
                if selected == "all"
                else [self.selected_result(selected)]
            )
            dem = self.dem
            self.start_task(
                lambda progress, cancel: exporter(path, dem, results),
                self.export_finished,
            )

    def export_finished(self, result):
        self.statusBar().showMessage(TEXT[self.language]["saved"])
