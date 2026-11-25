"""Localized comparison and scientific-validation widgets."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lineament_analyzer.engine import METHODS
from lineament_analyzer.validation.datasets import (
    FOLDS,
    Dataset,
    generate_synthetic,
    reference_layers,
)
from lineament_analyzer.validation.experiment import run_experiment
from lineament_analyzer.validation.geometry import MatchParameters
from lineament_analyzer.validation.report import export_report

ACCURACY_MISSING = (
    "Недостаточно эталонных данных для расчёта метрик точности",
    "Insufficient reference data to calculate accuracy metrics",
)


class ExperimentPanel(QWidget):
    def __init__(self, window, scientific):
        super().__init__(window)
        self.window = window
        self.scientific = scientific
        self.report = None
        self.dataset = None
        self.reference = None
        self.reference_layer = None
        self.crs = None
        self.methods = QListWidget()
        self.methods.setMaximumHeight(140)
        for algorithm, method in METHODS.items():
            item = QListWidgetItem(method.name, self.methods)
            item.setData(Qt.ItemDataRole.UserRole, algorithm)
            item.setCheckState(Qt.CheckState.Checked if algorithm in ("lsd", "edlines") else Qt.CheckState.Unchecked)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.01, 10000)
        self.tolerance.setValue(20)
        self.tolerance.setSuffix(" m")
        self.angle = QDoubleSpinBox()
        self.angle.setRange(0, 90)
        self.angle.setValue(20)
        self.angle.setSuffix("°")
        self.angle_label = QLabel()
        self.tolerance_label = QLabel()
        self.consensus = QCheckBox()
        self.consensus.setChecked(True)
        self.post = QCheckBox()
        self.run = QPushButton()
        self.run.clicked.connect(self.start)
        self.view_report = QPushButton()
        self.view_report.setEnabled(False)
        self.view_report.clicked.connect(self.show_report)
        self.save_report = QPushButton()
        self.save_report.setEnabled(False)
        self.save_report.clicked.connect(self.save)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout = QVBoxLayout(self)
        if scientific:
            self.open_dem = QPushButton()
            self.open_dem.clicked.connect(window.open_file)
            self.pick_reference = QPushButton()
            self.pick_reference.clicked.connect(self.choose_reference)
            self.save_dataset = QPushButton()
            self.save_dataset.clicked.connect(self.choose_save_dataset)
            self.open_dataset = QPushButton()
            self.open_dataset.clicked.connect(self.choose_dataset)
            self.generate = QPushButton()
            self.generate.clicked.connect(self.generate_dataset)
            self.reference_note = QLabel()
            self.reference_note.setWordWrap(True)
            self.fold = QComboBox()
            for name in FOLDS:
                self.fold.addItem(name, name)
            self.fold.setCurrentIndex(1)
            for widget in (self.open_dem, self.pick_reference, self.reference_note,
                           self.fold, self.open_dataset, self.save_dataset, self.generate):
                layout.addWidget(widget)
        layout.addWidget(self.methods)
        form = QFormLayout()
        form.addRow(self.tolerance_label, self.tolerance)
        form.addRow(self.angle_label, self.angle)
        layout.addLayout(form)
        for widget in (self.consensus, self.post, self.run, self.view_report,
                       self.save_report, self.message):
            layout.addWidget(widget)
        layout.addStretch(1)

    def localize(self):
        i = int(self.window.language == "en")
        self.run.setText(("Запустить валидацию", "Run validation")[i] if self.scientific else
                         ("Запустить выбранные методы", "Run selected methods")[i])
        self.tolerance_label.setText(("Допуск совпадения", "Matching tolerance")[i])
        self.angle_label.setText(("Угловой допуск", "Orientation tolerance")[i])
        self.post.setText(("Добавить постобработку", "Include postprocessing")[i])
        self.consensus.setText(("Рассчитать консенсус", "Calculate consensus")[i])
        self.view_report.setText(("Таблица и графики", "Table and charts")[i])
        self.save_report.setText(("Экспорт отчёта", "Export report")[i])
        if self.scientific:
            self.open_dem.setText(("Выбрать DEM", "Choose DEM")[i])
            self.pick_reference.setText(("Выбрать эталон", "Choose reference")[i])
            self.save_dataset.setText(("Сохранить набор валидации", "Save validation dataset")[i])
            self.open_dataset.setText(("Открыть набор валидации", "Open validation dataset")[i])
            self.generate.setText(("Создать синтетический набор", "Generate synthetic dataset")[i])
            for index, label in enumerate((
                ("Калибровка", "Calibration"), ("Валидация", "Validation"),
                ("Независимый тест", "Independent test"),
            )):
                self.fold.setItemText(index, label[i])
            self.reference_note.setText(self.reference or ACCURACY_MISSING[i])

    def clear_results(self):
        self.report = None
        self.view_report.setEnabled(False)
        self.save_report.setEnabled(False)
        self.message.clear()

    def choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.pick_reference.text(), "", "Vector (*.geojson *.json *.shp *.gpkg)"
        )
        if not path:
            return
        try:
            layers = reference_layers(path)
            if not layers:
                raise ValueError("Reference file contains no vector layers")
            selected = layers[0]
            if len(layers) > 1:
                selected, ok = QInputDialog.getItem(self, self.pick_reference.text(),
                                                     "Layer", layers, editable=False)
                if not ok:
                    return
            import fiona

            with fiona.open(path, layer=selected) as collection:
                crs = collection.crs_wkt
            confirmed = None
            if not crs:
                confirmed, ok = QInputDialog.getText(self, "CRS", "EPSG or WKT")
                if not ok or not confirmed.strip():
                    return
            self.reference, self.reference_layer, self.crs = path, selected, confirmed
            self.dataset = None
            self.localize()
        except (OSError, ValueError, RuntimeError) as error:
            self.window.show_error(str(error))

    def choose_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, self.open_dataset.text(), "", "JSON (*.json)")
        if path:
            try:
                self.accept_dataset(Dataset.load(path))
            except (OSError, ValueError, TypeError) as error:
                self.window.show_error(str(error))

    def accept_dataset(self, dataset):
        if not self.window.load_path(dataset.dem_path):
            return
        self.dataset = dataset
        self.reference = dataset.reference_path
        self.reference_layer = dataset.reference_layer
        self.crs = dataset.confirmed_crs
        self.fold.setCurrentIndex(FOLDS.index(dataset.fold))
        self.localize()

    def choose_save_dataset(self):
        if not self.reference or not self.window.source:
            return
        path, _ = QFileDialog.getSaveFileName(self, self.save_dataset.text(), "", "JSON (*.json)")
        if path:
            try:
                dataset = self.current_dataset()
                dataset.save(Path(path).with_suffix(".json"))
                self.dataset = dataset
            except (ValueError, OSError) as error:
                self.window.show_error(str(error))

    def current_dataset(self):
        fold = self.fold.currentData()
        previous = self.dataset
        if previous and previous.dem_path == self.window.source and previous.fold == fold:
            return previous
        return Dataset(self.window.source, self.reference,
                       previous.region if previous else Path(self.window.source).stem,
                       fold, self.reference_layer, self.crs,
                       synthetic=bool(previous and previous.synthetic))

    def generate_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, self.generate.text())
        if not directory:
            return
        fold = self.fold.currentData()
        size, ok = QInputDialog.getInt(self, self.generate.text(), "Pixels / px", 256, 64, 2048)
        if not ok:
            return
        root = Path(directory) / f"lineament-synthetic-{fold}"
        if root.exists():
            self.window.show_error(f"Dataset directory already exists: {root}")
            return
        generated = []
        self.window.start_task(
            lambda progress, cancel: generate_synthetic(root, size=size, fold=fold,
                                                        progress=progress, cancel=cancel),
            generated.append,
        )
        if self.window.task is not None:
            self.window.task.finished.connect(
                lambda: self.accept_dataset(generated[0]) if generated else None
            )

    def start(self):
        source = self.window.source
        if not source or self.window.task is not None:
            return
        configurations = {}
        for row in range(self.methods.count()):
            item = self.methods.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                algorithm = item.data(Qt.ItemDataRole.UserRole)
                configurations[algorithm] = self.window._selected_parameters(algorithm)
        if not configurations:
            self.message.setText(("Выберите алгоритм", "Select an algorithm")[self.window.language == "en"])
            return
        dataset = self.current_dataset() if self.scientific and self.reference else None
        matching = MatchParameters(self.tolerance.value(), self.angle.value())
        post = self.window._postprocess_parameters() if self.post.isChecked() else None
        agreement = self.consensus.isChecked()
        options = self.window.analysis_options()
        self.window.start_task(
            lambda progress, cancel: run_experiment(
                dataset, source, configurations, matching, postprocessing=post,
                consensus=agreement, progress=progress, cancel=cancel, **options,
            ),
            self.accept,
        )

    def accept(self, response):
        raw, post, unique, self.report = response
        self.report["unique_candidates"] = unique
        self.window.results_by_algorithm.update(raw)
        self.window.postprocessed_results_by_algorithm.update({
            result.algorithm_id: result for result in post.values()
        })
        self.window._refresh_layers()
        self.window.refresh_result_choices()
        i = int(self.window.language == "en")
        texts = []
        if not self.report["dataset"] and self.scientific:
            texts.append(ACCURACY_MISSING[i])
        if "consensus" in raw:
            texts.append(f"Consensus · {len(raw['consensus'].segments)}")
        for label, ids in unique.items():
            texts.append(f"{label} · {('уникальных', 'unique')[i]} {len(ids)}")
        self.message.setText("\n".join(texts))
        self.view_report.setEnabled(True)
        self.save_report.setEnabled(True)

    def show_report(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.view_report.text())
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        rows = self.report["results"]
        keys = list(dict.fromkeys(key for row in rows for key in (*row["metrics"], *row["measurement"])))
        table = QTableWidget(len(rows), len(keys) + 1)
        table.setHorizontalHeaderLabels(["Algorithm", *keys])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row_index, row in enumerate(rows):
            values = row["metrics"] | row["measurement"]
            cells = [row["label"], *(values.get(key) for key in keys)]
            for column, value in enumerate(cells):
                label = f"{value:.4g}" if isinstance(value, float) else str(value if value is not None else "—")
                table.setItem(row_index, column, QTableWidgetItem(label))
        table.resizeColumnsToContents()
        tabs.addTab(table, ("Таблица", "Table")[self.window.language == "en"])
        if self.report["dataset"]:
            tabs.addTab(AccuracyChart(rows), "Precision / Recall / F1")
            details = QWidget()
            details_layout = QVBoxLayout(details)
            picker = QComboBox()
            picker.addItems([row["label"] for row in rows])
            chart = SpatialChart(rows[0])
            picker.currentIndexChanged.connect(lambda i: chart.select(rows[i]))
            details_layout.addWidget(picker)
            details_layout.addWidget(chart)
            tabs.addTab(details, ("Карта и направления", "Map and orientations")[self.window.language == "en"])
        dialog.resize(1050, 580)
        dialog.exec()

    def save(self):
        path, selected = QFileDialog.getSaveFileName(
            self, self.save_report.text(), "", "HTML (*.html);;JSON (*.json);;CSV (*.csv)"
        )
        if path:
            if not Path(path).suffix:
                path += "." + selected.split(".")[-1].rstrip(")")
            report = self.report
            self.window.start_task(lambda _progress, _cancel: export_report(path, report),
                                   self.window.export_finished)


class AccuracyChart(QWidget):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = ("#347daa", "#57834c", "#9b5849")
        metrics = ("precision", "recall", "f1")
        for column, (label, color) in enumerate(zip(metrics, colors, strict=True)):
            painter.setPen(QColor(color))
            painter.drawText(170 + column * 130, 24, label)
        height = max(1, (self.height() - 48) / len(self.rows))
        width = max(1, self.width() - 190)
        for index, row in enumerate(self.rows):
            offset = 36 + height * index
            painter.setPen(self.palette().text().color())
            painter.drawText(6, round(offset + 12), row["label"])
            for column, (key, color) in enumerate(zip(metrics, colors, strict=True)):
                value = row["metrics"].get(key, 0)
                painter.fillRect(QRectF(165, offset + column * height / 3, width * value,
                                        max(1, height / 3 - 2)), QColor(color))


class SpatialChart(QWidget):
    def __init__(self, row):
        super().__init__()
        self.row = row

    def select(self, row):
        self.row = row
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        row = self.row
        found, expected = row["detected_coordinates"], row["reference_coordinates"]
        matched_found = {pair["detected"] for pair in row["matches"]}
        matched_expected = {pair["reference"] for pair in row["matches"]}
        all_xy = np.asarray([point for path in (*found, *expected) for point in path])
        if all_xy.size:
            lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
            scale = min((self.width() * .63 - 30) / max(upper[0] - lower[0], 1e-12),
                        (self.height() - 70) / max(upper[1] - lower[1], 1e-12))
            for lines, matched, colors in (
                (expected, matched_expected, ("#7187ca", "#8a57a6")),
                (found, matched_found, ("#40a877", "#d85b39")),
            ):
                for i, points in enumerate(lines):
                    coordinates = (np.asarray(points) - lower) * scale
                    path = QPainterPath(QPointF(14 + coordinates[0, 0], self.height() - 24 - coordinates[0, 1]))
                    for x, y in coordinates[1:]:
                        path.lineTo(14 + x, self.height() - 24 - y)
                    painter.setPen(QPen(QColor(colors[i not in matched]), 2))
                    painter.drawPath(path)
        x0 = round(self.width() * .65)
        painter.setPen(self.palette().text().color())
        painter.drawText(12, 22, "TP / FP / FN · CRS")
        painter.drawText(x0, 22, "Axial azimuth · 0–180°")
        center = QPointF(x0 + (self.width() - x0) / 2, self.height() * .30)
        radius = min(self.height() * .19, (self.width() - x0) * .39)
        painter.drawEllipse(center, radius, radius)
        roses = (row["reference_rose"], row["detected_rose"])
        maximum = max(1, *(count for rose in roses for count in rose))
        for rose, color in zip(roses, ("#8a57a6", "#40a877"), strict=True):
            painter.setPen(QPen(QColor(color), 3))
            for index, count in enumerate(rose):
                angle = np.deg2rad(index * 15 + 7.5)
                delta = QPointF(radius * count / maximum * np.sin(angle),
                                -radius * count / maximum * np.cos(angle))
                painter.drawLine(center - delta, center + delta)
        errors = row["orientation_errors_deg"]
        painter.setPen(self.palette().text().color())
        painter.drawText(x0, round(self.height() * .58), "Axial orientation errors · °")
        bins = np.histogram(errors, bins=np.arange(0, 91, 10))[0]
        width = (self.width() - x0 - 20) / len(bins)
        scale = self.height() * .28 / max(1, max(bins))
        bottom = self.height() - 18
        for index, count in enumerate(bins):
            painter.fillRect(QRectF(x0 + index * width, bottom - count * scale,
                                    max(1, width - 2), count * scale), QColor("#347daa"))
