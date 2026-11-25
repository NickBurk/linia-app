"""Exercise the installed or frozen desktop application with self-contained data."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import fiona
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from lineament_analyzer.core.windowed import process_dem
from lineament_analyzer.data.dem import load_dem
from lineament_analyzer.engine import METHODS
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.export.geopackage import export_geopackage
from lineament_analyzer.export.shapefile import export_shapefile
from lineament_analyzer.gui.window import MainWindow
from lineament_analyzer.validation.datasets import generate_synthetic
from lineament_analyzer.validation.report import export_report


def _wait(window, errors):
    deadline = monotonic() + 75
    while window.task is not None and monotonic() < deadline:
        QApplication.processEvents()
        sleep(0.002)
    assert window.task is None, "Timed out during an application task"
    assert not errors, errors


def _window(errors):
    window = MainWindow()
    window.show_error = errors.append
    window.show_notice = errors.append
    window.show()
    QApplication.processEvents()
    assert window.isVisible()
    return window


def restore_in_new_process(project):
    """Verify a prior process's project with a fresh QApplication and settings."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(project.parent))
    errors = []
    restored = _window(errors)
    restored.open_project_path(project)
    _wait(restored, errors)
    assert Path(restored.source).samefile(project.parent / "input" / "synthetic.tif")
    assert set(METHODS).issubset(restored.results_by_algorithm)
    assert restored.results_by_algorithm["lsd"].segments
    assert restored.parameters["layer"] == "hillshade"
    restored.close()
    QApplication.processEvents()
    return 0


def run():
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        dataset = generate_synthetic(root / "input", size=192, seed=7)
        second = generate_synthetic(root / "second", size=128, seed=91)
        for kind in ("slope", "aspect", "tpi"):
            target = root / f"{kind}.tif"
            process_dem(dataset.dem_path, target, kind, block_size=23)
            assert np.isfinite(load_dem(target).values).any()

        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root))
        errors = []
        window = _window(errors)
        window.new_project()
        assert window.source is None
        for source in (dataset.dem_path, dataset.dem_path, second.dem_path, dataset.dem_path):
            assert window.load_path(source)
            _wait(window, errors)
            assert window.source == source

        window.layer.setCurrentIndex(1)
        _wait(window, errors)
        assert window.parameters["layer"] == "hillshade"
        assert np.isfinite(window.image).any()
        window.layer.setCurrentIndex(0)
        _wait(window, errors)
        assert window.parameters["layer"] == "elevation"

        for method in METHODS:
            window.algorithm_box.setCurrentIndex(window.algorithm_box.findData(method))
            window.run_detection()
            _wait(window, errors)
            assert window.results_by_algorithm[method].segments, method
        assert set(window.results_by_algorithm) == set(METHODS)
        window.algorithm_box.setCurrentIndex(window.algorithm_box.findData("lsd"))
        window.run_postprocessing()
        _wait(window, errors)
        assert window.postprocessed_results_by_algorithm["lsd"].segments

        for panel in (window.comparison, window.validation):
            for i in range(panel.methods.count()):
                item = panel.methods.item(i)
                item.setCheckState(Qt.CheckState.Checked if i < 3 else Qt.CheckState.Unchecked)
        window.comparison.start()
        _wait(window, errors)
        assert "consensus" in window.results_by_algorithm and window.comparison.report
        window.validation.accept_dataset(dataset)
        _wait(window, errors)
        window.validation.post.setChecked(True)
        window.validation.start()
        _wait(window, errors)
        report = window.validation.report
        assert report and report["dataset"]["synthetic"]
        assert all("f1" in row["metrics"] for row in report["results"])
        export_report(root / "validation.html", report)

        result = window.results_by_algorithm["lsd"]
        geojson = root / "lines.geojson"
        export_geojson(geojson, window.dem, result)
        assert len(json.loads(geojson.read_text(encoding="utf-8"))["features"]) == len(result.segments)
        for filename, writer in (("lines.shp", export_shapefile), ("lines.gpkg", export_geopackage)):
            target = root / filename
            writer(target, window.dem, result)
            with fiona.open(target) as vector:
                assert len(vector) == len(result.segments) and vector.crs

        window.layer.setCurrentIndex(1)
        _wait(window, errors)
        project = root / "self-test.laproj"
        window.project_path = str(project)
        window.save_project()
        assert project.is_file()
        window.close()
        QApplication.processEvents()
        assert not window.isVisible()

        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.extend(("-m", "lineament_analyzer.app"))
        command.extend(("--self-test-restore", str(project)))
        subprocess.run(
            command, check=True, timeout=90,
            env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
        )
    return 0
