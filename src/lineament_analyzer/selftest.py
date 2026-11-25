"""Installed and frozen application smoke test, including GDAL/PROJ and Qt plugins."""

import json
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from PySide6.QtWidgets import QApplication
from rasterio.transform import from_origin

from lineament_analyzer.algorithms.canny_hough import (
    CannyHoughDetector,
    CannyHoughParameters,
)
from lineament_analyzer.algorithms.edlines import EdLinesDetector, EdLinesParameters
from lineament_analyzer.algorithms.lsd import LsdParameters, detect
from lineament_analyzer.core.windowed import process_dem
from lineament_analyzer.data.dem import load_dem
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.gui.window import MainWindow


def run():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "dem.tif"
        y, x = np.indices((80, 80))
        z = (x + y + (x > 40) * 100).astype(float)
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            width=80,
            height=80,
            count=1,
            dtype="float64",
            crs="EPSG:32633",
            transform=from_origin(500000, 5000000, 10, 10),
        ) as dst:
            dst.write(z, 1)
            dst.set_band_unit(1, "m")
        for kind in ("slope", "aspect", "hillshade", "tpi"):
            result = root / f"{kind}.tif"
            process_dem(source, result, kind, block_size=17)
            assert np.isfinite(load_dem(result).values[10:-10, 10:-10]).all()
        dem = load_dem(source)
        lines = detect(z, dem.valid, LsdParameters())
        assert len(lines) > 0
        synthetic = np.zeros((80, 80), dtype=float)
        synthetic[38:43, 8:72] = 100
        detection_dem = replace(dem, values=synthetic)
        canny_result = CannyHoughDetector().run(
            detection_dem,
            CannyHoughParameters(hough_threshold=10, min_segment_length_m=100),
        )
        edlines_result = EdLinesDetector().run(
            detection_dem,
            EdLinesParameters(gradient_threshold=12, min_segment_length_m=100),
        )
        assert canny_result.segments
        assert edlines_result.segments
        target = root / "lines.geojson"
        export_geojson(target, dem, lines)
        assert len(json.loads(target.read_text())["features"]) == len(lines)
        window = MainWindow()
        window.show()
        for candidate in (source, source):
            assert window.load_path(candidate)
            deadline = time.monotonic() + 15
            while window.task is not None and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.002)
            assert window.task is None and window.source == str(source.resolve())
        second = root / "dem-second.tif"
        with rasterio.open(
            second, "w", driver="GTiff", width=80, height=80, count=1,
            dtype="float64", crs="EPSG:32633",
            transform=from_origin(501000, 5000000, 10, 10),
        ) as dst:
            dst.write(z + 5, 1)
        assert window.load_path(second)
        deadline = time.monotonic() + 15
        while window.task is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.002)
        assert window.task is None and window.source == str(second.resolve())
        window.close()
    return 0
