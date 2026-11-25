import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lineament_analyzer.algorithms.lsd import LsdParameters, detect
from lineament_analyzer.core.terrain import derivative
from lineament_analyzer.data.dem import load_dem
from lineament_analyzer.export.geojson import export_geojson


def test_dem_derivatives_and_lsd(tmp_path):
    path = tmp_path / "dem.tif"
    y, x = np.mgrid[:80, :80]
    values = (x + y + (x > 40) * 100).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", width=80, height=80, count=1,
                       dtype="float32", crs="EPSG:32633", transform=from_origin(0, 800, 10, 10)) as dst:
        dst.write(values, 1)
    dem = load_dem(path)
    assert str(dem.crs) == "EPSG:32633"
    for kind in ("hillshade", "slope", "aspect", "tpi"):
        result = derivative(dem, kind)
        assert result.shape == values.shape
        assert np.isfinite(result).all()
    lines = detect(values, dem.valid, LsdParameters(min_length=15))
    assert len(lines) > 0
    assert np.isfinite(lines).all()
    output = tmp_path / "lines.geojson"
    export_geojson(output, dem, lines[:1])
    feature = json.loads(output.read_text())["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert abs(feature["geometry"]["coordinates"][0][0]) <= 180


def test_gui_language():
    from PySide6.QtWidgets import QApplication

    from lineament_analyzer.gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MainWindow()
    assert window.open_button.text() == "Открыть GeoTIFF"
    window.change_language(1)
    assert window.open_button.text() == "Open GeoTIFF"
    window.close()
