"""Installed and frozen application smoke test, including GDAL/PROJ and Qt plugins."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_origin

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
        target = root / "lines.geojson"
        export_geojson(target, dem, lines)
        assert len(json.loads(target.read_text())["features"]) == len(lines)
        window = MainWindow()
        window.show()
        window.close()
    return 0
