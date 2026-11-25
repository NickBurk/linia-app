import json

import cv2
import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS

from lineament_analyzer.algorithms.lsd import LsdParameters, detect
from lineament_analyzer.data.dem import Dem
from lineament_analyzer.export.geojson import export_geojson


@pytest.mark.parametrize("end", [(110, 20), (20, 110), (110, 110)])
def test_known_line(end):
    image = np.zeros((128, 128), np.uint8)
    cv2.line(image, (20, 20), end, 255, 3)
    lines = detect(image, np.ones_like(image, bool), LsdParameters(50))
    assert len(lines) >= 2
    direction = np.array(end) - 20
    expected = np.arctan2(direction[1], direction[0]) % np.pi
    angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]) % np.pi
    difference = np.abs((angles - expected + np.pi / 2) % np.pi - np.pi / 2)
    assert np.min(difference) < np.radians(2)
    assert np.all(np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]) >= 50)
    assert np.all((lines >= 0) & (lines < 128))


def test_mask_boundary_does_not_create_lines():
    image = np.ones((128, 128)) * 100
    image[10:40] = 200
    valid = np.ones_like(image, bool)
    valid[:, 60:70] = False
    lines = detect(image, valid, LsdParameters(20))
    for x1, y1, x2, y2 in lines:
        assert not (min(x1, x2) <= 70 and max(x1, x2) >= 60)
        assert abs(x2 - x1) > abs(y2 - y1)


@pytest.mark.parametrize("epsg", [32633, 4326, 2277])
def test_export_coordinates_attributes(tmp_path, epsg):
    t = (
        Affine(0.01, 0.002, 10, 0.001, -0.01, 50)
        if epsg == 4326
        else (Affine(10, 2, 500000, 1, -10, 4000000))
    )
    dem = Dem(np.zeros((50, 50)), np.ones((50, 50), bool), t, CRS.from_epsg(epsg), None, "m")
    segments = np.array([[1.25, 2.5, 20.5, 30.25]])
    path = tmp_path / "result.geojson"
    export_geojson(path, dem, segments, parameters={"min_length": 10})
    data = json.loads(path.read_text())
    assert "crs" not in data
    assert data["source_crs"] == dem.crs.to_wkt()
    feature = data["features"][0]
    expected = []
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    for x, y in segments.reshape(2, 2):
        expected.append(transformer.transform(*(t * (x + 0.5, y + 0.5))))
    np.testing.assert_allclose(feature["geometry"]["coordinates"], expected)
    properties = feature["properties"]
    assert properties["length_m"] > 0
    assert 0 <= properties["orientation_deg"] < 180
    assert properties["parameters"]["min_length"] == 10
    np.testing.assert_allclose(properties["length_px"], np.hypot(19.25, 27.75))


def test_invalid_export_preserves_file(tmp_path):
    dem = Dem(np.zeros((5, 5)), np.ones((5, 5), bool), Affine.identity(), CRS.from_epsg(4326), None)
    path = tmp_path / "result.geojson"
    path.write_text("original")
    with pytest.raises(ValueError):
        export_geojson(path, dem, np.array([[0, 0, np.nan, 1]]))
    assert path.read_text() == "original"
