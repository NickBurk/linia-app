import json

import numpy as np
import pytest
import shapefile
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.postprocessing import PostprocessParameters, postprocess_result
from lineament_analyzer.algorithms.results import build_detection_result
from lineament_analyzer.data.dem import Dem
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.export.shapefile import export_shapefile


def test_shapefile_roundtrip_preserves_crs_and_postprocessing_attributes(tmp_path):
    dem = Dem(
        np.zeros((80, 80)),
        np.ones((80, 80), dtype=bool),
        Affine(10, 2, 500000, 1, -10, 4500000),
        CRS.from_epsg(32633),
        None,
    )
    raw = build_detection_result(
        dem,
        [[10, 20, 30, 20], [32, 20, 50, 20]],
        algorithm_id="canny-hough",
        parameters={"threshold": 8},
        confidences=[0.75, None],
    )
    result = postprocess_result(
        raw,
        PostprocessParameters(max_endpoint_distance_px=5, max_gap_px=5),
    )
    path = tmp_path / "lines.shp"
    export_shapefile(path, dem, result)

    reader = shapefile.Reader(str(path), encoding="utf-8")
    assert len(reader) == 1
    fields = [field[0] for field in reader.fields[1:]]
    assert all(len(field) <= 10 for field in fields)
    record = reader.record(0).as_dict()
    assert record["source_alg"] == "canny-hough"
    assert record["object_id"] == "canny-hough-000001"
    assert record["length_m"] == pytest.approx(result.segments[0].length_m, abs=0.001)
    assert record.get("class_id")
    assert (tmp_path / "lines.prj").read_text(encoding="utf-8").startswith("PROJCS[")
    assert (tmp_path / "lines.cpg").read_text() == "UTF-8"
    metadata = json.loads((tmp_path / "lines.metadata.json").read_text(encoding="utf-8"))
    assert metadata["features"][0]["detection_parameters"] == {"threshold": 8}
    assert metadata["features"][0]["postprocessing_parameters"]["max_gap_px"] == 5

    geojson = tmp_path / "lines.geojson"
    export_geojson(geojson, dem, result)
    feature = json.loads(geojson.read_text(encoding="utf-8"))["features"][0]
    props = feature["properties"]
    assert props["source_algorithm"] == "canny-hough"
    assert props["postprocessing_parameters"]["max_gap_px"] == 5
    assert props["object_id"] == record["object_id"]
