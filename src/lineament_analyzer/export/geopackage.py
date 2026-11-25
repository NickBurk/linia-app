"""Export polyline candidates and full provenance into a GeoPackage."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona

from lineament_analyzer.algorithms.results import DetectionResult
from lineament_analyzer.export.geometry import validate_coordinates, validate_result_grid


def export_geopackage(path, dem, results):
    if dem.crs is None:
        raise ValueError("A source CRS is required for GeoPackage export")
    if isinstance(results, DetectionResult):
        results = (results,)
    target = Path(path).with_suffix(".gpkg")
    schema = {"geometry": "LineString", "properties": {
        "object_id": "str", "algorithm": "str", "stage": "str", "class_id": "str",
        "length_m": "float", "confidence": "float", "metadata": "str",
    }}
    records = []
    for result in results:
        validate_result_grid(result, dem)
        for index, line in enumerate(result.segments, start=1):
            coords = validate_coordinates(line.coordinates)
            metadata = {"algorithm_parameters": dict(result.parameters),
                        "postprocessing_parameters": dict(result.postprocessing_parameters),
                        "analysis": dict(result.analysis_metadata), "dem": str(dem.path),
                        "properties": dict(line.properties)}
            records.append({
                "geometry": {"type": "LineString", "coordinates": coords.tolist()},
                "properties": {
                    "object_id": str(line.properties.get("object_id", f"{result.algorithm_id}-{index}")),
                    "algorithm": result.algorithm_id,
                    "stage": "postprocessed" if result.postprocessing_parameters else "raw",
                    "class_id": line.properties.get("class_id"), "length_m": line.length_m,
                    "confidence": line.confidence,
                    "metadata": json.dumps(metadata, ensure_ascii=False, allow_nan=False),
                },
            })
    with TemporaryDirectory(dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        with fiona.open(temporary, "w", driver="GPKG", layer="lineaments",
                        crs_wkt=dem.crs.to_wkt(), schema=schema) as collection:
            collection.writerecords(records)
        temporary.replace(target)
