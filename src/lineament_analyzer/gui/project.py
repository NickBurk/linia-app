"""Versioned JSON persistence for GUI projects and detection results."""

import json
from pathlib import Path

from lineament_analyzer.algorithms.results import DetectionResult, LineSegment

PROJECT_VERSION = 1


def _result_to_dict(result):
    return {
        "algorithm_id": result.algorithm_id,
        "parameters": dict(result.parameters),
        "source_crs_wkt": result.source_crs_wkt,
        "source_transform": list(result.source_transform),
        "source_shape": list(result.source_shape),
        "postprocessing_parameters": dict(result.postprocessing_parameters),
        "segments": [
            {
                "pixel_start": list(line.pixel_start),
                "pixel_end": list(line.pixel_end),
                "source_start": list(line.source_start),
                "source_end": list(line.source_end),
                "length_px": line.length_px,
                "length_m": line.length_m,
                "azimuth_deg": line.azimuth_deg,
                "orientation_deg": line.orientation_deg,
                "confidence": line.confidence,
                "algorithm_id": line.algorithm_id,
                "properties": dict(line.properties),
            }
            for line in result.segments
        ],
    }


def _result_from_dict(data):
    segments = tuple(
        LineSegment(
            pixel_start=tuple(item["pixel_start"]),
            pixel_end=tuple(item["pixel_end"]),
            source_start=tuple(item["source_start"]),
            source_end=tuple(item["source_end"]),
            length_px=float(item["length_px"]),
            length_m=item["length_m"],
            azimuth_deg=item["azimuth_deg"],
            orientation_deg=item["orientation_deg"],
            confidence=item["confidence"],
            algorithm_id=item["algorithm_id"],
            properties=item["properties"],
        )
        for item in data["segments"]
    )
    return DetectionResult(
        algorithm_id=data["algorithm_id"],
        parameters=data["parameters"],
        source_crs_wkt=data["source_crs_wkt"],
        source_transform=tuple(data["source_transform"]),
        source_shape=tuple(data["source_shape"]),
        segments=segments,
        postprocessing_parameters=data.get("postprocessing_parameters", {}),
    )


def save_project(path, state):
    """Atomically write a project document, keeping the old file on failure."""
    target = Path(path)
    document = {
        "format": "lineament-analyzer-project",
        "version": PROJECT_VERSION,
        "dem_path": str(state["dem_path"]),
        "display": state.get("display", {}),
        "class_visibility": state.get("class_visibility", {}),
        "detection_results": {
            key: _result_to_dict(value)
            for key, value in state.get("detection_results", {}).items()
        },
        "postprocessed_results": {
            key: _result_to_dict(value)
            for key, value in state.get("postprocessed_results", {}).items()
        },
    }
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def load_project(path):
    """Read and validate project JSON without opening its DEM."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "lineament-analyzer-project":
        raise ValueError("Not a Lineament Analyzer project")
    if document.get("version") != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {document.get('version')}")
    return {
        "dem_path": document["dem_path"],
        "display": document.get("display", {}),
        "class_visibility": document.get("class_visibility", {}),
        "detection_results": {
            key: _result_from_dict(value)
            for key, value in document.get("detection_results", {}).items()
        },
        "postprocessed_results": {
            key: _result_from_dict(value)
            for key, value in document.get("postprocessed_results", {}).items()
        },
    }
