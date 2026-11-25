"""QGIS-compatible ESRI Shapefile export in the source DEM CRS."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import shapefile

from lineament_analyzer.algorithms.results import DetectionResult
from lineament_analyzer.data.dem import Dem
from lineament_analyzer.export.geometry import validate_coordinates, validate_result_grid


def export_shapefile(
    path: str | Path,
    dem: Dem,
    segments: np.ndarray | DetectionResult | Sequence[DetectionResult],
    *,
    parameters=None,
) -> None:
    """Write line features and source CRS sidecars for QGIS.

    Geometry is stored in the original raster CRS, avoiding a lossy round-trip
    from WGS84 and retaining the meaning of the source-coordinate endpoints.
    """
    if dem.crs is None:
        raise ValueError("A source CRS is required for Shapefile export")
    if isinstance(segments, DetectionResult):
        structured = [segments]
    elif isinstance(segments, Sequence) and all(
        isinstance(result, DetectionResult) for result in segments
    ):
        structured = list(segments)
    else:
        structured = None
    records = []
    if structured is not None:
        for result in structured:
            validate_result_grid(result, dem)
            for segment in result.segments:
                records.append((result, segment))
    else:
        pixel_segments = np.asarray(segments, dtype=float)
        if (
            pixel_segments.ndim != 2
            or pixel_segments.shape[1] != 4
            or not np.isfinite(pixel_segments).all()
        ):
            raise ValueError("Segments must be a finite Nx4 array")
        height, width = dem.values.shape
        for line in pixel_segments:
            x1, y1, x2, y2 = line
            if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
                raise ValueError("Segment endpoints outside raster")
            start = dem.transform * (float(x1) + 0.5, float(y1) + 0.5)
            end = dem.transform * (float(x2) + 0.5, float(y2) + 0.5)
            records.append((None, (line, start, end)))

    destination = Path(path)
    if destination.suffix.lower() != ".shp":
        destination = destination.with_suffix(".shp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lineament-shp-", dir=destination.parent) as temp:
        temporary_shp = Path(temp) / destination.name
        writer = shapefile.Writer(str(temporary_shp), shapeType=shapefile.POLYLINE, encoding="utf-8")
        writer.field("object_id", "C", size=80)
        writer.field("source_alg", "C", size=40)
        writer.field("detect_par", "C", size=254)
        writer.field("post_par", "C", size=254)
        writer.field("length_px", "N", size=16, decimal=3)
        writer.field("length_m", "N", size=18, decimal=3)
        writer.field("azimuth", "N", size=10, decimal=3)
        writer.field("orient_deg", "N", size=10, decimal=3)
        writer.field("conf", "N", size=10, decimal=5)
        writer.field("class_id", "C", size=100)
        metadata_records = []
        for index, (result, record) in enumerate(records, start=1):
            if result is None:
                line, start, end = record
                algorithm_id = "lsd"
                detect_params = parameters or {}
                post_params = {}
                length_px = float(np.hypot(line[2] - line[0], line[3] - line[1]))
                length_m = azimuth = orientation = confidence = None
                properties = {}
            else:
                segment = record
                start, end = segment.source_start, segment.source_end
                algorithm_id = result.algorithm_id
                detect_params = dict(result.parameters)
                post_params = dict(result.postprocessing_parameters)
                length_px = segment.length_px
                length_m = segment.length_m
                azimuth = segment.azimuth_deg
                orientation = segment.orientation_deg
                confidence = segment.confidence
                properties = dict(segment.properties)
            geometry = validate_coordinates(segment.coordinates if result else (start, end))
            object_id = str(properties.get("object_id", f"{algorithm_id}-{index:06d}"))
            writer.line([geometry.tolist()])
            writer.record(
                object_id,
                algorithm_id,
                _dbf_json(detect_params),
                _dbf_json(post_params),
                length_px,
                length_m,
                azimuth,
                orientation,
                confidence,
                str(properties.get("class_id", ""))[:100],
            )
            metadata_records.append({
                "object_id": object_id,
                "source_algorithm": algorithm_id,
                "detection_parameters": detect_params,
                "postprocessing_parameters": post_params,
                "length_px": length_px,
                "length_m": length_m,
                "azimuth_deg": azimuth,
                "orientation_deg": orientation,
                "confidence": confidence,
                "class_id": properties.get("class_id"),
                "properties": properties,
                "source_endpoints": [list(start), list(end)],
                "source_geometry": geometry.tolist(),
                "analysis": dict(result.analysis_metadata) if result else {},
            })
        writer.close()
        prj = temporary_shp.with_suffix(".prj")
        prj.write_text(dem.crs.to_wkt(), encoding="utf-8")
        temporary_shp.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
        temporary_shp.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "source_crs_wkt": dem.crs.to_wkt(),
                    "dem": str(dem.path),
                    "source_transform": [float(value) for value in tuple(dem.transform)[:6]],
                    "source_shape": list(dem.values.shape),
                    "features": metadata_records,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            sidecar = temporary_shp.with_suffix(suffix)
            if sidecar.exists():
                os.replace(sidecar, destination.with_suffix(suffix))
        os.replace(
            temporary_shp.with_suffix(".metadata.json"),
            destination.with_suffix(".metadata.json"),
        )


def _dbf_json(value) -> str:
    """Keep the DBF attribute valid and point to the companion when it is too long."""
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return serialized if len(serialized) <= 254 else "See metadata.json"
