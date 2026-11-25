"""RFC 7946 GeoJSON with WGS84 geometry and auditable source attributes."""

import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from pyproj import Geod
from rasterio.warp import transform as reproject_points

from lineament_analyzer.algorithms.results import DetectionResult
from lineament_analyzer.data.dem import Dem


def export_geojson(
    path: str | Path,
    dem: Dem,
    segments: np.ndarray | DetectionResult | Sequence[DetectionResult],
    *,
    parameters=None,
) -> None:
    """Convert zero-based pixel-centre endpoints to WGS84 longitude/latitude.

    Orientations are axial bearings clockwise from geographic north in [0,180).
    Source CRS is retained as metadata. RFC 7946 geometry always uses WGS84.
    """
    if dem.crs is None:
        raise ValueError("A source CRS is required for GeoJSON export")
    structured = None
    if isinstance(segments, DetectionResult):
        structured = [segments]
    elif isinstance(segments, Sequence) and all(
        isinstance(result, DetectionResult) for result in segments
    ):
        structured = list(segments)
    if structured is not None:
        for result in structured:
            if result.source_crs_wkt != dem.crs.to_wkt():
                raise ValueError("Detection result CRS does not match the export DEM")
            transform = tuple(float(value) for value in tuple(dem.transform)[:6])
            if result.source_transform != transform or result.source_shape != dem.values.shape:
                raise ValueError("Detection result raster does not match the export DEM")
        records = [(result, segment) for result in structured for segment in result.segments]
        pixel_segments = np.asarray(
            [(*segment.pixel_start, *segment.pixel_end) for _, segment in records], dtype=float
        ).reshape(-1, 4)
    else:
        records = None
        pixel_segments = np.asarray(segments, dtype=float)
    if (
        pixel_segments.ndim != 2
        or pixel_segments.shape[1] != 4
        or not np.isfinite(pixel_segments).all()
    ):
        raise ValueError("Segments must be a finite Nx4 array")
    height, width = dem.values.shape
    features = []
    geod = Geod(ellps="WGS84")
    for index, (x1, y1, x2, y2) in enumerate(pixel_segments):
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
            raise ValueError("Segment endpoints outside raster")
        if records is None:
            first = dem.transform * (float(x1) + 0.5, float(y1) + 0.5)
            second = dem.transform * (float(x2) + 0.5, float(y2) + 0.5)
            algorithm_id = "lsd"
            method = "LSD"
            params = parameters or {}
            detector_segment = None
        else:
            result, detector_segment = records[index]
            first = detector_segment.source_start
            second = detector_segment.source_end
            algorithm_id = result.algorithm_id
            method = {
                "lsd": "LSD",
                "canny-hough": "Canny + Hough",
                "edlines": "EDLines",
            }.get(algorithm_id, algorithm_id)
            params = dict(result.parameters)
        longitude, latitude = reproject_points(
            dem.crs, "EPSG:4326", [first[0], second[0]], [first[1], second[1]]
        )
        coordinates = np.column_stack((longitude, latitude))
        if (
            not np.isfinite(coordinates).all()
            or np.any(np.abs(longitude) > 180)
            or np.any(np.abs(latitude) > 90)
        ):
            raise ValueError("Invalid WGS84 coordinates")
        azimuth, _, length = geod.inv(longitude[0], latitude[0], longitude[1], latitude[1])
        if detector_segment is not None:
            length = detector_segment.length_m
            azimuth = detector_segment.azimuth_deg
            orientation = detector_segment.orientation_deg
            length_px = detector_segment.length_px
            confidence = detector_segment.confidence
            detector_properties = dict(detector_segment.properties)
        else:
            orientation = azimuth % 180
            length_px = float(np.hypot(x2 - x1, y2 - y1))
            confidence = None
            detector_properties = {}
        features.append(
            {
                "type": "Feature",
                "id": index,
                "properties": {
                    "method": method,
                    "algorithm_id": algorithm_id,
                    "source_algorithm": algorithm_id,
                    "object_id": str(
                        detector_properties.get("object_id", f"{algorithm_id}-{index + 1:06d}")
                    ),
                    "class_id": detector_properties.get("class_id"),
                    "length_px": length_px,
                    "length_m": length,
                    "azimuth_deg": azimuth,
                    "orientation_deg": orientation,
                    "confidence": confidence,
                    "pixel_endpoints": [float(x1), float(y1), float(x2), float(y2)],
                    "source_endpoints": [list(first), list(second)],
                    "parameters": params,
                    "detection_parameters": params,
                    "postprocessing_parameters": dict(result.postprocessing_parameters)
                    if records is not None
                    else {},
                    "detection_properties": detector_properties,
                },
                "geometry": {"type": "LineString", "coordinates": coordinates.tolist()},
            }
        )
    content = json.dumps(
        {"type": "FeatureCollection", "source_crs": dem.crs.to_wkt(), "features": features},
        allow_nan=False,
        ensure_ascii=False,
    )
    destination = Path(path)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, suffix=".geojson", delete=False
    ) as output:
        temporary = Path(output.name)
        output.write(content)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
