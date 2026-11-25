"""GeoJSON line export in WGS84 per RFC 7946."""

import json
from pathlib import Path

import numpy as np
from rasterio.warp import transform as reproject_points

from lineament_analyzer.data.dem import Dem


def export_geojson(path: str | Path, dem: Dem, segments: np.ndarray) -> None:
    """Write pixel line segments as geographically referenced GeoJSON."""
    if dem.crs is None:
        raise ValueError("A source CRS is required for GeoJSON export")
    features = []
    for x1, y1, x2, y2 in segments:
        first = dem.transform @ (float(x1) + 0.5, float(y1) + 0.5)
        second = dem.transform @ (float(x2) + 0.5, float(y2) + 0.5)
        longitude, latitude = reproject_points(
            dem.crs, "EPSG:4326", [first[0], second[0]], [first[1], second[1]]
        )
        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": [[longitude[0], latitude[0]], [longitude[1], latitude[1]]],
            },
        })
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
