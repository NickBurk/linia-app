"""RFC 7946 GeoJSON with WGS84 geometry and auditable source attributes."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from pyproj import Geod
from rasterio.warp import transform as reproject_points

from lineament_analyzer.data.dem import Dem


def export_geojson(path: str | Path, dem: Dem, segments: np.ndarray, *, parameters=None) -> None:
    """Convert zero-based pixel-centre endpoints to WGS84 longitude/latitude.

    Orientations are axial bearings clockwise from geographic north in [0,180).
    Source CRS is retained as metadata. RFC 7946 geometry always uses WGS84.
    """
    if dem.crs is None:
        raise ValueError("A source CRS is required for GeoJSON export")
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 2 or segments.shape[1] != 4 or not np.isfinite(segments).all():
        raise ValueError("Segments must be a finite Nx4 array")
    height, width = dem.values.shape
    features = []
    geod = Geod(ellps="WGS84")
    for index, (x1, y1, x2, y2) in enumerate(segments):
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
            raise ValueError("Segment endpoints outside raster")
        first = dem.transform * (float(x1) + 0.5, float(y1) + 0.5)
        second = dem.transform * (float(x2) + 0.5, float(y2) + 0.5)
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
        features.append(
            {
                "type": "Feature",
                "id": index,
                "properties": {
                    "method": "LSD",
                    "length_px": float(np.hypot(x2 - x1, y2 - y1)),
                    "length_m": length,
                    "orientation_deg": azimuth % 180,
                    "pixel_endpoints": [x1, y1, x2, y2],
                    "source_endpoints": [list(first), list(second)],
                    "parameters": parameters or {},
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
