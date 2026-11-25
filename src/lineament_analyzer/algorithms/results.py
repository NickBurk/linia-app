"""Shared detector contract and georeferenced line-segment results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pyproj import CRS as ProjCRS
from rasterio.warp import transform as reproject_points

from lineament_analyzer.data.dem import Dem

Point = tuple[float, float]


@dataclass(frozen=True)
class LineSegment:
    """One candidate segment in pixel-centre and source-CRS coordinates.

    ``azimuth_deg`` is the forward geodesic bearing from start to end in
    clockwise degrees from true north. ``orientation_deg`` is the axial bearing
    in [0, 180). Metric fields are ``None`` when the DEM has no assigned CRS.
    """

    pixel_start: Point
    pixel_end: Point
    source_start: Point
    source_end: Point
    length_px: float
    length_m: float | None
    azimuth_deg: float | None
    orientation_deg: float | None
    confidence: float | None
    algorithm_id: str
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class DetectionResult:
    """Detector output with common metadata and an immutable segment tuple."""

    algorithm_id: str
    parameters: Mapping[str, Any]
    source_crs_wkt: str | None
    segments: tuple[LineSegment, ...]

    def as_pixel_array(self) -> np.ndarray:
        """Return the legacy-compatible Nx4 pixel endpoint representation."""
        return np.asarray(
            [(*segment.pixel_start, *segment.pixel_end) for segment in self.segments],
            dtype=float,
        ).reshape(-1, 4)


@runtime_checkable
class LineDetector(Protocol):
    """Interface implemented by each independently configurable detector."""

    algorithm_id: str

    def run(
        self,
        dem: Dem,
        parameters: Any,
        *,
        progress=None,
        cancel=None,
    ) -> DetectionResult:
        """Detect candidate segments and return the shared geospatial result."""


def parameters_dict(parameters: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping into JSON-friendly parameter metadata."""
    if is_dataclass(parameters) and not isinstance(parameters, type):
        values = asdict(parameters)
    elif isinstance(parameters, Mapping):
        values = dict(parameters)
    else:
        raise TypeError("Detector parameters must be a dataclass or mapping")
    result = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"Parameter {key!r} must be finite")
            result[str(key)] = value
        else:
            raise TypeError(f"Parameter {key!r} is not JSON-compatible")
    return result


def build_detection_result(
    dem: Dem,
    segments: np.ndarray,
    *,
    algorithm_id: str,
    parameters: Any,
    confidences: Sequence[float | None] | None = None,
    properties: Sequence[Mapping[str, Any]] | None = None,
) -> DetectionResult:
    """Map Nx4 zero-based pixel-centre segments through the DEM affine/CRS.

    Coordinates use the existing PHASE 02 convention: pixel values identify
    centres, so the affine receives ``pixel + 0.5``. Length and bearing are
    WGS84 ellipsoidal geodesics between transformed endpoints. If CRS is
    unknown, source coordinates are still returned and metric fields are None.
    """
    lines = np.asarray(segments, dtype=float)
    if lines.size == 0:
        lines = np.empty((0, 4), dtype=float)
    if lines.ndim != 2 or lines.shape[1] != 4 or not np.isfinite(lines).all():
        raise ValueError("Segments must be a finite Nx4 array")
    if not algorithm_id or not isinstance(algorithm_id, str):
        raise ValueError("Algorithm ID must be a non-empty string")
    if dem.values.ndim != 2 or dem.valid.shape != dem.values.shape:
        raise ValueError("DEM values and validity mask must be matching 2D arrays")

    height, width = dem.values.shape
    if len(lines) and (
        np.any(lines[:, (0, 2)] < 0)
        or np.any(lines[:, (0, 2)] >= width)
        or np.any(lines[:, (1, 3)] < 0)
        or np.any(lines[:, (1, 3)] >= height)
    ):
        raise ValueError("Segment endpoints outside raster")
    if confidences is None:
        confidence_values: list[float | None] = [None] * len(lines)
    else:
        confidence_values = list(confidences)
        if len(confidence_values) != len(lines):
            raise ValueError("Confidence count must match segment count")
        for confidence in confidence_values:
            if confidence is not None and (not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Confidence values must be in [0, 1] or None")
    property_values = list(properties) if properties is not None else [{} for _ in lines]
    if len(property_values) != len(lines):
        raise ValueError("Properties count must match segment count")

    pixel_endpoints: list[Point] = []
    source_endpoints: list[Point] = []
    for x1, y1, x2, y2 in lines:
        pixel_endpoints.extend(((float(x1), float(y1)), (float(x2), float(y2))))
        source_endpoints.extend(
            (
                tuple(float(v) for v in dem.transform * (x1 + 0.5, y1 + 0.5)),
                tuple(float(v) for v in dem.transform * (x2 + 0.5, y2 + 0.5)),
            )
        )

    metric_values: list[tuple[float, float, float] | None] = [None] * len(lines)
    if dem.crs is not None and len(lines):
        crs = ProjCRS.from_user_input(dem.crs)
        source_x = [point[0] for point in source_endpoints]
        source_y = [point[1] for point in source_endpoints]
        longitude, latitude = reproject_points(crs, "EPSG:4326", source_x, source_y)
        geographic = np.column_stack((longitude, latitude))
        if (
            not np.isfinite(geographic).all()
            or np.any(np.abs(longitude) > 180)
            or np.any(np.abs(latitude) > 90)
        ):
            raise ValueError("DEM affine produced invalid geographic coordinates")
        geod = crs.get_geod()
        for index in range(len(lines)):
            azimuth, _, length = geod.inv(
                longitude[2 * index],
                latitude[2 * index],
                longitude[2 * index + 1],
                latitude[2 * index + 1],
            )
            forward = float(azimuth % 360)
            metric_values[index] = (float(length), forward, forward % 180)

    result_segments = []
    for index, (x1, y1, x2, y2) in enumerate(lines):
        metric = metric_values[index]
        result_segments.append(
            LineSegment(
                pixel_start=pixel_endpoints[2 * index],
                pixel_end=pixel_endpoints[2 * index + 1],
                source_start=source_endpoints[2 * index],
                source_end=source_endpoints[2 * index + 1],
                length_px=float(np.hypot(x2 - x1, y2 - y1)),
                length_m=metric[0] if metric else None,
                azimuth_deg=metric[1] if metric else None,
                orientation_deg=metric[2] if metric else None,
                confidence=float(confidence_values[index])
                if confidence_values[index] is not None
                else None,
                algorithm_id=algorithm_id,
                properties=dict(property_values[index]),
            )
        )
    return DetectionResult(
        algorithm_id=algorithm_id,
        parameters=parameters_dict(parameters),
        source_crs_wkt=dem.crs.to_wkt() if dem.crs is not None else None,
        segments=tuple(result_segments),
    )
