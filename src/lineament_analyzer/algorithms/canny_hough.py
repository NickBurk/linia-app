"""Canny edge detection followed by probabilistic Hough line extraction."""

from dataclasses import dataclass

import numpy as np

from lineament_analyzer.algorithms.canny import CannyParameters, detect_edges
from lineament_analyzer.algorithms.hough import HoughParameters, detect_lines
from lineament_analyzer.algorithms.results import (
    DetectionResult,
    build_detection_result,
    parameters_dict,
)
from lineament_analyzer.algorithms.spatial import pixel_size_metres
from lineament_analyzer.algorithms.support import detector_support_mask, sample_mask_support
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.data.dem import Dem


@dataclass(frozen=True)
class CannyHoughParameters:
    """Canny thresholds are normalized 8-bit contrast values; lengths are metres."""

    canny_low_threshold: float = 30.0
    canny_high_threshold: float = 90.0
    aperture_size: int = 3
    l2_gradient: bool = True
    hough_threshold: int = 20
    min_segment_length_m: float = 100.0
    max_gap_m: float = 20.0

    def validate(self) -> None:
        self.canny.validate()
        self.hough.validate()

    @property
    def canny(self) -> CannyParameters:
        return CannyParameters(
            self.canny_low_threshold,
            self.canny_high_threshold,
            self.aperture_size,
            self.l2_gradient,
        )

    @property
    def hough(self) -> HoughParameters:
        return HoughParameters(
            self.hough_threshold,
            self.min_segment_length_m,
            self.max_gap_m,
        )


def detect(
    image: np.ndarray,
    valid: np.ndarray,
    parameters: CannyHoughParameters,
    *,
    pixel_size_m: float,
    cancel=None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel endpoints, edge support fractions and the Canny edge image."""
    parameters.validate()
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("Pixel size in metres must be positive and finite")
    check_cancel(cancel)
    if progress is not None:
        progress(0)
    edges, mask = detect_edges(image, valid, parameters.canny, cancel=cancel)
    check_cancel(cancel)
    if progress is not None:
        progress(10)
        progress(55)
    lines = detect_lines(
        edges,
        parameters.hough,
        pixel_size_m=pixel_size_m,
        cancel=cancel,
    )
    if progress is not None:
        progress(85)
    if not len(lines):
        return np.empty((0, 4)), np.empty(0), edges

    segments = lines
    erosion_radius = max(2, parameters.aperture_size // 2)
    safe_mask = detector_support_mask(mask, radius=erosion_radius)
    accepted, support_fractions = [], []
    for segment in segments:
        samples = sample_mask_support(safe_mask, segment)
        if samples is None:
            continue
        x, y = samples
        accepted.append(segment)
        support_fractions.append(float(np.count_nonzero(edges[y, x]) / len(x)))
    return (
        np.asarray(accepted, dtype=float).reshape(-1, 4),
        np.asarray(support_fractions, dtype=float),
        edges,
    )


class CannyHoughDetector:
    """Metric-aware detector that implements the common line-detector API."""

    algorithm_id = "canny-hough"

    def run(
        self,
        dem: Dem,
        parameters: CannyHoughParameters,
        *,
        progress=None,
        cancel=None,
    ) -> DetectionResult:
        parameters.validate()
        scale = pixel_size_metres(dem)
        pixels, supports, _ = detect(
            dem.values,
            dem.valid,
            parameters,
            pixel_size_m=scale,
            cancel=cancel,
            progress=progress,
        )
        result = build_detection_result(
            dem,
            pixels,
            algorithm_id=self.algorithm_id,
            parameters={**parameters_dict(parameters), "metric_pixel_size_m": scale},
            confidences=supports,
            properties=[{"edge_support_fraction": float(value)} for value in supports],
        )
        check_cancel(cancel)
        keep = tuple(
            segment
            for segment in result.segments
            if segment.length_m is not None and segment.length_m >= parameters.min_segment_length_m
        )
        if progress is not None:
            progress(100)
        return DetectionResult(
            algorithm_id=result.algorithm_id,
            parameters=result.parameters,
            source_crs_wkt=result.source_crs_wkt,
            source_transform=result.source_transform,
            source_shape=result.source_shape,
            segments=keep,
        )
