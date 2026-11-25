"""Canny edge detection followed by probabilistic Hough line extraction."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

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
        if (
            not np.isfinite(self.canny_low_threshold)
            or not np.isfinite(self.canny_high_threshold)
            or self.canny_low_threshold < 0
            or self.canny_high_threshold <= self.canny_low_threshold
            or self.canny_high_threshold > 255
        ):
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if self.aperture_size not in {3, 5, 7}:
            raise ValueError("Canny aperture size must be 3, 5 or 7")
        if self.hough_threshold < 1:
            raise ValueError("Hough threshold must be a positive integer")
        if not np.isfinite(self.min_segment_length_m) or self.min_segment_length_m <= 0:
            raise ValueError("Minimum segment length must be positive and finite")
        if not np.isfinite(self.max_gap_m) or self.max_gap_m < 0:
            raise ValueError("Maximum gap must be non-negative and finite")


def _normalized_raster(
    image: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.asarray(image, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("Canny-Hough requires a 2D image and matching validity mask")
    mask = mask & np.isfinite(values)
    if not np.any(mask):
        return None
    low, high = np.percentile(values[mask], [2, 98])
    if high <= low:
        low, high = float(np.min(values[mask])), float(np.max(values[mask]))
    if high <= low:
        return None
    values = np.where(mask, values, low)
    scaled = np.clip((values - low) / (high - low) * 255.0, 0, 255)
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    scaled[~mask] = scaled[tuple(indices[:, ~mask])]
    return scaled.astype(np.uint8), mask


def detect(
    image: np.ndarray,
    valid: np.ndarray,
    parameters: CannyHoughParameters,
    *,
    pixel_size_m: float,
    cancel=None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel endpoints, edge support fractions and normalized edges."""
    parameters.validate()
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("Pixel size in metres must be positive and finite")
    check_cancel(cancel)
    if progress is not None:
        progress(0)
    normalized = _normalized_raster(image, valid)
    if normalized is None:
        return np.empty((0, 4)), np.empty(0), np.zeros(np.shape(image), np.uint8)
    raster, mask = normalized
    check_cancel(cancel)
    if progress is not None:
        progress(10)
    edges = cv2.Canny(
        raster,
        parameters.canny_low_threshold,
        parameters.canny_high_threshold,
        apertureSize=parameters.aperture_size,
        L2gradient=parameters.l2_gradient,
    )
    check_cancel(cancel)
    if progress is not None:
        progress(55)
    min_length_px = max(1, round(parameters.min_segment_length_m / pixel_size_m))
    max_gap_px = max(0, round(parameters.max_gap_m / pixel_size_m))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=parameters.hough_threshold,
        minLineLength=min_length_px,
        maxLineGap=max_gap_px,
    )
    check_cancel(cancel)
    if progress is not None:
        progress(85)
    if lines is None:
        return np.empty((0, 4)), np.empty(0), edges

    segments = lines.reshape(-1, 4).astype(float)
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
