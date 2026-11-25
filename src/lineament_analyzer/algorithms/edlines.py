"""EDLines through OpenCV's independent Edge Drawing implementation."""

from dataclasses import dataclass
from math import ceil

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

EDGE_OPERATORS = {
    "PREWITT": cv2.ximgproc.EDGE_DRAWING_PREWITT if hasattr(cv2, "ximgproc") else 0,
    "SOBEL": cv2.ximgproc.EDGE_DRAWING_SOBEL if hasattr(cv2, "ximgproc") else 1,
    "SCHARR": cv2.ximgproc.EDGE_DRAWING_SCHARR if hasattr(cv2, "ximgproc") else 2,
}


@dataclass(frozen=True)
class EdLinesParameters:
    """EDLines tuning; segment length is in metres and path controls are pixels."""

    sigma: float = 1.0
    gradient_threshold: int = 20
    anchor_threshold: int = 0
    min_path_length_px: int = 10
    min_segment_length_m: float = 100.0
    line_fit_error_threshold: float = 1.0
    max_error_threshold: float = 1.3
    max_distance_between_lines_px: float = 6.0
    nfa_validation: bool = True
    pf_mode: bool = False
    edge_operator: str = "SOBEL"
    scan_interval: int = 1

    def validate(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("EDLines sigma must be positive and finite")
        if self.gradient_threshold < 1 or self.anchor_threshold < 0:
            raise ValueError("EDLines gradient/anchor thresholds are invalid")
        if self.min_path_length_px < 1 or self.scan_interval < 1:
            raise ValueError("EDLines path length and scan interval must be positive")
        if not np.isfinite(self.min_segment_length_m) or self.min_segment_length_m <= 0:
            raise ValueError("Minimum segment length must be positive and finite")
        if (
            not np.isfinite(self.line_fit_error_threshold)
            or self.line_fit_error_threshold <= 0
            or not np.isfinite(self.max_error_threshold)
            or self.max_error_threshold <= 0
            or not np.isfinite(self.max_distance_between_lines_px)
            or self.max_distance_between_lines_px < 0
        ):
            raise ValueError("EDLines fitting and separation parameters are invalid")
        if self.edge_operator not in EDGE_OPERATORS:
            raise ValueError(f"Unsupported EDLines edge operator: {self.edge_operator}")
        if not isinstance(self.nfa_validation, bool) or not isinstance(self.pf_mode, bool):
            raise TypeError("EDLines mode options must be boolean")


def _prepare_image(image: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Normalize valid DEM values and extend nearby values through NoData holes."""
    values = np.asarray(image, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("EDLines requires a 2D image and matching validity mask")
    mask = mask & np.isfinite(values)
    if not np.any(mask):
        return None
    low, high = np.percentile(values[mask], [2, 98])
    if high <= low:
        low, high = float(np.min(values[mask])), float(np.max(values[mask]))
    if high <= low:
        return None
    values = np.where(mask, values, low)
    raster = np.clip((values - low) / (high - low) * 255, 0, 255)
    if np.any(~mask):
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        raster[~mask] = raster[tuple(indices[:, ~mask])]
    return raster.astype(np.uint8), mask


def _new_backend(parameters: EdLinesParameters, minimum_length_px: int):
    """Create the actual Edge Drawing backend; never substitute another detector."""
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "createEdgeDrawing"):
        raise RuntimeError("EDLines requires OpenCV contrib with cv2.ximgproc.createEdgeDrawing")
    backend = ximgproc.createEdgeDrawing()
    config = backend.Params()
    config.Sigma = parameters.sigma
    config.GradientThresholdValue = parameters.gradient_threshold
    config.AnchorThresholdValue = parameters.anchor_threshold
    config.MinPathLength = parameters.min_path_length_px
    config.MinLineLength = minimum_length_px
    config.LineFitErrorThreshold = parameters.line_fit_error_threshold
    config.MaxErrorThreshold = parameters.max_error_threshold
    config.MaxDistanceBetweenTwoLines = parameters.max_distance_between_lines_px
    config.NFAValidation = parameters.nfa_validation
    config.PFmode = parameters.pf_mode
    config.EdgeDetectionOperator = EDGE_OPERATORS[parameters.edge_operator]
    config.ScanInterval = parameters.scan_interval
    backend.setParams(config)
    return backend


def _detect_with_support(
    image: np.ndarray,
    valid: np.ndarray,
    parameters: EdLinesParameters,
    *,
    pixel_size_m: float,
    cancel=None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return supported EDLines endpoints and their normalized edge support."""
    parameters.validate()
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("Pixel size in metres must be positive and finite")
    check_cancel(cancel)
    if progress is not None:
        progress(0)
    prepared = _prepare_image(image, valid)
    if prepared is None:
        return np.empty((0, 4), dtype=float), np.empty(0, dtype=float)
    raster, mask = prepared
    minimum_length_px = max(1, round(parameters.min_segment_length_m / pixel_size_m))
    backend = _new_backend(parameters, minimum_length_px)
    check_cancel(cancel)
    backend.detectEdges(raster)
    check_cancel(cancel)
    if progress is not None:
        progress(55)
    lines = backend.detectLines()
    check_cancel(cancel)
    if progress is not None:
        progress(85)
    if lines is None:
        return np.empty((0, 4), dtype=float), np.empty(0, dtype=float)
    candidates = np.asarray(lines, dtype=float).reshape(-1, 4)
    edge_image = backend.getEdgeImage()
    if edge_image is None or edge_image.shape != mask.shape:
        edge_image = np.zeros(mask.shape, dtype=np.uint8)
    safe_radius = max(2, ceil(parameters.sigma * 3))
    safe_mask = detector_support_mask(mask, radius=safe_radius)
    accepted, support_fractions = [], []
    for line in candidates:
        samples = sample_mask_support(safe_mask, line)
        if samples is None:
            continue
        x, y = samples
        accepted.append(line)
        support_fractions.append(float(np.count_nonzero(edge_image[y, x]) / len(x)))
    return (
        np.asarray(accepted, dtype=float).reshape(-1, 4),
        np.asarray(support_fractions, dtype=float),
    )


def detect(
    image: np.ndarray,
    valid: np.ndarray,
    parameters: EdLinesParameters,
    *,
    pixel_size_m: float,
    cancel=None,
    progress=None,
) -> np.ndarray:
    """Return supported EDLines endpoints as a finite Nx4 pixel array."""
    segments, _ = _detect_with_support(
        image,
        valid,
        parameters,
        pixel_size_m=pixel_size_m,
        cancel=cancel,
        progress=progress,
    )
    return segments


class EdLinesDetector:
    """Standalone EDLines detector implementing the common result contract."""

    algorithm_id = "edlines"

    def run(
        self,
        dem: Dem,
        parameters: EdLinesParameters,
        *,
        progress=None,
        cancel=None,
    ) -> DetectionResult:
        parameters.validate()
        scale = pixel_size_metres(dem)
        pixels, support_fractions = _detect_with_support(
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
            confidences=support_fractions,
            properties=[{"edge_support_fraction": float(value)} for value in support_fractions],
        )
        check_cancel(cancel)
        segments = tuple(
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
            segments=segments,
        )
