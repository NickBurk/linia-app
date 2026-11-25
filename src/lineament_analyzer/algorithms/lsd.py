"""LSD segment detection using OpenCV's image processing primitive."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_erosion

from lineament_analyzer.algorithms.results import DetectionResult, build_detection_result
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.data.dem import Dem


@dataclass(frozen=True)
class LsdParameters:
    min_length: float = 10.0
    scale: float = 0.8


def detect(image: np.ndarray, valid: np.ndarray, params: LsdParameters) -> np.ndarray:
    """Return Nx4 pixel endpoint coordinates of candidate segments."""
    if image.ndim != 2 or image.shape != valid.shape:
        raise ValueError("LSD requires a 2D image and matching validity mask")
    if not np.isfinite(params.min_length) or params.min_length < 0 or not 0 < params.scale <= 1:
        raise ValueError("Invalid LSD parameters")
    finite = np.isfinite(image) & valid
    if not np.any(finite):
        return np.empty((0, 4), dtype=float)
    low, high = np.percentile(image[finite], [2, 98])
    if high <= low:
        low, high = np.min(image[finite]), np.max(image[finite])
    if high <= low:
        return np.empty((0, 4), dtype=float)
    normalized = np.clip(
        (np.where(finite, image, low) - low) / max(high - low, 1e-12) * 255, 0, 255
    )
    raster = normalized.astype(np.uint8)
    raster[~finite] = 0
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, scale=params.scale).detect(raster)[0]
    if lines is None:
        return np.empty((0, 4), dtype=float)
    segments = lines.reshape(-1, 4).astype(float)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    # Reject segments whose full support touches NoData, including the detector's
    # interpolation footprint. Preserve support at the actual image border.
    safe = binary_erosion(finite, iterations=max(2, int(np.ceil(2 / params.scale))), border_value=1)
    accepted = []
    for segment in segments[lengths >= params.min_length]:
        x1, y1, x2, y2 = segment
        samples = max(2, int(np.ceil(np.hypot(x2 - x1, y2 - y1) * 2)) + 1)
        x = np.rint(np.linspace(x1, x2, samples)).astype(int)
        y = np.rint(np.linspace(y1, y2, samples)).astype(int)
        inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        if inside.all() and safe[y, x].all():
            accepted.append(segment)
    return np.asarray(accepted, dtype=float).reshape(-1, 4)


class LsdDetector:
    """Shared-interface adapter around the existing OpenCV LSD primitive."""

    algorithm_id = "lsd"

    def run(
        self, dem: Dem, parameters: LsdParameters, *, progress=None, cancel=None
    ) -> DetectionResult:
        check_cancel(cancel)
        if progress is not None:
            progress(0)
        pixels = detect(dem.values, dem.valid, parameters)
        check_cancel(cancel)
        result = build_detection_result(
            dem,
            pixels,
            algorithm_id=self.algorithm_id,
            parameters=parameters,
        )
        if progress is not None:
            progress(100)
        return result
